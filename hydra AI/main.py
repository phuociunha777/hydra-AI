"""
=============================================================================
                    NEURAL ENTROPY SEARCH SYSTEM
   Tối ưu hóa mô hình đánh giá không gian mật khẩu bằng Heuristic & Self-Play
=============================================================================
"""

import sys
import os
import argparse
import asyncio

# Đảm bảo UTF-8 cho console
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Thêm thư mục hiện tại vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.config import SystemConfig
from core.self_play import AdversarialSelfPlay
from core.profile_generator import ProfileGenerator, calculate_repetition_and_class_entropy
from core.neural_evaluator import LightweightEvaluator
from core.tiered_architecture import InferenceCore
from async_dispatcher.rate_limiter import TokenBucketRateLimiter, MockTargetService
from async_dispatcher.dispatcher import AsyncVerificationDispatcher
from embedded_engine.quantizer import NNUEQuantizer
from embedded_engine.wrapper import NNUESIMDEngine
from benchmarks.rate_limit_benchmark import run_rate_limiting_benchmark
from core.security_evaluator import SecurityAuditor

async def run_pipeline_demo():
    print("=" * 75)
    print("   NEURAL ENTROPY SEARCH - KIẾN TRÚC HỆ THỐNG XỬ LÝ PHÂN TẦNG (PIPELINE)")
    print("=" * 75)
    print("1. Tầng phân tích mô hình (Inference Core): Multi-stage Filtering + SIMD Ranking")
    print("2. Tầng kiểm thử bất đồng bộ (Async Verification Dispatcher): Non-blocking I/O\n")

    # 1. Khởi tạo tác nhân A và sinh mật khẩu mục tiêu
    generator = ProfileGenerator()
    target_pwd, entropy = generator.generate_candidate(keywords=["system", "security"], user_profile={"name": "hanoi", "year": "2025"})
    _, ent_details = calculate_repetition_and_class_entropy(target_pwd)

    l_cnt = ent_details.get('letter_count', 0)
    d_cnt = ent_details.get('digit_count', 0)
    s_cnt = ent_details.get('special_count', 0)
    l_freq = ent_details.get('letter_freq', l_cnt / max(len(target_pwd), 1))
    d_freq = ent_details.get('digit_freq', d_cnt / max(len(target_pwd), 1))
    s_freq = ent_details.get('special_freq', s_cnt / max(len(target_pwd), 1))

    print(f"[*] Mật khẩu mục tiêu được sinh ra: '{target_pwd}'")
    print(f"    • Độ hỗn loạn (Entropy): {entropy:.3f} bits")
    print(f"    • Phân bố: Chữ={l_cnt} ({l_freq*100:.1f}%), "
          f"Số={d_cnt} ({d_freq*100:.1f}%), "
          f"Ký tự lạ={s_cnt} ({s_freq*100:.1f}%)")
    print(f"    • Tỷ lệ lặp lại ký tự : {ent_details.get('repetition_ratio', 0.0)*100:.1f}% | "
          f"Quy luật bàn phím: {'CÓ (Bị phạt bias)' if ent_details.get('keyboard_bias_detected', False) else 'KHÔNG'}")

    # 2. Sinh tập ứng viên tìm kiếm ban đầu
    raw_candidates = [
        "admin", "root", "guest", "test", "password", "system", "hanoi", "matrix",
        "system123", "system_2024", "hanoi2025", "Hanoi@2025", "Neural@2026",
        "h@noi2025", "syst3m@2024", target_pwd
    ]
    # Mở rộng ngẫu nhiên
    for w in ["admin", "user", "neural", "vietnam"]:
        raw_candidates.extend([f"{w}{s}" for s in ["123", "2024", "!@#", "@2025"]])

    print(f"[*] Không gian mẫu đầu vào: {len(raw_candidates)} chuỗi")

    # 3. Tầng phân tích mô hình (Inference Core)
    evaluator = LightweightEvaluator()
    # Nạp trọng số đã train nếu có
    checkpoint_file = os.path.join(os.path.dirname(__file__), "models", "evaluator_weights.npz")
    if os.path.exists(checkpoint_file):
        evaluator.load(checkpoint_file)
        print(f"[*] Đã nạp ma trận trọng số tối ưu từ: models/evaluator_weights.npz")

    sys_config = SystemConfig()
    inference_core = InferenceCore(evaluator=evaluator, config=sys_config, pool_size=25, classification_threshold=0.35)
    candidate_pool = inference_core.generate_candidate_pool(raw_candidates)
    stage_stats = inference_core.last_stage_stats

    print(f"[*] Cơ chế sàng lọc Heuristic nhiều tầng (Multi-stage Filtering theo Entropy):")
    print(f"    • Ngưỡng rẽ nhánh Entropy Cutoff: {stage_stats.get('cutoff_threshold', sys_config.filtering.entropy_cutoff_threshold)}")
    print(f"    • Nhánh Entropy thấp -> Cây nông (Shallow): {stage_stats.get('low_entropy_count', 0)} chuỗi (Tiết kiệm xung nhịp CPU)")
    print(f"    • Nhánh Entropy cao  -> Động cơ C++ SIMD AVX2: {stage_stats.get('high_entropy_count', 0)} chuỗi")
    print(f"[*] Inference Core đã lọc thành công: {len(candidate_pool)} ứng viên vào Candidate Pool")
    print("\nTop 5 ứng viên có xác suất heuristic cao nhất:")
    for rank, (cand, score, cluster) in enumerate(candidate_pool[:5], 1):
        print(f"  [{rank}] {cand:<20} | Score: {score:.4f} | Họ cụm: {cluster}")

    # 4. Tầng kiểm thử bất đồng bộ (Async Dispatcher)
    print("\n[*] Kích hoạt Tầng kiểm thử bất đồng bộ (Non-blocking Dispatcher)...")
    limiter = TokenBucketRateLimiter(capacity=30, refill_rate=15.0)
    service = MockTargetService(secret_target=target_pwd, rate_limiter=limiter, simulated_network_latency_ms=2.0)
    dispatcher = AsyncVerificationDispatcher(target_service=service, max_concurrency=10)

    summary = await dispatcher.dispatch_candidate_pool(candidate_pool)
    print(f"[*] Kết quả Dispatcher:")
    print(f"  • Tổng request gửi đi         : {summary['total_sent']}")
    print(f"  • Bị giới hạn rate limit (429): {summary['status_429_throttled']}")
    print(f"  • Xác thực thành công (200)   : {summary['status_200_count']}")
    print(f"  • Thời gian hoàn tất          : {summary['duration_sec']:.3f} s")
    print(f"  • Thông lượng I/O             : {summary['throughput_req_per_sec']:.1f} req/s")
    if summary['match_details']:
        print(f"  • Mật khẩu đã tìm thấy        : '{summary['match_details']['candidate']}'")

    print("=" * 75 + "\n")

def run_simd_benchmark():
    print("=" * 75)
    print("   NEURAL ENTROPY SEARCH - C/C++ SIMD AVX2 EMBEDDED INFERENCE BENCHMARK")
    print("=" * 75)
    simd = NNUESIMDEngine()
    if not simd.is_ready():
        print("[!] Không thể nạp libnnue.dll")
        return

    print(f"[*] Hỗ trợ phần cứng Intel AVX2: {simd.has_avx2()}")
    
    samples = ["admin@2024", "P@ssw0rd2025!", "Neural_Engine", "root_system99", "h@n0i#2026"]
    print("\nKiểm tra đánh giá suy luận song song:")
    scores = simd.evaluate_batch(samples)
    for s, score in zip(samples, scores):
        print(f"  • Chuỗi: '{s:<18}' -> Heuristic Score: {score:.4f}")

    print("\nĐang thực hiện benchmark 100,000 lần suy luận trên CPU...")
    lat_us = simd.benchmark_latency(iterations=100000)
    inferences_per_sec = int(1e6 / lat_us)
    print(f"[*] Độ trễ trung bình: {lat_us:.3f} micro-giây / 1 lần suy luận")
    print(f"[*] Thông lượng xử lý : ~{inferences_per_sec:,} suy luận / giây trên 1 lõi CPU!")
    print("=" * 75 + "\n")

def run_audit_cli(args):
    """Thực thi đánh giá an toàn mật khẩu theo thông số CLI hoặc giao diện tương tác."""
    target = args.target
    profile = {}

    if not target:
        print("\n" + "=" * 75)
        print("    NEURAL ENTROPY SEARCH - GIAO DIỆN ĐÁNH GIÁ AN TOÀN CHUỖI MẬT KHẨU")
        print("=" * 75)
        try:
            target = input(">> Nhập chuỗi mật khẩu cần kiểm thử an toàn: ").strip()
            if not target:
                print("[!] Lỗi: Chuỗi mật khẩu không được rỗng.")
                return
            
            name = input(">> Nhập tên / username giả định (Enter để bỏ qua): ").strip()
            year = input(">> Nhập năm sinh / số gắn liền giả định (Enter để bỏ qua): ").strip()
            keywords = input(">> Nhập từ khóa liên quan giả định (phân cách bằng dấu phẩy): ").strip()
            company = input(">> Nhập tên tổ chức / công ty giả định (Enter để bỏ qua): ").strip()

            if name: profile["name"] = name
            if year: profile["year"] = year
            if keywords: profile["keywords"] = keywords
            if company: profile["company"] = company
        except (KeyboardInterrupt, EOFError):
            print("\nĐã hủy quy trình đánh giá.")
            return
    else:
        if args.name: profile["name"] = args.name
        if args.year: profile["year"] = args.year
        if args.keywords: profile["keywords"] = args.keywords
        if args.company: profile["company"] = args.company

    auditor = SecurityAuditor()
    result = auditor.evaluate_target(
        target=target, 
        profile_metadata=profile, 
        max_steps=args.max_steps
    )
    auditor.print_audit_report(result)

def main():
    parser = argparse.ArgumentParser(description="Neural-Entropy-Search - Password Search Space Optimization & Audit Engine")
    parser.add_argument("--mode", type=str, default="all",
                        choices=["all", "self_play", "rate_limit", "benchmark_simd", "export_nnue", "pipeline", "audit"],
                        help="Chế độ thực thi")
    parser.add_argument("--episodes", type=int, default=10, help="Số vòng tự chơi đối kháng (Self-Play)")
    parser.add_argument("--fresh", action="store_true", help="Bắt đầu train lại từ đầu (không nạp checkpoint trước)")

    # Các tham số dành cho chế độ đánh giá an toàn (audit)
    parser.add_argument("--target", type=str, default=None, help="Chuỗi văn bản mục tiêu cần kiểm thử an toàn")
    parser.add_argument("--name", type=str, default=None, help="Tên người dùng / tài khoản giả định")
    parser.add_argument("--year", type=str, default=None, help="Năm sinh / mốc thời gian giả định")
    parser.add_argument("--keywords", type=str, default=None, help="Từ khóa ngữ cảnh liên quan (ngăn cách bởi dấu phẩy)")
    parser.add_argument("--company", type=str, default=None, help="Tên tổ chức / dịch vụ giả định")
    parser.add_argument("--max_steps", type=int, default=25, help="Giới hạn số bước duyệt tối đa của Heuristic Tree Search")

    args = parser.parse_args()

    resume = not args.fresh

    if args.mode == "audit":
        run_audit_cli(args)
        return

    if args.mode in ["all", "self_play"]:
        from benchmarks.run_self_play import run_self_play_training
        run_self_play_training(episodes=args.episodes, resume=resume, export_weights=True)

    if args.mode in ["all", "benchmark_simd"]:
        run_simd_benchmark()

    if args.mode in ["all", "pipeline"]:
        asyncio.run(run_pipeline_demo())

    if args.mode in ["all", "rate_limit"]:
        asyncio.run(run_rate_limiting_benchmark())

    if args.mode == "export_nnue":
        ev = LightweightEvaluator()
        quantizer = NNUEQuantizer(ev)
        h_path = os.path.join(os.path.dirname(__file__), "embedded_engine", "nnue_weights.h")
        b_path = os.path.join(os.path.dirname(__file__), "embedded_engine", "nnue_weights.bin")
        quantizer.export_c_header(h_path)
        quantizer.export_binary(b_path)

if __name__ == "__main__":
    main()
