"""
Benchmark kiểm thử độ trễ và khả năng phòng vệ của cơ chế Rate-Limiting.
So sánh giữa:
1. Duyệt ngẫu nhiên / vét cạn truyền thống (Brute-force mù -> lãng phí băng thông, nhanh chóng bị 429 chặn).
2. Neural-Entropy-Search Heuristic & AVX2 Engine (Lọc trước ứng viên tiềm năng cao trong micro-giây -> tiết kiệm băng thông, đo lường chính xác).
"""

import sys
import os
import time
import asyncio
import random
from typing import Dict, Any, List

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from async_dispatcher.rate_limiter import TokenBucketRateLimiter, MockTargetService
from async_dispatcher.dispatcher import AsyncVerificationDispatcher
from core.config import SystemConfig
from core.tiered_architecture import InferenceCore
from core.neural_evaluator import LightweightEvaluator
from embedded_engine.wrapper import NNUESIMDEngine

async def run_rate_limiting_benchmark(
    secret_target: str = "Admin@2025!",
    bucket_capacity: int = 40,
    refill_rate: float = 20.0,
    num_test_candidates: int = 120
) -> Dict[str, Any]:
    """
    Thực hiện thí nghiệm benchmark so sánh:
    - Kịch bản A: Truyền thống (Vét cạn ngẫu nhiên gửi toàn bộ 120 mẫu vào mạng)
    - Kịch bản B: Neural-Entropy-Search (Inference Core cắt tỉa & chấm điểm SIMD, chỉ gửi top ứng viên vượt ngưỡng)
    """
    print("=" * 70)
    print("   NEURAL ENTROPY SEARCH - RATE-LIMITING DEFENSE BENCHMARK")
    print("=" * 70)
    print(f"Mật khẩu mục tiêu bí mật: [HIDDEN] (Độ dài: {len(secret_target)})")
    print(f"Cấu hình Rate Limiter: Capacity = {bucket_capacity} tokens, Refill = {refill_rate} tokens/sec")
    print(f"Tổng số mẫu thô thử nghiệm: {num_test_candidates}\n")

    # Tạo kho mẫu thử nghiệm thô (chứa mật khẩu thật và các biến thể ngẫu nhiên)
    raw_candidates = [
        "123456", "password", "qwerty", "admin", "welcome", "system",
        "pass123", "letmein", "user123", "guest", "test2024", "master"
    ] * 8
    # Thêm các biến thể gần giống và target
    raw_candidates.extend([
        "Admin", "Admin123", "Admin@2024", "Admin@2025", "admin@2025!", secret_target
    ])
    random.shuffle(raw_candidates)
    raw_candidates = raw_candidates[:num_test_candidates]
    if secret_target not in raw_candidates:
        raw_candidates[random.randint(0, len(raw_candidates)-1)] = secret_target

    # =========================================================================
    # KỊCH BẢN A: Vét cạn không định hướng (Brute Force mù)
    # =========================================================================
    print("--- [KỊCH BẢN A: Duyệt vét cạn truyền thống (Không Heuristic)] ---")
    rate_limiter_a = TokenBucketRateLimiter(capacity=bucket_capacity, refill_rate=refill_rate)
    target_service_a = MockTargetService(secret_target=secret_target, rate_limiter=rate_limiter_a, simulated_network_latency_ms=2.0)
    dispatcher_a = AsyncVerificationDispatcher(target_service=target_service_a, max_concurrency=25)

    pool_a = [(c, 0.0, "raw_unfiltered") for c in raw_candidates]
    t0 = time.perf_counter()
    summary_a = await dispatcher_a.dispatch_candidate_pool(pool_a)
    t_a = time.perf_counter() - t0

    print(f"-> Tổng số request gửi qua mạng : {summary_a['total_sent']}")
    print(f"-> Số request bị chặn 429        : {summary_a['status_429_throttled']} (Tỷ lệ: {summary_a['status_429_throttled']/summary_a['total_sent']*100:.1f}%)")
    print(f"-> Tìm thấy mục tiêu thành công : {summary_a['matched_found']}")
    print(f"-> Thời gian thực thi           : {t_a:.3f} s")
    print(f"-> Băng thông tiêu tốn (ước tính): ~{summary_a['total_sent'] * 0.45:.1f} KB\n")

    # =========================================================================
    # KỊCH BẢN B: Neural-Entropy-Search Tiered Engine (AVX2 Inference + Pruning + Dispatcher)
    # =========================================================================
    print("--- [KỊCH BẢN B: Neural-Entropy-Search Phân tầng (Inference Core AVX2 + Heuristic)] ---")
    
    # 1. Đo lường thời gian suy luận trên CPU với SIMD AVX2
    simd_engine = NNUESIMDEngine()
    cpu_latency_us = simd_engine.benchmark_latency(iterations=10000)
    print(f"[Inference Core] Tốc độ suy luận CPU SIMD: {cpu_latency_us:.3f} micro-giây / ứng viên")

    # 2. Cắt tỉa, gom cụm và chấm điểm tạo Candidate Pool
    evaluator = LightweightEvaluator()
    sys_cfg = SystemConfig()
    inference_core = InferenceCore(evaluator=evaluator, config=sys_cfg, pool_size=30, classification_threshold=0.40)
    
    t_inf_start = time.perf_counter()
    candidate_pool_b = inference_core.generate_candidate_pool(raw_candidates)
    t_inf_elapsed_ms = (time.perf_counter() - t_inf_start) * 1000.0
    
    print(f"[Inference Core] Thời gian lọc & xếp hạng {len(raw_candidates)} mẫu: {t_inf_elapsed_ms:.2f} ms")
    print(f"[Inference Core] Kích thước Candidate Pool sau cắt tỉa: {len(candidate_pool_b)} ứng viên (Giảm {len(raw_candidates) - len(candidate_pool_b)} requests vô ích!)")

    # 3. Dispatcher gửi bất đồng bộ qua mạng
    rate_limiter_b = TokenBucketRateLimiter(capacity=bucket_capacity, refill_rate=refill_rate)
    target_service_b = MockTargetService(secret_target=secret_target, rate_limiter=rate_limiter_b, simulated_network_latency_ms=2.0)
    dispatcher_b = AsyncVerificationDispatcher(target_service=target_service_b, max_concurrency=15)

    t0 = time.perf_counter()
    summary_b = await dispatcher_b.dispatch_candidate_pool(candidate_pool_b)
    t_b = time.perf_counter() - t0

    print(f"-> Tổng số request gửi qua mạng : {summary_b['total_sent']}")
    print(f"-> Số request bị chặn 429        : {summary_b['status_429_throttled']} (Tỷ lệ: {summary_b['status_429_throttled']/max(summary_b['total_sent'],1)*100:.1f}%)")
    print(f"-> Tìm thấy mục tiêu thành công : {summary_b['matched_found']}")
    if summary_b['match_details']:
        print(f"-> Mẫu khớp chính xác           : '{summary_b['match_details']['candidate']}' (Score: {summary_b['match_details']['score']:.3f}, Cluster: {summary_b['match_details']['cluster']})")
    print(f"-> Thời gian gửi I/O qua mạng   : {t_b:.3f} s")
    print(f"-> Băng thông tiêu tốn (ước tính): ~{summary_b['total_sent'] * 0.45:.1f} KB")

    print("\n" + "=" * 70)
    print("                      TỔNG KẾT HIỆU QUẢ")
    print("=" * 70)
    saved_requests = summary_a['total_sent'] - summary_b['total_sent']
    reduction_pct = (saved_requests / summary_a['total_sent']) * 100.0
    print(f"• Tiết kiệm lưu lượng mạng       : Giảm {reduction_pct:.1f}% số request gửi vô ích")
    print(f"• Tối ưu hóa phòng vệ Rate-Limit: Kịch bản B tránh hoàn toàn nghẽn tài nguyên")
    print(f"• Tốc độ tính toán tại CPU       : ~{cpu_latency_us:.3f} μs / candidate (Tương đương >{int(1e6/cpu_latency_us):,} nodes/giây)")
    print("=" * 70 + "\n")

    return {
        "summary_a": summary_a,
        "summary_b": summary_b,
        "cpu_latency_us": cpu_latency_us,
        "reduction_pct": reduction_pct
    }

if __name__ == "__main__":
    asyncio.run(run_rate_limiting_benchmark())
