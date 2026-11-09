"""
Hàm đánh giá an toàn mật khẩu (Security Audit & Resilience Evaluator)
Đầu vào:
- target string (chuỗi mật khẩu mục tiêu cần kiểm thử)
- profile metadata (thông tin ngữ cảnh giả định: tên, năm sinh, từ khóa, công ty, v.v.)
Đầu ra:
- Shannon Entropy thực tế và Entropy cấu trúc (tần số lặp chữ, số, ký tự lạ @)
- Mô phỏng Heuristic Tree Search: số bước duyệt (steps / guesses) và số node duyệt trong bộ nhớ
- Xếp loại độ phức tạp chuỗi theo thang đo bảo mật 5 cấp độ (NIST/OWASP benchmark)
"""

import os
import sys
import time
from typing import Dict, Any, Optional, List, Tuple

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.config import SystemConfig, EstimatorConfig
from core.profile_generator import calculate_repetition_and_class_entropy
from core.neural_evaluator import LightweightEvaluator
from core.heuristic_search import HeuristicSearcher
from embedded_engine.wrapper import NNUESIMDEngine

class SecurityAuditor:
    """
    Hệ thống đánh giá an toàn và đo lường độ chịu lỗi của chuỗi mật khẩu
    dựa trên mô phỏng tìm kiếm Heuristic & Entropy nhiều tầng.
    """
    def __init__(self, config: Optional[SystemConfig] = None, weights_path: Optional[str] = None):
        self.config = config or SystemConfig()
        self.evaluator = LightweightEvaluator(self.config.estimator)
        
        # Nạp trọng số đã học từ checkpoint nếu có
        if weights_path is None:
            default_weights = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                "models", "evaluator_weights.npz"
            )
            if os.path.exists(default_weights):
                weights_path = default_weights

        if weights_path and os.path.exists(weights_path):
            self.evaluator.load(weights_path)

        # Cấu hình searcher với beam_width phù hợp cho audit sâu
        audit_estimator_cfg = EstimatorConfig(
            beam_width=12,
            max_search_depth=10,
            convergence_threshold_n=self.config.estimator.convergence_threshold_n
        )
        self.searcher = HeuristicSearcher(self.evaluator, audit_estimator_cfg)
        self.simd_engine = NNUESIMDEngine()

    def evaluate_target(self, 
                        target: str, 
                        profile_metadata: Optional[Dict[str, str]] = None, 
                        max_steps: int = 25) -> Dict[str, Any]:
        """
        Thực hiện đánh giá toàn diện độ an toàn của chuỗi văn bản mục tiêu.
        """
        if not target:
            raise ValueError("Chuỗi mục tiêu (target string) không được rỗng.")

        profile = profile_metadata or {}
        start_t = time.perf_counter()

        # 1. Tính toán Shannon Entropy thực tế và Entropy cấu trúc
        entropy_score, ent_stats = calculate_repetition_and_class_entropy(target)

        # 2. Xây dựng tập hạt giống (Seed Roots) và hậu tố (Suffixes) từ Profile Metadata
        seeds: List[str] = []
        custom_suffixes: List[str] = []

        # Trích xuất từ ngữ cảnh
        if "name" in profile and profile["name"]:
            seeds.append(profile["name"].lower())
            seeds.append(profile["name"].capitalize())
            custom_suffixes.append(profile["name"].lower())

        if "year" in profile and profile["year"]:
            custom_suffixes.append(str(profile["year"]))

        if "keywords" in profile and profile["keywords"]:
            kws = [k.strip().lower() for k in str(profile["keywords"]).split(",") if k.strip()]
            seeds.extend(kws)

        if "company" in profile and profile["company"]:
            seeds.append(profile["company"].lower())
            custom_suffixes.append(profile["company"].lower())

        # Bổ sung các từ khóa cốt lõi thông dụng nếu chưa đủ
        for kw in ["admin", "root", "user", "pass", "system", "vietnam", "hanoi"]:
            if kw not in seeds:
                seeds.append(kw)

        # Cấu hình hậu tố cho searcher
        self.searcher.set_custom_suffixes(custom_suffixes)

        # 3. Mô phỏng tìm kiếm Heuristic (Tác nhân B duyệt cây)
        def verifier_oracle(candidate_str: str) -> bool:
            return candidate_str == target

        converged, steps_taken, explored_candidates, found_str = self.searcher.search(
            seed_roots=seeds,
            verifier_oracle=verifier_oracle,
            max_steps=max_steps
        )
        elapsed_sec = time.perf_counter() - start_t
        nodes_explored = self.searcher.total_nodes_evaluated

        # 4. Ước tính số bước thử (Estimated Guesses / Search Space Complexity)
        if converged:
            estimated_guesses = len(explored_candidates)
        else:
            # Nếu không hội tụ trong ngân sách max_steps:
            # Ước tính không gian trạng thái dựa trên beam width và entropy
            estimated_guesses = int(nodes_explored * (2.0 ** max(entropy_score, 1.0)))

        # 5. Phân hạng an toàn (Security Complexity Rating) theo thang đo chuẩn
        rating_info = self._classify_security_level(
            target=target,
            converged=converged,
            steps_taken=steps_taken,
            entropy=entropy_score,
            ent_stats=ent_stats
        )

        # 6. Kiểm tra tốc độ suy luận SIMD CPU AVX2 trên chuỗi
        simd_score = 0.0
        if self.simd_engine.is_ready():
            try:
                simd_score = self.simd_engine.evaluate_string(target)
            except Exception:
                pass

        result = {
            "target": target,
            "target_length": len(target),
            "profile_context": profile,
            "shannon_char_entropy": ent_stats["char_entropy"],
            "class_diversity_entropy": ent_stats["class_entropy"],
            "composite_entropy": entropy_score,
            "converged": converged,
            "steps_to_converge": steps_taken if converged else None,
            "max_steps_budget": max_steps,
            "nodes_in_memory": nodes_explored,
            "total_explored_candidates": len(explored_candidates),
            "estimated_guesses": estimated_guesses,
            "elapsed_ms": round(elapsed_sec * 1000.0, 2),
            "simd_heuristic_score": round(simd_score, 4),
            "max_digit_run": ent_stats["max_digit_run"],
            "max_letter_run": ent_stats["max_letter_run"],
            "transition_rate": ent_stats["transition_rate"],
            "keyboard_bias_detected": ent_stats["keyboard_bias_detected"],
            "security_rating": rating_info["level"],
            "security_label": rating_info["label"],
            "security_stars": rating_info["stars"],
            "risk_assessment": rating_info["risk"],
            "vulnerabilities": rating_info["vulnerabilities"],
            "recommendations": rating_info["recommendations"]
        }
        return result

    def _classify_security_level(self, 
                                 target: str, 
                                 converged: bool, 
                                 steps_taken: int, 
                                 entropy: float, 
                                 ent_stats: Dict[str, Any]) -> Dict[str, Any]:
        """
        Xếp loại độ phức tạp chuỗi theo thang đo bảo mật 5 cấp độ:
        - Mức 1: CỰC KỲ NGUY HIỂM / RẤT YẾU (Critical / Very Weak)
        - Mức 2: YẾU / DỄ BẮT BÀI (Weak / Vulnerable)
        - Mức 3: TRUNG BÌNH (Moderate)
        - Mức 4: MẠNH (Strong)
        - Mức 5: RẤT MẠNH / CHUẨN AN TOÀN DOANH NGHIỆP (Hardened / Enterprise)
        """
        vulnerabilities = []
        recommendations = []

        if ent_stats["max_digit_run"] >= 4:
            vulnerabilities.append(f"Cụm chữ số lặp dồn dập liên tiếp ({ent_stats['max_digit_run']} số liền kề).")
            recommendations.append("Nên đan xen các ký tự chữ và ký tự lạ vào giữa các chữ số.")

        if ent_stats["max_letter_run"] >= 6:
            vulnerabilities.append(f"Dải chữ cái liền mạch quá dài ({ent_stats['max_letter_run']} chữ liên tiếp).")

        if ent_stats["keyboard_bias_detected"]:
            vulnerabilities.append("Chứa quy luật gõ bàn phím liên tiếp (ziczac / chuỗi số tiến lùi).")
            recommendations.append("Tránh các chuỗi phím gần nhau trên bàn phím (như qwerty, 1qaz, 123456).")

        if ent_stats["special_count"] == 0:
            vulnerabilities.append("Không có ký tự đặc biệt/ký tự lạ nào (@, #, $, !).")
            recommendations.append("Bổ sung thêm ít nhất 1 hoặc 2 ký tự đặc biệt lạ (@, #, $, !).")

        if len(target) < 8:
            vulnerabilities.append(f"Độ dài quá ngắn ({len(target)} ký tự, chuẩn an toàn tối thiểu là 10-12 ký tự).")
            recommendations.append("Tăng độ dài chuỗi lên từ 12 ký tự trở lên.")

        # Đánh giá cấp độ dựa trên sự kết hợp giữa:
        # Số bước hội tụ Heuristic + Chỉ số Entropy thực tế
        if converged and steps_taken <= 2:
            return {
                "level": 1,
                "label": "RẤT YẾU (CỰC KỲ NGUY HIỂM)",
                "stars": "★☆☆☆☆",
                "risk": "Cực kỳ cao. Bị mô hình Heuristic bắt bài ngay lập tức trong 1-2 bước duyệt.",
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations or ["Thay đổi toàn bộ mật khẩu ngay lập tức sang chuỗi ngẫu nhiên."]
            }
        elif (converged and steps_taken <= 5) or entropy < 1.8:
            return {
                "level": 2,
                "label": "YẾU (DỄ BỊ BẮT BÀI)",
                "stars": "★★☆☆☆",
                "risk": "Cao. Nằm trong vùng ngưỡng phạt Minimax (N <= 5 bước), dễ bị đoán từ thông tin cá nhân/ngữ cảnh.",
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations or ["Tăng tính đan xen giữa chữ, số và ký tự lạ @."]
            }
        elif (converged and steps_taken <= 12) or entropy < 2.8:
            return {
                "level": 3,
                "label": "TRUNG BÌNH (TƯƠNG ĐỐI AN TOÀN)",
                "stars": "★★★☆☆",
                "risk": "Trung bình. Kháng cự được tìm kiếm nông nhưng vẫn bị dò quét ra nếu lộ ngữ cảnh người dùng.",
                "vulnerabilities": vulnerabilities,
                "recommendations": recommendations or ["Nâng cao độ dài chuỗi và biến đổi leetspeak bất quy tắc hơn."]
            }
        elif not converged and entropy < 3.8:
            return {
                "level": 4,
                "label": "MẠNH (ĐỘ BẢO MẬT CAO)",
                "stars": "★★★★☆",
                "risk": "Thấp. Cây tìm kiếm Heuristic không thể hội tụ trong ngân sách duyệt thông thường.",
                "vulnerabilities": vulnerabilities or ["Không phát hiện điểm yếu rõ rệt."],
                "recommendations": recommendations or ["Chuỗi có độ phức tạp tốt, có thể duy trì sử dụng."]
            }
        else:
            return {
                "level": 5,
                "label": "RẤT MẠNH (CHUẨN DOANH NGHIỆP)",
                "stars": "★★★★★",
                "risk": "Gần như bằng 0. Hoàn toàn vô hiệu hóa cả phương pháp tìm kiếm vét cạn lẫn suy luận Heuristic có định hướng.",
                "vulnerabilities": vulnerabilities or ["Không có điểm yếu."],
                "recommendations": ["Chuỗi tối ưu, đạt chuẩn an toàn cao trong phân tích hệ thống."]
            }

    def print_audit_report(self, result: Dict[str, Any]):
        """In giao diện bảng điều khiển CLI/UI đánh giá an toàn chuyên nghiệp."""
        sep = "=" * 75
        dash = "-" * 75

        print("\n" + sep)
        print("     NEURAL ENTROPY SEARCH - BÁO CÁO PHÂN TÍCH AN TOÀN & ĐỘ PHỨC TẠP CHUỖI")
        print(sep)
        print(f"[*] Chuỗi văn bản kiểm tra : '{result['target']}' (Độ dài: {result['target_length']} ký tự)")
        
        ctx_str = ", ".join(f"{k}='{v}'" for k, v in result['profile_context'].items() if v)
        print(f"[*] Ngữ cảnh giả định       : {ctx_str if ctx_str else '(Không cung cấp ngữ cảnh giả định)'}")

        print(dash)
        print("1. KẾT QUẢ ĐO LƯỜNG ĐỘ HỖN LOẠN (ENTROPY ANALYSIS)")
        print(f"  • Shannon Entropy ký tự   : {result['shannon_char_entropy']:.3f} bits")
        print(f"  • Độ hỗn loạn phân lớp    : {result['class_diversity_entropy']:.3f} bits")
        print(f"  • Entropy cấu trúc thực tế: {result['composite_entropy']:.3f} bits")
        print(f"  • Dải số lặp dồn dập      : {result['max_digit_run']} số liên tiếp")
        print(f"  • Tần suất xen kẽ chuyển đổi: {result['transition_rate'] * 100:.1f}%")
        print(f"  • Bẫy quy luật bàn phím   : {'CÓ (Bị phạt bias)' if result['keyboard_bias_detected'] else 'KHÔNG (Tự nhiên)'}")

        print(dash)
        print("2. KẾT QUẢ MÔ PHỎNG TÌM KIẾM HEURISTIC (SEARCH CONVERGENCE)")
        if result['converged']:
            print(f"  • Trạng thái hội tụ       : [THÀNH CÔNG] Đã bị cây tìm kiếm tìm thấy!")
            print(f"  • Số bước duyệt tối thiểu : {result['steps_to_converge']} bước (Ngân sách: {result['max_steps_budget']})")
        else:
            print(f"  • Trạng thái hội tụ       : [THẤT BẠI] Mục tiêu nằm ngoài tầm với của cây tìm kiếm!")
            print(f"  • Số bước duyệt thử nghiệm: > {result['max_steps_budget']} bước (Vượt ngưỡng ngân sách)")

        print(f"  • Số node duyệt trong RAM : {result['nodes_in_memory']:,} nodes")
        print(f"  • Tổng số mẫu ứng viên    : {result['total_explored_candidates']:,} mẫu")
        print(f"  • Ước tính không gian thử : ~{result['estimated_guesses']:,} lần đoán")
        print(f"  • Thời gian phân tích CPU : {result['elapsed_ms']} ms")
        if result['simd_heuristic_score'] > 0:
            print(f"  • Điểm Heuristic SIMD AVX2: {result['simd_heuristic_score']}")

        print(dash)
        print("3. XẾP LOẠI BẢO MẬT & ĐÁNH GIÁ ĐỘ CHỊU LỖI")
        print(f"  • Thang đo phức tạp       : [{result['security_stars']}] - {result['security_label']}")
        print(f"  • Mức độ rủi ro           : {result['risk_assessment']}")

        if result['vulnerabilities']:
            print("  • Các lỗ hổng phát hiện   :")
            for v in result['vulnerabilities']:
                print(f"    - [!] {v}")

        if result['recommendations']:
            print("  • Khuyến nghị nâng cấp    :")
            for r in result['recommendations']:
                print(f"    - [✓] {r}")

        print(sep + "\n")
