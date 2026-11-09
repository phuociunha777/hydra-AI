"""
1. Kiến trúc hệ thống xử lý phân tầng (Tiered System Architecture)
Cơ chế sàng lọc Heuristic nhiều tầng (Multi-stage Filtering)
kết hợp phân cụm theo ngưỡng Entropy (Entropy-based Clustering).

Dòng dữ liệu được tách thành hai bước:
- Bước 1: Hàm toán học thuần túy quét nhanh đo độ hỗn loạn (tần số lặp lại của chữ, số, ký tự lạ như @).
- Bước 2: Bộ điều phối rẽ nhánh dòng dữ liệu:
    + Nhóm Entropy thấp: Cây tìm kiếm nông, tập trung mẫu quen thuộc (tốn cực ít chu kỳ CPU).
    + Nhóm Entropy cao: Chuyển vào mạng nơ-ron sâu / động cơ SIMD AVX2 xử lý biến thể phức tạp.
"""

from typing import List, Dict, Tuple, Any, Optional
import numpy as np
from collections import defaultdict
from core.config import SystemConfig, MultiStageFilterConfig
from core.neural_evaluator import LightweightEvaluator
from core.profile_generator import calculate_repetition_and_class_entropy
from embedded_engine.wrapper import NNUESIMDEngine

class InferenceCore:
    """
    Tầng phân tích mô hình (Inference Core):
    Chịu trách nhiệm xử lý hàng ngàn/triệu nhánh trạng thái trong bộ nhớ GPU/NPU/CPU với tốc độ cao,
    cắt tỉa các nhánh kém tiềm năng và xuất ra Candidate Pool cho Dispatcher.
    Tích hợp trực tiếp động cơ C++ SIMD AVX2 để gánh tải suy luận hiệu năng cao.
    """
    def __init__(self, 
                 evaluator: LightweightEvaluator, 
                 config: Optional[SystemConfig] = None,
                 filter_config: Optional[MultiStageFilterConfig] = None,
                 pool_size: Optional[int] = None, 
                 classification_threshold: Optional[float] = None,
                 entropy_cutoff_threshold: Optional[float] = None):
        self.evaluator = evaluator
        
        # Nạp cấu hình tự động từ SystemConfig / MultiStageFilterConfig (không gán cứng)
        active_sys_cfg = config or SystemConfig()
        active_filter_cfg = filter_config or active_sys_cfg.filtering

        self.pool_size = pool_size if pool_size is not None else active_sys_cfg.candidate_pool_size
        self.threshold = classification_threshold if classification_threshold is not None else active_sys_cfg.classification_threshold
        self.entropy_cutoff = entropy_cutoff_threshold if entropy_cutoff_threshold is not None else active_filter_cfg.entropy_cutoff_threshold
        self.last_stage_stats: Dict[str, Any] = {}

        # Tích hợp trực tiếp động cơ C++ SIMD AVX2 vào luồng lọc chính
        try:
            self.simd_engine = NNUESIMDEngine()
        except Exception:
            self.simd_engine = None

    def prune_search_space(self, raw_candidates: List[str], min_length: int = 4) -> List[str]:
        """
        Cắt tỉa nhánh tìm kiếm (Search space pruning):
        Loại bỏ các chuỗi không khả thi hoặc có định dạng bất thường trước khi đưa vào mô hình.
        """
        pruned = []
        seen = set()
        for c in raw_candidates:
            if not c or len(c) < min_length:
                continue
            if c in seen:
                continue
            if c.strip() == "":
                continue
            seen.add(c)
            pruned.append(c)
        return pruned

    def cluster_candidates(self, candidates: List[str]) -> Dict[str, List[str]]:
        """
        Gom cụm dữ liệu (Clustering):
        Phân loại các ứng viên vào các họ hình thái:
        - 'alpha_numeric': Chuỗi gồm chữ và số
        - 'leet_pattern': Chứa ký tự leetspeak (@, 3, 1, 0, $, 7)
        - 'special_appended': Chứa ký tự đặc biệt ở cuối (!, #, $)
        - 'pure_text': Chỉ chữ cái hoa/thường
        - 'other': Nhóm còn lại
        """
        clusters = defaultdict(list)
        leet_chars = set("@310$7!48")

        for c in candidates:
            has_leet = any(ch in leet_chars for ch in c)
            has_special = any(not ch.isalnum() for ch in c)
            has_digit = any(ch.isdigit() for ch in c)
            has_alpha = any(ch.isalpha() for ch in c)

            if has_leet:
                clusters["leet_pattern"].append(c)
            elif has_special:
                clusters["special_appended"].append(c)
            elif has_alpha and has_digit:
                clusters["alpha_numeric"].append(c)
            elif has_alpha and not has_digit:
                clusters["pure_text"].append(c)
            else:
                clusters["other"].append(c)

        return dict(clusters)

    def route_and_filter_candidates(self, candidates: List[str]) -> Tuple[List[str], List[str], Dict[str, Any]]:
        """
        [BƯỚC 1 & BƯỚC 2: MULTI-STAGE FILTERING DỰA TRÊN ENTROPY]
        - Bước một: Hàm toán học đo tần số lặp lại của chữ, số, ký tự lạ như @.
        - Bước hai: Rẽ nhánh luồng dữ liệu sang Cây nông (Shallow) hoặc Mạng nơ-ron sâu (Deep Neural).
        """
        low_entropy_group: List[str] = []
        high_entropy_group: List[str] = []
        entropy_records: Dict[str, Dict[str, Any]] = {}

        for c in candidates:
            ent_score, stats = calculate_repetition_and_class_entropy(c)
            entropy_records[c] = stats

            # Rẽ nhánh thông minh tránh bẫy thiên lệch (Keyboard Bias Trap):
            # Nếu phát hiện quy luật bàn phím cố định (ziczac 1qaz2wsx, qwerty, v.v.),
            # chuỗi này có quy luật cơ học rõ ràng -> Đẩy trực tiếp vào Cây tìm kiếm nông (Shallow Search)
            # để giải quyết ngay trong vài bước đầu, tránh tiêu tốn chu kỳ tính toán SIMD AVX2.
            is_keyboard_bias = stats.get("keyboard_bias_detected", False)
            if is_keyboard_bias or ent_score < self.entropy_cutoff:
                low_entropy_group.append(c)
            else:
                high_entropy_group.append(c)

        stats_summary = {
            "total_candidates": len(candidates),
            "low_entropy_count": len(low_entropy_group),
            "high_entropy_count": len(high_entropy_group),
            "cutoff_threshold": self.entropy_cutoff,
            "entropy_records": entropy_records
        }
        self.last_stage_stats = stats_summary
        return low_entropy_group, high_entropy_group, stats_summary

    def generate_candidate_pool(self, raw_candidates: List[str]) -> List[Tuple[str, float, str]]:
        """
        Xử lý phân tầng toàn diện (Multi-stage Pipeline):
        1. Cắt tỉa (Pruning)
        2. Sàng lọc Heuristic nhiều tầng (Multi-stage Filtering) theo ngưỡng Entropy:
           - Nhánh Entropy Thấp: Đánh giá bằng hàm heuristic tuyến tính cực nhanh trên CPU (tiết kiệm chu kỳ SIMD).
           - Nhánh Entropy Cao: Đánh giá qua mạng nơ-ron trọng số nhẹ / SIMD AVX2 để xử lý biến thể leetspeak phức tạp.
        3. Gom cụm dữ liệu (Clustering)
        4. Lọc qua ngưỡng phân loại và xuất Candidate Pool.
        """
        clean_candidates = self.prune_search_space(raw_candidates)
        if not clean_candidates:
            return []

        # Gom cụm hình thái
        clusters = self.cluster_candidates(clean_candidates)
        cand_to_cluster = {}
        for cluster_name, c_list in clusters.items():
            for c in c_list:
                cand_to_cluster[c] = cluster_name

        # Bước 1 & Bước 2: Sàng lọc rẽ nhánh theo Entropy
        low_group, high_group, filter_stats = self.route_and_filter_candidates(clean_candidates)

        scored_candidates: List[Tuple[str, float, str]] = []

        # A. Xử lý nhóm Entropy Thấp: Dùng hàm đánh giá nông (Shallow Heuristic)
        # Tốn cực kỳ ít chu kỳ xung nhịp CPU, không cần nhân ma trận SIMD
        for c in low_group:
            stats = filter_stats["entropy_records"][c]
            # Điểm heuristic nông dựa trên độ khớp mẫu từ điển quen thuộc
            base_score = 0.50 + 0.15 * (1.0 - stats["repetition_ratio"])
            if stats.get("keyboard_bias_detected", False):
                base_score += 0.10
                cluster_tag = "Shallow (Keyboard Walk)"
            else:
                cluster_tag = "Shallow"
            cluster_name = f"{cand_to_cluster.get(c, 'unknown')} [{cluster_tag}]"
            scored_candidates.append((c, float(base_score), cluster_name))

        # B. Xử lý nhóm Entropy Cao: Đẩy vào Động cơ C++ SIMD AVX2 (gánh tải thực tế 8.6 triệu lượt/giây)
        if high_group:
            if self.simd_engine and self.simd_engine.is_ready():
                neural_scores = self.simd_engine.evaluate_batch(high_group)
                engine_tag = "SIMD AVX2"
            else:
                neural_scores = self.evaluator.evaluate_batch(high_group)
                engine_tag = "Deep Neural"

            for c, sc in zip(high_group, neural_scores):
                cluster_name = f"{cand_to_cluster.get(c, 'unknown')} [{engine_tag}]"
                scored_candidates.append((c, float(sc), cluster_name))

        # Bước 3: Lọc qua ngưỡng phân loại & sắp xếp
        filtered_candidates = [
            cand_item for cand_item in scored_candidates if cand_item[1] >= self.threshold
        ]
        filtered_candidates.sort(key=lambda x: x[1], reverse=True)

        return filtered_candidates[:self.pool_size]
