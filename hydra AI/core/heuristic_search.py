"""
Tác nhân B - Bộ ước lượng xác suất (Entropy Estimator) & Duyệt cây Heuristic
Sử dụng tìm kiếm theo chùm tia (Beam Search) kết hợp cắt tỉa nhánh (Pruning)
và mạng nơ-ron trọng số nhẹ để hội tụ về chuỗi mục tiêu với số bước tối thiểu.
"""

from typing import List, Tuple, Set, Optional, Callable
import heapq
from core.config import EstimatorConfig
from core.neural_evaluator import LightweightEvaluator

class SearchNode:
    """Nút trạng thái trong cây tìm kiếm."""
    def __init__(self, text: str, score: float, depth: int, parent=None):
        self.text = text
        self.score = score
        self.depth = depth
        self.parent = parent

    def __lt__(self, other):
        # Ưu tiên nút có điểm heuristic cao hơn (Max-Heap)
        return self.score > other.score

class HeuristicSearcher:
    """
    Tác nhân B: Không có quyền truy cập chuỗi gốc của Tác nhân A.
    Duyệt cây tìm kiếm heuristic có hướng dẫn từ Neural Evaluator.
    """
    def __init__(self, evaluator: Optional[LightweightEvaluator] = None, config: Optional[EstimatorConfig] = None):
        self.config = config or EstimatorConfig()
        self.evaluator = evaluator or LightweightEvaluator(self.config)
        self.visited_nodes: Set[str] = set()
        self.custom_suffixes: List[str] = []
        self.total_nodes_evaluated: int = 0

    def set_custom_suffixes(self, suffixes: List[str]):
        """Cấu hình thêm hậu tố ngữ cảnh từ metadata người dùng (năm sinh, từ khóa)."""
        self.custom_suffixes = suffixes

    def generate_successors(self, current_str: str) -> List[str]:
        """
        Sinh các nhánh biến thể (Successor states) bằng các phép toán:
        - Đổi chữ hoa/thường
        - Thay thế leetspeak
        - Thêm ký tự đặc biệt & số hậu tố
        """
        successors = []
        
        # 1. Biến thể hoa/thường
        if current_str.islower():
            successors.append(current_str.capitalize())
            successors.append(current_str.upper())

        # 2. Thay thế Leetspeak
        leet_map = {'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7'}
        for orig, sub in leet_map.items():
            if orig in current_str.lower():
                # Thay thế lần lượt từng ký tự hoặc toàn bộ
                variant = current_str.replace(orig, sub).replace(orig.upper(), sub)
                successors.append(variant)

        # 3. Mở rộng hậu tố (Suffix expansions)
        suffixes = ["123", "2024", "2025", "2026", "!", "@", "#", "@123", "!@#"]
        if self.custom_suffixes:
            for s in self.custom_suffixes:
                if s not in suffixes:
                    suffixes.append(s)

        for suf in suffixes:
            if not any(current_str.endswith(s) for s in ["!", "@", "#", "123"]):
                successors.append(f"{current_str}{suf}")
                successors.append(f"{current_str}_{suf}")
                successors.append(f"{current_str}@{suf}")

        # 4. Thêm ký tự đặc biệt cuối
        for sp in ["!", "!!", "??", "##"]:
            if not current_str.endswith(sp):
                successors.append(f"{current_str}{sp}")

        return list(set(successors))

    def search(self, 
               seed_roots: List[str], 
               verifier_oracle: Callable[[str], bool], 
               max_steps: int = 15) -> Tuple[bool, int, List[str], Optional[str]]:
        """
        Thực thi tìm kiếm Heuristic có cắt tỉa (Heuristic Search with Pruning).
        - seed_roots: Tập các từ khóa khởi đầu (từ ngữ cảnh/từ điển).
        - verifier_oracle: Hàm kiểm thử xem chuỗi có khớp mục tiêu hay không (Black-box).
        - max_steps: Giới hạn số bước duyệt.
        
        Trả về:
            (is_converged, steps_taken, explored_candidates, found_string)
        """
        self.visited_nodes.clear()
        self.total_nodes_evaluated = 0
        beam_width = self.config.beam_width

        # Khởi tạo Beam ban đầu từ seed_roots
        beam: List[SearchNode] = []
        for root in seed_roots:
            score = self.evaluator.evaluate_string(root)
            beam.append(SearchNode(text=root, score=score, depth=0))
        self.total_nodes_evaluated += len(seed_roots)

        # Sắp xếp beam theo điểm giảm dần
        beam.sort(key=lambda n: n.score, reverse=True)
        beam = beam[:beam_width]

        steps = 0
        explored_history = []

        while beam and steps < max_steps:
            steps += 1
            next_candidates: List[SearchNode] = []

            # Kiểm thử các ứng viên hiện tại trong beam
            for node in beam:
                explored_history.append(node.text)
                self.visited_nodes.add(node.text)

                # Gọi hàm thẩm định oracle
                if verifier_oracle(node.text):
                    return True, steps, explored_history, node.text

                # Sinh các nhánh tiếp theo
                children_text = self.generate_successors(node.text)
                # Lọc các nhánh đã duyệt
                unvisited = [c for c in children_text if c not in self.visited_nodes]
                
                # Kiểm tra mục tiêu sớm nếu có (Goal-test on generation)
                for child in unvisited:
                    if verifier_oracle(child):
                        explored_history.append(child)
                        return True, steps, explored_history, child

                if unvisited:
                    self.total_nodes_evaluated += len(unvisited)
                    # Đánh giá heuristic hàng loạt (Batch neural evaluation)
                    scores = self.evaluator.evaluate_batch(unvisited)
                    for text, sc in zip(unvisited, scores):
                        # Hàm heuristic kết hợp điểm neural và phạt độ sâu
                        heuristic_score = float(sc) - (node.depth + 1) * 0.05
                        next_candidates.append(
                            SearchNode(text=text, score=heuristic_score, depth=node.depth + 1, parent=node)
                        )

            if not next_candidates:
                break

            # Cắt tỉa (Pruning): Giữ lại top beam_width ứng viên có điểm cao nhất
            next_candidates.sort(key=lambda n: n.score, reverse=True)
            beam = next_candidates[:beam_width]

        return False, steps, explored_history, None
