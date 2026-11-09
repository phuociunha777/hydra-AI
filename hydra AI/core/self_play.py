"""
Cơ chế huấn luyện đối kháng (Adversarial Self-Play)
Sử dụng nguyên lý Minimax mô phỏng hai tác nhân A và B để đo lường và tối ưu hóa:
- Nếu Tác nhân B hội tụ <= N bước (N <= 5): Tác nhân A nhận phản hồi phạt.
- Nếu vượt quá ngưỡng N bước: Tác nhân B cập nhật trọng số mạng neural.
"""

import os
import sys
import json
from typing import List, Dict, Any, Tuple
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from core.config import SystemConfig
from core.profile_generator import ProfileGenerator, calculate_repetition_and_class_entropy
from core.heuristic_search import HeuristicSearcher
from core.neural_evaluator import LightweightEvaluator

class AdversarialSelfPlay:
    """
    Vòng lặp huấn luyện đối kháng giữa Generator (Tác nhân A) và Searcher (Tác nhân B).
    """
    def __init__(self, config: SystemConfig = SystemConfig()):
        self.config = config
        self.generator = ProfileGenerator(config.generator)
        self.evaluator = LightweightEvaluator(config.estimator)
        self.searcher = HeuristicSearcher(self.evaluator, config.estimator)
        self.history: List[Dict[str, Any]] = []
        self.total_episodes_trained = 0

    def save_checkpoint(self, checkpoint_dir: str = "models"):
        """Lưu toàn bộ trọng số mạng neural và trạng thái của cả Agent A & B."""
        os.makedirs(checkpoint_dir, exist_ok=True)
        weights_path = os.path.join(checkpoint_dir, "evaluator_weights.npz")
        meta_path = os.path.join(checkpoint_dir, "self_play_state.json")

        # 1. Lưu trọng số mạng neural
        self.evaluator.save(weights_path)

        # 2. Lưu trạng thái Agent A và tổng số vòng đấu
        state = {
            "total_episodes_trained": self.total_episodes_trained,
            "generator_state": self.generator.get_state()
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        print(f"[Self-Play] Đã lưu checkpoint thành công tại: {checkpoint_dir}/")

    def load_checkpoint(self, checkpoint_dir: str = "models") -> bool:
        """Nạp lại trọng số và trạng thái từ các lần huấn luyện trước."""
        weights_path = os.path.join(checkpoint_dir, "evaluator_weights.npz")
        meta_path = os.path.join(checkpoint_dir, "self_play_state.json")

        if not os.path.exists(weights_path) or not os.path.exists(meta_path):
            return False

        try:
            # 1. Nạp trọng số mạng neural
            loaded_w = self.evaluator.load(weights_path)
            # 2. Nạp trạng thái Agent A
            with open(meta_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            self.total_episodes_trained = state.get("total_episodes_trained", 0)
            self.generator.load_state(state.get("generator_state", {}))
            print(f"[Self-Play] Đã nạp checkpoint trước đó! (Đã tích lũy {self.total_episodes_trained} vòng đấu, Mutation: {self.generator.mutation_prob:.2f})")
            return loaded_w
        except Exception as e:
            print(f"[Self-Play] Lỗi khi nạp checkpoint: {e}")
            return False

    def run_episode(self, episode_idx: int) -> Dict[str, Any]:
        """
        Thực thi một vòng đấu đối kháng:
        1. Tác nhân A sinh mật khẩu mục tiêu bí mật T từ ngữ cảnh.
        2. Tác nhân B chỉ nhận seed ban đầu và tìm cách hội tụ về T mà không biết T.
        3. Đánh giá số bước hội tụ so với ngưỡng N (mặc định N = 5).
        4. Cập nhật trọng số theo luật đối kháng Minimax.
        """
        N = self.config.estimator.convergence_threshold_n

        # 1. Tác nhân A sinh mục tiêu
        target, target_entropy = self.generator.generate_candidate()

        # Tạo oracle thẩm định chuỗi đối với Tác nhân B (Black-box verification)
        def verifier_oracle(candidate_str: str) -> bool:
            return candidate_str == target

        # Tác nhân B bắt đầu từ các từ khóa ngữ cảnh
        seeds = list(self.generator.config.default_keywords)

        # 2. Tác nhân B duyệt cây tìm kiếm heuristic
        converged, steps_taken, explored, found_str = self.searcher.search(
            seed_roots=seeds,
            verifier_oracle=verifier_oracle,
            max_steps=N + 3
        )

        outcome = ""
        # 3. Vòng lặp cập nhật trọng số Minimax
        if converged and steps_taken <= N:
            # Tác nhân B hội tụ thành công trong ngưỡng N bước (N <= 5)
            # Tác nhân A nhận phản hồi phạt để buộc sinh biến thể entropy cao hơn
            self.generator.receive_penalty_feedback(steps_taken)
            outcome = f"AGENT_B_CONVERGED_FAST (Steps={steps_taken} <= {N}) -> Agent A penalized"
        else:
            # Vượt quá ngưỡng N bước hoặc không hội tụ
            # Tác nhân B nhận phản hồi cập nhật ma trận trọng số
            # Học từ các chuỗi đã duyệt để ưu tiên phân phối mới vào hàng đợi duyệt
            self.generator.receive_success_feedback()
            
            # Target score cho các ứng viên:
            # Gán điểm cao hơn cho các chuỗi có đặc điểm gần với target hoặc các nhánh sâu
            if explored:
                target_scores = []
                for s in explored:
                    # Điểm gán dựa trên độ tương đồng n-gram hoặc chiều dài
                    overlap = sum(c1 == c2 for c1, c2 in zip(s, target))
                    score = min(1.0, overlap / max(len(target), 1) + 0.3)
                    target_scores.append(score)
                self.evaluator.update_weights(explored, np.array(target_scores, dtype=np.float32))

            outcome = f"AGENT_A_DEFENDED (Steps={steps_taken} > {N}) -> Agent B weights updated"

        _, ent_stats = calculate_repetition_and_class_entropy(target)

        episode_stat = {
            "episode": episode_idx,
            "target": target,
            "target_entropy": round(target_entropy, 3),
            "letters": ent_stats["letter_count"],
            "digits": ent_stats["digit_count"],
            "specials": ent_stats["special_count"],
            "repetition_ratio": ent_stats["repetition_ratio"],
            "converged": converged,
            "steps_taken": steps_taken,
            "mutation_prob": round(self.generator.mutation_prob, 3),
            "outcome": outcome,
            "total_explored": len(explored)
        }
        self.history.append(episode_stat)
        return episode_stat

    def train(self, num_episodes: int = 20) -> List[Dict[str, Any]]:
        """Chạy huấn luyện nhiều vòng đối kháng."""
        print(f"=== Bắt đầu huấn luyện đối kháng (Self-Play) với {num_episodes} vòng ===")
        print(f"Ngưỡng hội tụ chuẩn N = {self.config.estimator.convergence_threshold_n} bước")
        if self.total_episodes_trained > 0:
            print(f"[*] Tiếp tục học tích lũy từ vòng đấu thứ {self.total_episodes_trained + 1}...\n")
        else:
            print("[*] Bắt đầu huấn luyện mới từ đầu (Fresh Start)...\n")
        
        for ep in range(1, num_episodes + 1):
            self.total_episodes_trained += 1
            stat = self.run_episode(self.total_episodes_trained)
            print(f"[Vòng {stat['episode']:02d}] Target: {stat['target']:<18} | "
                  f"Entropy: {stat['target_entropy']:.2f} (Chữ:{stat['letters']} Số:{stat['digits']} Lạ:{stat['specials']}) | "
                  f"Steps: {stat['steps_taken']:<2} | Mutation: {stat['mutation_prob']:.2f} | {stat['outcome']}")
            
        print("\n=== Huấn luyện hoàn tất! ===")
        return self.history
