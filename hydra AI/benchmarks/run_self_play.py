"""
Huấn luyện đối kháng (Adversarial Self-Play) giữa Agent A và Agent B
Theo dõi sự tiến hóa của Shannon Entropy, tỷ lệ đột biến và số bước hội tụ.
"""

import sys
import os

# Thêm thư mục gốc vào PYTHONPATH
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.config import SystemConfig
from core.self_play import AdversarialSelfPlay
from embedded_engine.quantizer import NNUEQuantizer
import subprocess

def run_self_play_training(episodes: int = 15, resume: bool = True, export_weights: bool = True):
    print("=" * 75)
    print("   NEURAL ENTROPY SEARCH - ADVERSARIAL SELF-PLAY TRAINING (MINIMAX)")
    print("=" * 75)
    print("Mục tiêu: Tối ưu hóa mô hình đánh giá không gian mật khẩu bằng Self-Play")
    print("Tác nhân A: Profile-based Generator (mô phỏng thói quen người dùng)")
    print("Tác nhân B: Entropy Estimator (Duyệt cây Heuristic kết hợp Neural Evaluator)\n")

    config = SystemConfig()
    self_play = AdversarialSelfPlay(config)

    checkpoint_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
    if resume:
        self_play.load_checkpoint(checkpoint_dir)

    history = self_play.train(num_episodes=episodes)

    # Lưu checkpoint cho các lần huấn luyện tiếp theo
    self_play.save_checkpoint(checkpoint_dir)

    # Thống kê kết quả
    avg_entropy_start = sum(h["target_entropy"] for h in history[:3]) / max(len(history[:3]), 1)
    avg_entropy_end = sum(h["target_entropy"] for h in history[-3:]) / max(len(history[-3]), 1)
    fast_convergence_count = sum(1 for h in history if h["converged"] and h["steps_taken"] <= config.estimator.convergence_threshold_n)

    print("\n" + "=" * 75)
    print("                     KẾT QUẢ HUẤN LUYỆN ĐỐI KHÁNG")
    print("=" * 75)
    print(f"• Số vòng vừa đấu (episodes)       : {episodes}")
    print(f"• Tổng tích lũy các đợt train      : {self_play.total_episodes_trained} vòng")
    print(f"• Số lần Agent B bắt bài (<=5 bước): {fast_convergence_count} lần")
    print(f"• Số lần Agent A phòng thủ         : {episodes - fast_convergence_count} lần")
    print(f"• Shannon Entropy trung bình đợt này: {avg_entropy_end:.2f} bits")
    print(f"• Tỷ lệ đột biến leetspeak của A   : {self_play.generator.mutation_prob:.2f}")

    if export_weights:
        print("\n[Export] Đang trích xuất ma trận trọng số và lượng tử hóa sang định dạng NNUE...")
        quantizer = NNUEQuantizer(self_play.evaluator)
        engine_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embedded_engine")
        header_path = os.path.join(engine_dir, "nnue_weights.h")
        bin_path = os.path.join(engine_dir, "nnue_weights.bin")
        quantizer.export_c_header(header_path)
        quantizer.export_binary(bin_path)

        # Biên dịch lại C++ DLL với trọng số mới
        print("[Build] Đang cập nhật ma trận trọng số vào C/C++ SIMD AVX2 Engine (libnnue.dll)...")
        cpp_file = os.path.join(engine_dir, "nnue_engine.cpp")
        dll_file = os.path.join(engine_dir, "libnnue.dll")
        cmd = ["g++", "-O3", "-mavx2", "-shared", "-o", dll_file, cpp_file]
        try:
            subprocess.run(cmd, cwd=engine_dir, capture_output=True, text=True, check=True)
            print("[Build] Đã cập nhật và biên dịch thành công libnnue.dll với trọng số mới!")
        except Exception as e:
            print(f"[Build] Cảnh báo lỗi khi biên dịch C++ DLL: {e}")

    print("=" * 75 + "\n")
    return history

if __name__ == "__main__":
    run_self_play_training(episodes=12)
