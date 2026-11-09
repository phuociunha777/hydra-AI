"""
Mạng nơ-ron trọng số nhẹ đánh giá Heuristic (Lightweight NNUE-style Evaluator)
Được thiết kế để tối ưu hóa việc suy luận cực nhanh trên CPU/GPU/NPU,
tương tự nguyên lý mạng NNUE trong các công cụ cờ vua hiện đại (Stockfish).
"""

import os
import numpy as np
from typing import List, Tuple
from core.config import EstimatorConfig

def extract_features(s: str, dim: int = 64) -> np.ndarray:
    """
    Trích xuất vector đặc trưng cố định 64 chiều từ chuỗi ký tự:
    - 0..31: Hash n-gram / ký tự vào 32 buckets
    - 32: Độ dài chuỗi chuẩn hóa
    - 33: Tỷ lệ số (digits)
    - 34: Tỷ lệ ký tự hoa (uppercase)
    - 35: Tỷ lệ ký tự đặc biệt (specials)
    - 36: Tỷ lệ ký tự thường (lowercase)
    - 37..44: Các mẫu leetspeak phát hiện (@, 3, 1, 0, 5, 7, 4, 8)
    - 45..63: Đặc trưng chuyển dịch trạng thái (case transitions, consecutive digits, entropy)
    """
    feat = np.zeros(dim, dtype=np.float32)
    if not s:
        return feat

    length = len(s)
    feat[32] = min(length / 20.0, 1.0)
    
    digits = sum(c.isdigit() for c in s)
    uppers = sum(c.isupper() for c in s)
    lowers = sum(c.islower() for c in s)
    specials = length - (digits + uppers + lowers)

    feat[33] = digits / length
    feat[34] = uppers / length
    feat[35] = specials / length
    feat[36] = lowers / length

    # Biểu diễn ký tự n-gram băm vào 32 buckets
    for i, ch in enumerate(s):
        idx = (ord(ch) + i * 7) % 32
        feat[idx] += 1.0 / length

    # Dấu hiệu leetspeak phổ biến
    leets = ['@', '3', '1', '0', '$', '7', '4', '!']
    for idx, lch in enumerate(leets):
        if lch in s:
            feat[37 + idx] = 1.0

    # Chuyển đổi trạng thái (Transitions)
    transitions = 0
    for i in range(len(s) - 1):
        if (s[i].isalpha() and s[i+1].isdigit()) or (s[i].isdigit() and not s[i+1].isalnum()):
            transitions += 1
    feat[45] = min(transitions / 5.0, 1.0)

    # Ước lượng entropy đơn giản
    unique_chars = len(set(s))
    feat[46] = unique_chars / max(length, 1)

    return feat

class LightweightEvaluator:
    """
    Mạng nơ-ron đánh giá heuristic:
    - Layer 1: Linear(64 -> 32) + Clipped ReLU
    - Layer 2: Linear(32 -> 1) + Sigmoid
    Độ phức tạp tính toán rất nhỏ, thích hợp chạy hàng triệu nút/giây trên AVX2.
    """
    def __init__(self, config: EstimatorConfig = EstimatorConfig()):
        self.config = config
        self.input_dim = config.input_dim
        self.hidden_dim = config.hidden_dim
        self.output_dim = config.output_dim
        self.lr = config.learning_rate

        # Khởi tạo trọng số Xavier/He
        np.random.seed(42)
        scale1 = np.sqrt(2.0 / self.input_dim)
        self.W1 = np.random.randn(self.input_dim, self.hidden_dim).astype(np.float32) * scale1
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)

        scale2 = np.sqrt(2.0 / self.hidden_dim)
        self.W2 = np.random.randn(self.hidden_dim, self.output_dim).astype(np.float32) * scale2
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    def forward(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Lan truyền xuôi (Forward pass)."""
        # x shape: (batch_size, input_dim) hoặc (input_dim,)
        is_1d = (x.ndim == 1)
        if is_1d:
            x = x.reshape(1, -1)

        # Hidden layer
        z1 = np.dot(x, self.W1) + self.b1
        # Clipped ReLU [0, 1] - tương tự kiến trúc NNUE
        a1 = np.clip(z1, 0.0, 1.0)

        # Output layer
        z2 = np.dot(a1, self.W2) + self.b2
        # Sigmoid
        score = 1.0 / (1.0 + np.exp(-np.clip(z2, -15.0, 15.0)))

        if is_1d:
            return score[0, 0], a1[0], z1[0]
        return score, a1, z1

    def evaluate_string(self, s: str) -> float:
        """Đánh giá nhanh một chuỗi đơn lẻ."""
        feat = extract_features(s, self.input_dim)
        score, _, _ = self.forward(feat)
        return float(score)

    def evaluate_batch(self, strings: List[str]) -> np.ndarray:
        """Đánh giá theo lô (Batch inference) cho tầng phân tích mô hình."""
        if not strings:
            return np.array([], dtype=np.float32)
        feats = np.stack([extract_features(s, self.input_dim) for s in strings])
        scores, _, _ = self.forward(feats)
        return scores.squeeze(-1)

    def update_weights(self, strings: List[str], target_scores: np.ndarray):
        """
        Huấn luyện lan truyền ngược đơn giản (SGD update) từ phản hồi tự chơi đối kháng.
        """
        if not strings:
            return
        X = np.stack([extract_features(s, self.input_dim) for s in strings])
        Y = np.array(target_scores, dtype=np.float32).reshape(-1, 1)

        # Forward
        z1 = np.dot(X, self.W1) + self.b1
        a1 = np.clip(z1, 0.0, 1.0)
        z2 = np.dot(a1, self.W2) + self.b2
        pred = 1.0 / (1.0 + np.exp(-np.clip(z2, -15.0, 15.0)))

        # Gradient
        # dL/dz2 = (pred - Y)
        d_z2 = (pred - Y) / len(strings)
        d_W2 = np.dot(a1.T, d_z2)
        d_b2 = np.sum(d_z2, axis=0)

        # dL/da1 = d_z2 * W2.T
        d_a1 = np.dot(d_z2, self.W2.T)
        # d_z1 = d_a1 * (0 < z1 < 1)
        d_z1 = d_a1 * ((z1 > 0.0) & (z1 < 1.0))
        d_W1 = np.dot(X.T, d_z1)
        d_b1 = np.sum(d_z1, axis=0)

        # Cập nhật Gradient Descent với clipping để đảm bảo ổn định
        self.W1 -= self.lr * np.clip(d_W1, -1.0, 1.0)
        self.b1 -= self.lr * np.clip(d_b1, -1.0, 1.0)
        self.W2 -= self.lr * np.clip(d_W2, -1.0, 1.0)
        self.b2 -= self.lr * np.clip(d_b2, -1.0, 1.0)

    def save(self, filepath: str):
        """Lưu ma trận trọng số ra file .npz để tái sử dụng ở các lần train sau."""
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        np.savez_compressed(
            filepath,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2
        )

    def load(self, filepath: str) -> bool:
        """Nạp ma trận trọng số đã huấn luyện trước đó."""
        if not os.path.exists(filepath):
            return False
        try:
            data = np.load(filepath)
            self.W1 = data["W1"].astype(np.float32)
            self.b1 = data["b1"].astype(np.float32)
            self.W2 = data["W2"].astype(np.float32)
            self.b2 = data["b2"].astype(np.float32)
            return True
        except Exception as e:
            print(f"[NeuralEvaluator] Không thể nạp checkpoint {filepath}: {e}")
            return False
