"""
Bộ lượng tử hóa mô hình (Model Quantization) sang định dạng NNUE int8/int16
Tương tự cấu trúc mạng nơ-ron đánh giá thế cờ trong Stockfish.
Xuất ra file nhị phân và file header C/C++ để biên dịch trực tiếp vào engine native.
"""

import os
import sys
import struct
import numpy as np

# Đảm bảo UTF-8 cho console Windows
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from core.neural_evaluator import LightweightEvaluator

class NNUEQuantizer:
    """
    Lượng tử hóa trọng số float32 sang int8 và int32/int16:
    - W1: float32 (64, 32) -> int8
    - b1: float32 (32,) -> int32
    - W2: float32 (32, 1) -> int8
    - b2: float32 (1,) -> int32
    """
    def __init__(self, evaluator: LightweightEvaluator):
        self.evaluator = evaluator

    def quantize(self):
        w1 = self.evaluator.W1
        b1 = self.evaluator.b1
        w2 = self.evaluator.W2.squeeze(-1)
        b2 = self.evaluator.b2[0]

        # Xác định tỷ lệ co giãn (Scale factors)
        max_w1 = max(np.max(np.abs(w1)), 1e-5)
        scale_w1 = 127.0 / max_w1
        w1_int8 = np.clip(np.round(w1 * scale_w1), -128, 127).astype(np.int8)
        # Chuẩn hóa b1 theo thang đầu vào 255.0 của C++ (khắc phục lỗi lệch 64.0 vs 255.0)
        b1_int32 = np.round(b1 * scale_w1 * 255.0).astype(np.int32)

        max_w2 = max(np.max(np.abs(w2)), 1e-5)
        scale_w2 = 127.0 / max_w2
        w2_int8 = np.clip(np.round(w2 * scale_w2), -128, 127).astype(np.int8)
        # Chuẩn hóa b2 theo thang kích hoạt tầng ẩn 127.0
        b2_int32 = int(np.round(b2 * scale_w2 * 127.0))

        # Hệ số chia phân giải ngược (Divisors) đồng bộ chuẩn toán học
        div1 = float((255.0 * scale_w1) / 127.0)
        div2 = float(127.0 * scale_w2)

        return {
            "w1_int8": w1_int8,
            "b1_int32": b1_int32,
            "w2_int8": w2_int8,
            "b2_int32": b2_int32,
            "scale_w1": float(scale_w1),
            "scale_w2": float(scale_w2),
            "divisor_1": div1,
            "divisor_2": div2,
            "input_dim": 64,
            "hidden_dim": 32
        }

    def export_c_header(self, output_header_path: str):
        """Tích hợp trực tiếp ma trận trọng số vào mã nguồn C/C++ dưới dạng header static."""
        q = self.quantize()
        w1 = q["w1_int8"]
        b1 = q["b1_int32"]
        w2 = q["w2_int8"]
        b2 = q["b2_int32"]

        lines = [
            "// Auto-generated NNUE Quantized Weights for Neural-Entropy-Search SIMD Engine",
            "// Format: int8 weights, int32 biases (Mathematically Scale-Aligned)",
            "#ifndef NNUE_WEIGHTS_H",
            "#define NNUE_WEIGHTS_H",
            "",
            "#include <stdint.h>",
            "",
            "#define NNUE_INPUT_DIM 64",
            "#define NNUE_HIDDEN_DIM 32",
            f"#define NNUE_SCALE_W1 {q['scale_w1']}f",
            f"#define NNUE_SCALE_W2 {q['scale_w2']}f",
            f"#define NNUE_DIVISOR_1 {q['divisor_1']}f",
            f"#define NNUE_DIVISOR_2 {q['divisor_2']}f",
            f"#define NNUE_B2_VAL {b2}",
            "",
            "// W1 Matrix: 64 x 32 (int8)",
            "static const int8_t NNUE_W1[64][32] = {"
        ]

        for i in range(64):
            row_str = ", ".join(f"{int(w1[i, j])}" for j in range(32))
            suffix = "," if i < 63 else ""
            lines.append(f"    {{{row_str}}}{suffix}")
        lines.append("};")
        lines.append("")

        lines.append("// b1 Biases: 32 (int32)")
        b1_str = ", ".join(f"{int(b)}" for b in b1)
        lines.append(f"static const int32_t NNUE_B1[32] = {{{b1_str}}};")
        lines.append("")

        lines.append("// W2 Matrix: 32 (int8)")
        w2_str = ", ".join(f"{int(w)}" for w in w2)
        lines.append(f"static const int8_t NNUE_W2[32] = {{{w2_str}}};")
        lines.append("")

        lines.append("#endif // NNUE_WEIGHTS_H\n")

        os.makedirs(os.path.dirname(os.path.abspath(output_header_path)), exist_ok=True)
        with open(output_header_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        print(f"[Quantizer] Đã xuất file C Header trọng số lượng tử hóa: {output_header_path}")

    def export_binary(self, output_bin_path: str):
        """Xuất ma trận trọng số ra file nhị phân chuẩn."""
        q = self.quantize()
        os.makedirs(os.path.dirname(os.path.abspath(output_bin_path)), exist_ok=True)
        with open(output_bin_path, "wb") as f:
            # Header: magic(4) + input_dim(4) + hidden_dim(4)
            f.write(b"NNUE")
            f.write(struct.pack("<II", 64, 32))
            # W1: 64 * 32 bytes (int8)
            f.write(q["w1_int8"].tobytes())
            # b1: 32 * 4 bytes (int32)
            f.write(q["b1_int32"].tobytes())
            # W2: 32 bytes (int8)
            f.write(q["w2_int8"].tobytes())
            # b2: 4 bytes (int32)
            f.write(struct.pack("<i", q["b2_int32"]))
        print(f"[Quantizer] Đã xuất file nhị phân NNUE: {output_bin_path}")
