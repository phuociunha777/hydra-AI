"""
Python ctypes wrapper for libnnue.dll (C/C++ SIMD AVX2 NNUE Engine)
Cung cấp giao diện gọi trực tiếp hàm suy luận vi giây (microsecond inference) từ Python.
"""

import os
import sys
import ctypes
import subprocess
from typing import List, Optional

class NNUESIMDEngine:
    """
    Giao diện điều khiển động cơ nhúng C/C++ SIMD AVX2.
    """
    _instance = None

    def __init__(self, dll_path: Optional[str] = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if dll_path is None:
            dll_path = os.path.join(base_dir, "libnnue.dll")
        
        self.dll_path = dll_path
        self._lib = None
        self._load_or_compile()

    def _compile_dll(self):
        """Tự động biên dịch mã nguồn C/C++ tối ưu hóa AVX2 nếu chưa có DLL."""
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cpp_file = os.path.join(base_dir, "nnue_engine.cpp")
        
        cmd = ["g++", "-O3", "-mavx2", "-shared", "-o", self.dll_path, cpp_file]
        try:
            res = subprocess.run(cmd, cwd=base_dir, capture_output=True, text=True, check=True)
            print("[SIMD Engine] Biên dịch libnnue.dll thành công với cờ -O3 -mavx2!")
        except Exception as e:
            print(f"[SIMD Engine] Cảnh báo: Không thể biên dịch tự động: {e}")

    def _load_or_compile(self):
        if not os.path.exists(self.dll_path):
            self._compile_dll()

        if os.path.exists(self.dll_path):
            try:
                self._lib = ctypes.CDLL(self.dll_path)
                self._setup_bindings()
            except Exception as e:
                print(f"[SIMD Engine] Không thể nạp DLL: {e}")
                self._lib = None
        else:
            print("[SIMD Engine] libnnue.dll không tồn tại.")

    def _setup_bindings(self):
        # int nnue_init(void)
        self._lib.nnue_init.restype = ctypes.c_int
        self._lib.nnue_init.argtypes = []

        # int nnue_has_avx2(void)
        self._lib.nnue_has_avx2.restype = ctypes.c_int
        self._lib.nnue_has_avx2.argtypes = []

        # float nnue_evaluate_string(const char* str)
        self._lib.nnue_evaluate_string.restype = ctypes.c_float
        self._lib.nnue_evaluate_string.argtypes = [ctypes.c_char_p]

        # float nnue_calculate_structural_entropy(const char* str)
        self._lib.nnue_calculate_structural_entropy.restype = ctypes.c_float
        self._lib.nnue_calculate_structural_entropy.argtypes = [ctypes.c_char_p]

        # void nnue_batch_evaluate(const char** strings, int count, float* out_scores)
        self._lib.nnue_batch_evaluate.restype = None
        self._lib.nnue_batch_evaluate.argtypes = [
            ctypes.POINTER(ctypes.c_char_p),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float)
        ]

        # double nnue_benchmark_latency_us(int iterations)
        self._lib.nnue_benchmark_latency_us.restype = ctypes.c_double
        self._lib.nnue_benchmark_latency_us.argtypes = [ctypes.c_int]

        # Khởi tạo
        self._lib.nnue_init()

    def is_ready(self) -> bool:
        return self._lib is not None

    def has_avx2(self) -> bool:
        if not self._lib:
            return False
        return bool(self._lib.nnue_has_avx2())

    def evaluate_string(self, text: str) -> float:
        """Đánh giá điểm heuristic của một chuỗi đơn lẻ."""
        if not self._lib:
            raise RuntimeError("Engine C/C++ DLL chưa sẵn sàng.")
        c_str = text.encode('utf-8')
        return float(self._lib.nnue_evaluate_string(c_str))

    def calculate_structural_entropy(self, text: str) -> float:
        """Tính nhanh độ hỗn loạn (Entropy) dựa trên tần số lặp của chữ, số, ký tự lạ như @."""
        if not self._lib:
            raise RuntimeError("Engine C/C++ DLL chưa sẵn sàng.")
        c_str = text.encode('utf-8')
        return float(self._lib.nnue_calculate_structural_entropy(c_str))

    def evaluate_batch(self, strings: List[str]) -> List[float]:
        """Đánh giá theo lô thông qua mảng con trỏ chuỗi C."""
        if not self._lib:
            raise RuntimeError("Engine C/C++ DLL chưa sẵn sàng.")
        count = len(strings)
        if count == 0:
            return []

        c_arr = (ctypes.c_char_p * count)(*[s.encode('utf-8') for s in strings])
        out_arr = (ctypes.c_float * count)()
        self._lib.nnue_batch_evaluate(c_arr, count, out_arr)
        return [float(out_arr[i]) for i in range(count)]

    def benchmark_latency(self, iterations: int = 100000) -> float:
        """
        Benchmark độ trễ trực tiếp trên CPU.
        Trả về thời gian trung bình (micro-giây / 1 lần suy luận).
        """
        if not self._lib:
            raise RuntimeError("Engine C/C++ DLL chưa sẵn sàng.")
        return float(self._lib.nnue_benchmark_latency_us(iterations))
