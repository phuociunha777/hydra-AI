"""
Unit & Integration Tests for Neural-Entropy-Search Architecture.
Kiểm thử toàn bộ các thành phần kiến trúc nghiên cứu thuật toán:
1. Profile Generator & Char-level Markov Chain Generative Model
2. Multi-stage Filtering & Keyboard Walk Bias Trap Avoidance
3. Neural Evaluator & Heuristic Tree Search
4. Tiered System Architecture & SIMD AVX2 Integration
5. Asynchronous Verification Dispatcher & Rate Limiter
6. Quantizer & Native C++ SIMD Vector Engine
7. Security Auditor & Complexity Analysis
"""

import unittest
import asyncio
import numpy as np
import os
import sys

# Thêm thư mục gốc vào sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.profile_generator import (
    ProfileGenerator,
    MarkovWordGenerator,
    calculate_shannon_entropy,
    calculate_repetition_and_class_entropy,
    detect_keyboard_bias
)
from core.neural_evaluator import LightweightEvaluator, extract_features
from core.heuristic_search import HeuristicSearcher
from core.tiered_architecture import InferenceCore
from core.self_play import AdversarialSelfPlay
from async_dispatcher.rate_limiter import TokenBucketRateLimiter, MockTargetService
from async_dispatcher.dispatcher import AsyncVerificationDispatcher
from embedded_engine.quantizer import NNUEQuantizer
from embedded_engine.wrapper import NNUESIMDEngine
from core.security_evaluator import SecurityAuditor
from core.config import SystemConfig

class TestNeuralEntropySearch(unittest.TestCase):

    def test_01_profile_generator(self):
        gen = ProfileGenerator()
        cand, ent = gen.generate_candidate(keywords=["admin"], user_profile={"name": "hanoi", "year": "2025"})
        self.assertIsInstance(cand, str)
        self.assertGreater(len(cand), 3)
        self.assertGreater(ent, 0.0)

        # Test penalty update
        old_mut = gen.mutation_prob
        gen.receive_penalty_feedback(steps_taken=3)
        self.assertGreater(gen.mutation_prob, old_mut)

    def test_01b_entropy_with_classes_and_repetition(self):
        # 1. Đo lường tần số lặp của chữ, số và ký tự lạ @
        ent_repeat, stats_repeat = calculate_repetition_and_class_entropy("aaaa1111@@@@")
        ent_diverse, stats_diverse = calculate_repetition_and_class_entropy("Admin@2025!")

        # Chuỗi lặp nhiều (aaaa1111@@@@) phải có entropy thấp hơn chuỗi đa dạng
        self.assertGreater(stats_repeat["repetition_ratio"], 0.5)
        self.assertLess(ent_repeat, ent_diverse)
        self.assertGreater(stats_diverse["special_count"], 0)
        self.assertIn("@", "Admin@2025!")

        # 2. Kiểm tra phát hiện quy luật bàn phím
        ent_kb, stats_kb = calculate_repetition_and_class_entropy("1qaz2wsx")
        self.assertTrue(stats_kb["keyboard_bias_detected"])
        self.assertLessEqual(ent_kb, 1.45)

    def test_01c_contiguous_run_length_user_examples(self):
        # Kiểm thử đúng ví dụ của người dùng:
        # '49095killseo': có cụm chữ số liên tiếp lặp lại 5 lần (49095) -> Entropy THẤP
        # '2b4e5b6e': chữ và số đan xen liên tục (t_rate = 1.0), không lặp cụm -> Entropy CAO
        ent_low, stats_low = calculate_repetition_and_class_entropy("49095killseo")
        ent_high, stats_high = calculate_repetition_and_class_entropy("2b4e5b6e")

        self.assertEqual(stats_low["max_digit_run"], 5)
        self.assertEqual(stats_high["max_digit_run"], 1)
        self.assertGreater(stats_high["transition_rate"], 0.9)
        self.assertLess(stats_low["transition_rate"], 0.2)
        # Entropy của 2b4e5b6e phải cao hơn nhiều lần so với 49095killseo
        self.assertGreater(ent_high, ent_low * 3.0)

    def test_01d_markov_word_generator(self):
        # Kiểm thử mô hình ngôn ngữ Markov Chain mức ký tự
        markov = MarkovWordGenerator()
        words = [markov.generate_word() for _ in range(20)]
        self.assertEqual(len(words), 20)
        for w in words:
            self.assertGreaterEqual(len(w), 4)
            self.assertLessEqual(len(w), 12)
        # Độ đa dạng cao (tỷ lệ từ độc nhất >= 70%)
        unique_ratio = len(set(words)) / len(words)
        self.assertGreaterEqual(unique_ratio, 0.70)

    def test_01e_keyboard_bias_trap_avoidance(self):
        # Tránh bẫy thiên lệch: các chuỗi ziczac bàn phím phải vào nhóm Cây nông (Shallow Search)
        core = InferenceCore(evaluator=LightweightEvaluator(), entropy_cutoff_threshold=2.0)
        zigzag_samples = ["1qaz2wsx", "1qaz2wsx3edc", "zaq1xsw2", "qwerty12345"]
        
        low_g, high_g, stats = core.route_and_filter_candidates(zigzag_samples)
        for z in zigzag_samples:
            self.assertIn(z, low_g, f"Chuỗi bàn phím '{z}' phải vào nhóm Cây nông (Shallow Search)")
        self.assertEqual(len(high_g), 0)

    def test_02_neural_evaluator(self):
        ev = LightweightEvaluator()
        feat = extract_features("Admin@2024", 64)
        self.assertEqual(feat.shape, (64,))
        
        score = ev.evaluate_string("Admin@2024")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

        # Batch evaluation
        scores = ev.evaluate_batch(["Admin@2024", "password123", "root"])
        self.assertEqual(len(scores), 3)

    def test_03_heuristic_search(self):
        searcher = HeuristicSearcher()
        successors = searcher.generate_successors("admin")
        self.assertGreater(len(successors), 0)

        # Oracle test
        target = "admin123"
        converged, steps, explored, found = searcher.search(
            seed_roots=["admin"],
            verifier_oracle=lambda x: x == target,
            max_steps=5
        )
        self.assertTrue(converged)
        self.assertEqual(found, target)

    def test_04_tiered_architecture(self):
        ev = LightweightEvaluator()
        core = InferenceCore(evaluator=ev, pool_size=10, classification_threshold=0.3, entropy_cutoff_threshold=2.0)
        
        raw_list = ["admin", "Admin@123", "root", "", "a", "System#2024", "Admin@123", "aaaa1111@@@@"]
        pruned = core.prune_search_space(raw_list)
        self.assertNotIn("", pruned)
        self.assertNotIn("a", pruned)
        self.assertEqual(len(pruned), len(set(pruned)))

        # Kiểm thử cơ chế sàng lọc Heuristic nhiều tầng rẽ nhánh
        low_g, high_g, stats = core.route_and_filter_candidates(pruned)
        self.assertGreater(len(low_g) + len(high_g), 0)

        pool = core.generate_candidate_pool(raw_list)
        self.assertIsInstance(pool, list)
        self.assertLessEqual(len(pool), 10)
        for item in pool:
            self.assertEqual(len(item), 3) # (cand, score, cluster)

    def test_05_async_dispatcher_and_rate_limiter(self):
        async def run_async_test():
            limiter = TokenBucketRateLimiter(capacity=5, refill_rate=1.0)
            target = MockTargetService(secret_target="pass@123", rate_limiter=limiter, simulated_network_latency_ms=1.0)
            dispatcher = AsyncVerificationDispatcher(target_service=target, max_concurrency=10)

            pool = [
                ("wrong1", 0.9, "test"),
                ("wrong2", 0.8, "test"),
                ("pass@123", 0.7, "test"),
                ("wrong3", 0.6, "test"),
                ("wrong4", 0.5, "test"),
                ("wrong5", 0.4, "test"),
                ("wrong6", 0.3, "test"),
            ]
            summary = await dispatcher.dispatch_candidate_pool(pool)
            self.assertEqual(summary["total_sent"], 7)
            self.assertTrue(summary["matched_found"])
            self.assertGreater(summary["status_429_throttled"], 0) # Limited by capacity=5

        asyncio.run(run_async_test())

    def test_06_quantizer_and_simd_engine(self):
        ev = LightweightEvaluator()
        quantizer = NNUEQuantizer(ev)
        q_data = quantizer.quantize()
        self.assertEqual(q_data["w1_int8"].shape, (64, 32))
        self.assertEqual(q_data["w1_int8"].dtype, np.int8)

        # Test SIMD AVX2 Engine
        simd = NNUESIMDEngine()
        self.assertTrue(simd.is_ready())
        self.assertTrue(simd.has_avx2())
        score = simd.evaluate_string("P@ssword2025")
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)
        lat = simd.benchmark_latency(iterations=1000)
        self.assertLess(lat, 50.0) # Dưới 50 micro-giây (thực tế ~0.2 μs)

    def test_07_security_auditor(self):
        auditor = SecurityAuditor()
        
        # Test mật khẩu yếu
        res_weak = auditor.evaluate_target("Admin123", {"name": "admin"}, max_steps=10)
        self.assertTrue(res_weak["converged"])
        self.assertLessEqual(res_weak["steps_to_converge"], 2)
        self.assertIn(res_weak["security_rating"], [1, 2])
        self.assertGreater(res_weak["nodes_in_memory"], 0)

        # Test mật khẩu mạnh
        res_strong = auditor.evaluate_target("2@4#5!9&xK", {"name": "john"}, max_steps=10)
        self.assertFalse(res_strong["converged"])
        self.assertGreaterEqual(res_strong["security_rating"], 4)
        self.assertGreater(res_strong["composite_entropy"], 3.5)

if __name__ == "__main__":
    unittest.main()
