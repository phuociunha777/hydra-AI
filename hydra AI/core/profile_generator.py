"""
Tác nhân A - Bộ mô phỏng hành vi (Profile-based Generator)
Tiếp nhận các tham số ngữ cảnh (quy tắc ngôn ngữ, mẫu ký tự phổ biến, leetspeak)
và sinh ra các chuỗi tuân thủ thói quen ghi nhớ của người dùng.

Tích hợp hàm toán học đo độ hỗn loạn (Entropy) dựa trên:
1. Tần số lặp lại của từng ký tự (Character Repetition Frequency).
2. Tần số phân bố 3 lớp ký tự: chữ cái, chữ số, ký tự lạ/đặc biệt (như @, #, $, !).
3. Phát hiện quy luật bàn phím (Keyboard walk / sequential bias) để tránh bẫy thiên lệch.
"""

import math
import os
import random
from collections import defaultdict
from typing import List, Dict, Optional, Tuple, Any
from core.config import GeneratorConfig

# =============================================================================
# 1. Nhận diện quy luật bàn phím & Tránh bẫy thiên lệch (Keyboard Walk & Bias Detection)
# =============================================================================

# Các chuỗi mẫu bàn phím thường gặp (hàng ngang, cột dọc ziczac, đường chéo)
KEYBOARD_WALK_PATTERNS = [
    # Hàng ngang
    "qwerty", "asdfgh", "zxcvbn", "12345", "123456", "654321", "098765",
    "yuiop", "hjkl;", "bnm,.", "poiuy", "lkjhg", "mnbvc",
    # Cột dọc & ziczac
    "1qaz", "2wsx", "3edc", "4rfv", "5tgb", "6yhn", "7ujm", "8ik,", "9ol.", "0p;/",
    "qaz", "wsx", "edc", "rfv", "tgb", "yhn", "ujm", "ik,", "ol.",
    "zaq", "xsw", "cde", "vfr", "bgt", "nhy", "mju",
    # Ziczac đan xen
    "1qaz2wsx", "qazwsx", "zaq1", "xsw2", "cde3", "vfr4", "zaq1xsw2",
    "1qaz2wsx3edc", "qazwsxedc", "1a2s3d", "1a2s3d4f", "qsedft", "awdrgy"
]

KEYBOARD_COORDS = {
    '1': (0,0), '2': (0,1), '3': (0,2), '4': (0,3), '5': (0,4), '6': (0,5), '7': (0,6), '8': (0,7), '9': (0,8), '0': (0,9), '-': (0,10), '=': (0,11),
    'q': (1,0), 'w': (1,1), 'e': (1,2), 'r': (1,3), 't': (1,4), 'y': (1,5), 'u': (1,6), 'i': (1,7), 'o': (1,8), 'p': (1,9), '[': (1,10), ']': (1,11),
    'a': (2,0), 's': (2,1), 'd': (2,2), 'f': (2,3), 'g': (2,4), 'h': (2,5), 'j': (2,6), 'k': (2,7), 'l': (2,8), ';': (2,9), '\'': (2,10),
    'z': (3,0), 'x': (3,1), 'c': (3,2), 'v': (3,3), 'b': (3,4), 'n': (3,5), 'm': (3,6), ',': (3,7), '.': (3,8), '/': (3,9),
    '!': (0,0), '@': (0,1), '#': (0,2), '$': (0,3), '%': (0,4), '^': (0,5), '&': (0,6), '*': (0,7), '(': (0,8), ')': (0,9)
}

def detect_keyboard_bias(text: str) -> Tuple[bool, str]:
    """
    Phát hiện bẫy quy luật bàn phím (Keyboard walk & zigzag bias):
    1. So khớp mẫu ziczac / cột dọc / hàng ngang định sẵn.
    2. Phân tích hình học khoảng cách liền kề các phím trên bàn phím QWERTY.
    """
    lower_t = text.lower()
    for p in KEYBOARD_WALK_PATTERNS:
        if p in lower_t:
            return True, f"Pattern '{p}'"
            
    # Phân tích hình học bàn phím nếu độ dài >= 5
    coords = [KEYBOARD_COORDS[c] for c in lower_t if c in KEYBOARD_COORDS]
    if len(coords) >= 5 and len(coords) >= len(lower_t) * 0.8:
        adj_steps = 0
        total_steps = len(coords) - 1
        for i in range(total_steps):
            dr = abs(coords[i][0] - coords[i+1][0])
            dc = abs(coords[i][1] - coords[i+1][1])
            if dr <= 1 and dc <= 1:
                adj_steps += 1
        ratio = adj_steps / max(total_steps, 1)
        if ratio >= 0.70:
            return True, f"Geometry Walk (Ratio: {ratio:.2f})"
            
    return False, ""

def calculate_repetition_and_class_entropy(text: str) -> Tuple[float, Dict[str, Any]]:
    """
    Hàm toán học thuần túy đo độ hỗn loạn (Entropy) dựa trên:
    1. Tần số lặp lại liên tiếp của cùng một lớp ký tự (Contiguous Class Run-Length):
       - Ví dụ '49095killseo': có cụm chữ số '49095' lặp lại 5 lần liên tiếp -> Entropy THẤP.
       - Ví dụ '2b4e5b6e': chữ và số đan xen liên tục, không bị lặp cụm liên tiếp -> Entropy CAO.
    2. Tần suất xen kẽ chuyển đổi lớp (Class Alternation/Transition Rate).
    3. Tần số lặp lại ký tự đồng nhất (Identical Character Repetitions).
    4. Nhận diện các ký tự lạ/đặc biệt (@, #, $, !) và phát hiện quy luật bàn phím.
    
    Trả về:
        (composite_entropy, stats_dictionary)
    """
    if not text:
        return 0.0, {
            "length": 0, "letter_count": 0, "digit_count": 0, "special_count": 0,
            "letter_freq": 0.0, "digit_freq": 0.0, "special_freq": 0.0,
            "max_digit_run": 0, "max_letter_run": 0, "max_special_run": 0, "max_class_run": 0,
            "transition_rate": 0.0, "repetition_ratio": 0.0, "class_entropy": 0.0, "char_entropy": 0.0,
            "run_penalty": 0.0, "keyboard_bias_detected": False, "composite_entropy": 0.0
        }

    length = len(text)
    runs = []
    curr_type = None
    curr_len = 0
    digit_count = 0
    letter_count = 0
    special_count = 0
    char_freq: Dict[str, int] = {}

    for c in text:
        char_freq[c] = char_freq.get(c, 0) + 1
        if c.isdigit():
            t = "digit"
            digit_count += 1
        elif c.isalpha():
            t = "letter"
            letter_count += 1
        else:
            t = "special"
            special_count += 1

        if t == curr_type:
            curr_len += 1
        else:
            if curr_type:
                runs.append((curr_type, curr_len))
            curr_type = t
            curr_len = 1
    if curr_type:
        runs.append((curr_type, curr_len))

    # 1. Phân tích độ dài dải lặp lại liên tiếp theo từng lớp
    max_digit_run = max((r[1] for r in runs if r[0] == "digit"), default=0)
    max_letter_run = max((r[1] for r in runs if r[0] == "letter"), default=0)
    max_special_run = max((r[1] for r in runs if r[0] == "special"), default=0)
    max_class_run = max(r[1] for r in runs)
    max_run_ratio = max_class_run / length

    # 2. Tần suất chuyển đổi xen kẽ (Transition Rate)
    # Càng xen kẽ cao (như 2b4e5b6e -> 100%) thì entropy càng cao
    transitions = len(runs) - 1
    transition_rate = transitions / max(length - 1, 1)

    # 3. Phân bố và Entropy cơ sở
    counts = char_freq
    char_entropy = 0.0
    for cnt in counts.values():
        p_c = cnt / length
        char_entropy -= p_c * math.log2(p_c)

    class_entropy = 0.0
    for cnt in (letter_count, digit_count, special_count):
        if cnt > 0:
            p = cnt / length
            class_entropy -= p * math.log2(p)

    # 4. Tần số lặp ký tự đơn lẻ
    repeated_chars = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
    repetition_ratio = repeated_chars / length

    # 5. Phạt dải cùng loại lặp liên tiếp kéo dài (Contiguous Cluster Penalty)
    # Ví dụ chữ số liên tiếp >= 4 (như '49095' có 5 số liên tục)
    run_penalty = 0.0
    if max_digit_run >= 4:
        run_penalty += 0.45 + 0.15 * (max_digit_run - 4)
    if max_letter_run >= 5:
        run_penalty += 0.35 + 0.10 * (max_letter_run - 5)

    # 6. Phát hiện quy luật bàn phím & Bẫy thiên lệch (Keyboard Bias Trap)
    keyboard_bias_detected, keyboard_pattern_info = detect_keyboard_bias(text)
    keyboard_penalty = 1.35 if keyboard_bias_detected else 0.0

    # 7. Công thức tính Entropy tổng hợp
    # - Nhân hệ số xen kẽ: (0.35 + 0.85 * transition_rate)
    # - Nhân hệ số độ dài dải: (1.15 - 0.60 * max_run_ratio)
    # - Trừ hình phạt cụm liên tiếp kéo dài (run_penalty)
    # - Trừ hình phạt quy luật bàn phím (keyboard_penalty)
    composite_entropy = (
        (char_entropy * (0.35 + 0.85 * transition_rate) + 0.60 * class_entropy) 
        * (1.15 - 0.60 * max_run_ratio)
        - run_penalty 
        - 0.40 * repetition_ratio 
        - keyboard_penalty
    )
    # Chuỗi có quy luật bàn phím luôn có độ hỗn loạn thực tế thấp (< 1.45 bits)
    if keyboard_bias_detected:
        composite_entropy = min(composite_entropy, 1.40)
    composite_entropy = max(0.1, round(composite_entropy, 3))

    stats = {
        "length": length,
        "letter_count": letter_count,
        "digit_count": digit_count,
        "special_count": special_count,
        "letter_freq": round(letter_count / max(length, 1), 3),
        "digit_freq": round(digit_count / max(length, 1), 3),
        "special_freq": round(special_count / max(length, 1), 3),
        "max_digit_run": max_digit_run,
        "max_letter_run": max_letter_run,
        "max_special_run": max_special_run,
        "max_class_run": max_class_run,
        "transition_rate": round(transition_rate, 3),
        "repetition_ratio": round(repetition_ratio, 3),
        "class_entropy": round(class_entropy, 3),
        "char_entropy": round(char_entropy, 3),
        "run_penalty": round(run_penalty, 3),
        "keyboard_bias_detected": keyboard_bias_detected,
        "keyboard_pattern_info": keyboard_pattern_info,
        "composite_entropy": composite_entropy
    }
    return composite_entropy, stats

def calculate_shannon_entropy(text: str) -> float:
    """Wrapper tương thích ngược trả về composite entropy."""
    ent, _ = calculate_repetition_and_class_entropy(text)
    return ent

# =============================================================================
# 2. Mô hình ngôn ngữ Markov Chain mức ký tự (Char-level Markov Word Generator)
# =============================================================================

class MarkovWordGenerator:
    """
    Mô hình ngôn ngữ thống kê mức ký tự (Char-level Markov Chain - Order 2).
    Học phân phối xác suất chuyển đổi giữa các ký tự tự nhiên P(c_t | c_{t-1}, c_{t-2})
    để sinh các từ gốc hoàn toàn mới, đa dạng và không bị trùng lặp.
    
    Giải quyết triệt để vấn đề:
    'Không gian từ gốc quá hẹp, sau hàng nghìn vòng các từ Security, Vietnam, Hanoi, Admin
    bắt buộc phải lặp đi lặp lại liên tục'.
    """
    def __init__(self, seed_corpus: Optional[List[str]] = None):
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.starts: List[str] = []
        
        default_seed = [
            # Thuật ngữ bảo mật & công nghệ
            "admin", "administrator", "system", "security", "network", "database",
            "server", "access", "crypto", "cipher", "matrix", "vector", "socket",
            "terminal", "kernel", "daemon", "protocol", "quantum", "bytecode", "neural",
            # Thiên văn, không gian, thần thoại
            "phoenix", "dragon", "valkyrie", "shadow", "thunder", "aurora", "eclipse",
            "solaris", "stellar", "galaxy", "pulsar", "hyperion", "vortex", "zenith",
            "pegasus", "falcon", "hunter", "titan", "orion", "chronos", "apollo",
            # Tiền tố / Hậu tố và âm vị học tự nhiên
            "guardian", "sentinel", "vanguard", "tempest", "obsidian", "emerald",
            "starlight", "sunshine", "winter", "summer", "autumn", "spring",
            "cyber", "hyper", "mega", "ultra", "krypton", "veritas", "nexus", "aegis"
        ]
        self.train(seed_corpus or default_seed)

    def train(self, words: List[str]):
        """Xây dựng và cập nhật ma trận chuyển tiếp N-gram."""
        for w in words:
            clean_w = w.strip().lower()
            if len(clean_w) < 3 or not clean_w.isalpha():
                continue
            padded = f"^^{clean_w}$$"
            self.starts.append(padded[:2])
            for i in range(len(padded) - 2):
                ctx = padded[i:i+2]
                nxt = padded[i+2]
                self.transitions[ctx][nxt] += 1

    def generate_word(self, min_len: int = 5, max_len: int = 9, max_attempts: int = 20) -> str:
        """Sinh một từ gốc tự nhiên, mới lạ và phong phú."""
        if not self.transitions or not self.starts:
            return "matrix"

        for _ in range(max_attempts):
            ctx = random.choice(self.starts)
            word_chars = [ctx[1]] if ctx[1] != '^' else []
            for _ in range(max_len + 3):
                nxt_map = self.transitions.get(ctx)
                if not nxt_map:
                    break
                chars = list(nxt_map.keys())
                weights = list(nxt_map.values())
                nxt = random.choices(chars, weights=weights)[0]
                if nxt == '$':
                    break
                word_chars.append(nxt)
                ctx = ctx[1] + nxt
                if len(word_chars) >= max_len:
                    break
            
            cand = "".join(word_chars)
            if min_len <= len(cand) <= max_len:
                return cand

        # Fallback từ các âm vị chuẩn tự nhiên nếu ngẫu nhiên không đạt độ dài
        syllables_1 = ["val", "sol", "krypt", "zen", "aeg", "chron", "hyper", "nex", "shad", "phoe", "aur"]
        syllables_2 = ["or", "ion", "ix", "on", "ar", "is", "os", "el", "ex", "us", "ia", "en"]
        return f"{random.choice(syllables_1)}{random.choice(syllables_2)}"

# =============================================================================
# 3. Tác nhân A: Profile Generator (Đa dạng hóa từ gốc & thích nghi Entropy)
# =============================================================================

class ProfileGenerator:
    """
    Tác nhân A: Mô phỏng hành vi đặt chuỗi ký tự dựa trên ngữ cảnh người dùng.
    Tích hợp:
    - Kho từ vựng đa lĩnh vực phong phú (Multi-domain wordlist)
    - Mô hình ngôn ngữ Markov Chain mức ký tự (Char-level Markov Model) sinh từ tự do
    - Hỗ trợ nạp file wordlist ngoại vi hàng trăm ngàn từ
    - Thích nghi động tăng Shannon Entropy khi bị đối thủ hội tụ nhanh
    """
    def __init__(self, config: Optional[GeneratorConfig] = None, wordlist_path: Optional[str] = None):
        self.config = config or GeneratorConfig()
        self.mutation_prob = self.config.base_mutation_prob
        self.entropy_target = self.config.base_entropy_target
        self.penalty_count = 0
        self.generation_history: List[Tuple[str, float]] = []

        # Khởi tạo kho từ vựng mở rộng
        self.wordlist: List[str] = list(self.config.default_keywords)
        
        # Khởi tạo động cơ ngôn ngữ Markov Chain sinh ký tự tự do
        self.markov_engine = MarkovWordGenerator(seed_corpus=self.wordlist)

        # Nạp wordlist từ file bên ngoài nếu được cấu hình
        target_wordlist = wordlist_path or self.config.wordlist_file
        if target_wordlist:
            self.load_wordlist(target_wordlist)

    def load_wordlist(self, filepath: str):
        """
        Nạp tập từ điển wordlist mở rộng (hàng chục nghìn đến hàng trăm nghìn từ) từ file đĩa.
        Đồng thời huấn luyện bổ sung cho mạng Markov Chain để học phân bố ký tự mới.
        """
        if not os.path.exists(filepath):
            return
        try:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                new_words = [line.strip().lower() for line in f if line.strip() and len(line.strip()) >= 3]
            if new_words:
                self.wordlist.extend(new_words)
                self.markov_engine.train(new_words)
        except Exception:
            pass

    def apply_leetspeak(self, word: str, prob: float) -> str:
        """Áp dụng biến đổi leetspeak tuân thủ thói quen người dùng."""
        result = []
        for ch in word:
            lower_ch = ch.lower()
            if lower_ch in self.config.leetspeak_rules and random.random() < prob:
                sub = random.choice(self.config.leetspeak_rules[lower_ch])
                result.append(sub)
            else:
                result.append(ch)
        return "".join(result)

    def apply_capitalization_habit(self, word: str, style_choice: int) -> str:
        """Áp dụng quy tắc viết hoa theo thói quen ghi nhớ."""
        if not word:
            return word
        if style_choice == 1:
            return word.capitalize()
        elif style_choice == 2:
            return "".join(c.upper() if random.random() > 0.5 else c.lower() for c in word)
        elif style_choice == 3:
            return word.upper()
        return word.lower()

    def generate_candidate(self, 
                           keywords: Optional[List[str]] = None, 
                           user_profile: Optional[Dict[str, str]] = None) -> Tuple[str, float]:
        """
        Sinh chuỗi ký tự ứng viên dựa trên từ khóa ngữ cảnh và thói quen người dùng.
        Sử dụng kho từ vựng phong phú để tránh bị lặp từ gốc sau nhiều vòng.
        Trả về: (chuỗi_sinh_ra, composite_entropy)
        """
        active_keywords = keywords or self.wordlist

        # Lựa chọn từ gốc linh hoạt giữa Profile, Dictionary mở rộng và Mô hình Markov Chain
        # Triệt tiêu tình trạng lặp từ sau hàng nghìn vòng đấu
        if user_profile and "name" in user_profile and random.random() < 0.35:
            base_keyword = user_profile["name"]
        elif self.config.enable_markov_generator and (random.random() < 0.50 or self.entropy_target > 3.4):
            # Tự do sinh từ tự nhiên qua mô hình ngôn ngữ Markov Chain mức ký tự
            base_keyword = self.markov_engine.generate_word()
        else:
            base_keyword = random.choice(active_keywords)

        # 1. Chọn kiểu viết hoa
        cap_style = random.choices([0, 1, 2, 3], weights=[0.2, 0.5, 0.2, 0.1])[0]
        staged = self.apply_capitalization_habit(base_keyword, cap_style)

        # 2. Sử dụng entropy_target kết hợp mutation_prob để điều khiển mức độ đột biến
        # Đảm bảo entropy_target là biến trạng thái sống (Active State Variable)
        effective_mutation = max(self.mutation_prob, min(0.85, (self.entropy_target - 2.2) / 2.5))
        staged = self.apply_leetspeak(staged, effective_mutation)

        # 3. Thêm hậu tố (năm sinh / số / ký tự lạ đặc biệt như @, #, !)
        if user_profile and "year" in user_profile and random.random() > 0.4:
            suffix = user_profile["year"]
        else:
            suffix = random.choice(self.config.common_suffixes)

        # Chèn ký tự phân cách lạ (@, _, #, $, .) khi entropy_target yêu cầu cao
        separator = ""
        if (effective_mutation > 0.25 or self.entropy_target > 3.0) and random.random() > 0.35:
            separator = random.choice(["@", "_", ".", "#", "$", "!"])

        candidate = f"{staged}{separator}{suffix}"
        
        # Nếu entropy_target ở mức cao (> 3.5), tăng cường chèn ký tự lạ và số để đạt mục tiêu
        if self.entropy_target > 3.5 and random.random() > 0.4:
            extra_salt = random.choice(["!", "@99", "#2026", "$$!", "@_@"])
            candidate = f"{candidate}{extra_salt}"

        entropy, _ = calculate_repetition_and_class_entropy(candidate)

        # Nếu entropy vẫn thấp hơn entropy_target, thực hiện thêm 1 bước đan xen để thỏa mãn mục tiêu
        if entropy < self.entropy_target and self.entropy_target > 3.2 and "@" not in candidate:
            candidate = candidate.replace("a", "@").replace("A", "@")
            entropy, _ = calculate_repetition_and_class_entropy(candidate)

        self.generation_history.append((candidate, entropy))
        return candidate, entropy

    def receive_penalty_feedback(self, steps_taken: int):
        """
        Cập nhật trọng số Tác nhân A khi bị Tác nhân B hội tụ nhanh (<= N bước):
        Tăng tỷ lệ đột biến và chuyển dịch sang phân phối có entropy cao hơn (có trần vật lý 4.5).
        """
        self.penalty_count += 1
        urgency = max(1.0, 5.0 - steps_taken)
        self.mutation_prob = min(0.85, round(self.mutation_prob + 0.05 * urgency, 3))
        # Giới hạn trần vật lý của entropy_target ở mức 4.5 bits (tránh phình vô hạn)
        self.entropy_target = min(4.5, round(self.entropy_target + 0.10 * urgency, 3))
        if "@!#$" not in self.config.common_suffixes:
            self.config.common_suffixes.extend(["@!#$", "2026_Secure!", "@@99!!", "@2026#"])

    def receive_success_feedback(self):
        """
        Khi đối thủ B không hội tụ được trong N bước:
        Giảm dần cả mutation_prob và entropy_target về mức cân bằng tự nhiên.
        """
        self.mutation_prob = max(0.15, round(self.mutation_prob - 0.01, 3))
        # Giảm entropy_target về mức sàn 2.5 bits
        self.entropy_target = max(2.5, round(self.entropy_target - 0.05, 3))

    def get_state(self) -> Dict[str, Any]:
        """Lấy trạng thái hiện tại của Tác nhân A để lưu checkpoint."""
        return {
            "mutation_prob": self.mutation_prob,
            "entropy_target": self.entropy_target,
            "penalty_count": self.penalty_count
        }

    def load_state(self, state: Dict[str, Any]):
        """Khôi phục trạng thái Tác nhân A từ checkpoint trước."""
        self.mutation_prob = float(state.get("mutation_prob", self.mutation_prob))
        self.entropy_target = float(state.get("entropy_target", self.entropy_target))
        self.penalty_count = int(state.get("penalty_count", self.penalty_count))
