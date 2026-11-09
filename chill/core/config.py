"""
Neural-Entropy-Search: Hệ thống đánh giá không gian mật khẩu bằng Heuristic Search & Self-Play.
Configuration & Hyperparameters.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class GeneratorConfig:
    """Cấu hình cho Tác nhân A: Profile-based Generator"""
    # Mẫu leetspeak thông dụng theo thói quen con người
    leetspeak_rules: Dict[str, List[str]] = field(default_factory=lambda: {
        'a': ['4', '@'],
        'b': ['8'],
        'e': ['3'],
        'i': ['1', '!'],
        'l': ['1', 'I'],
        'o': ['0'],
        's': ['5', '$'],
        't': ['7']
    })
    
    # Hậu tố thường gặp (năm sinh, số thứ tự, ký tự đặc biệt)
    common_suffixes: List[str] = field(default_factory=lambda: [
        "123", "123456", "2023", "2024", "2025", "2026", "!", "@", "#", "@123", "!@#",
        "888", "999", "$$$", "2024!", "2025@", "777", "admin1", "pass1"
    ])
    
    # Mở rộng kho từ ngữ phong phú (Linguistic Dictionary / Wordlist mở rộng)
    # Khắc phục triệt để vấn đề "Không gian từ gốc quá hẹp bị lặp lại sau hàng nghìn vòng"
    default_keywords: List[str] = field(default_factory=lambda: [
        # Nhóm quản trị & công nghệ
        "admin", "password", "neural", "entropy", "system", "security", "root", "database", "network",
        "server", "access", "master", "login", "cloud", "crypto", "gateway", "service", "matrix",
        "vector", "quantum", "kernel", "daemon", "terminal", "protocol", "cipher", "bytecode",
        # Nhóm địa danh & văn hóa phổ biến
        "vietnam", "hanoi", "saigon", "danang", "haiphong", "cantho", "tokyo", "seoul", "london",
        "berlin", "paris", "kyoto", "singapore", "sydney", "geneva", "madrid", "toronto",
        # Nhóm tên & danh từ đời sống, thiên văn, thần thoại
        "dragon", "tiger", "phoenix", "falcon", "hunter", "shadow", "winter", "summer",
        "coffee", "football", "guitar", "sunshine", "galaxy", "starlight", "valkyrie",
        "titan", "aurora", "nebula", "pulsar", "zenith", "solaris", "vortex", "hyperion",
        # Nhóm tên riêng thông dụng
        "alex", "david", "john", "michael", "nam", "linh", "long", "minh", "hoang"
    ])
    
    # Nạp wordlist mở rộng từ file bên ngoài (nếu có)
    wordlist_file: Optional[str] = None
    # Kích hoạt mô hình ngôn ngữ Markov Chain sinh ký tự tự do (Char-level Markov Chain)
    enable_markov_generator: bool = True

    # Tham số đột biến ban đầu
    base_mutation_prob: float = 0.2
    base_entropy_target: float = 3.0

@dataclass
class EstimatorConfig:
    """Cấu hình cho Tác nhân B: Entropy Estimator & Heuristic Tree Search"""
    input_dim: int = 64        # Kích thước vector biểu diễn chuỗi
    hidden_dim: int = 32       # Kích thước tầng ẩn neural network (nhẹ, dạng NNUE)
    output_dim: int = 1        # Điểm heuristic [0, 1]
    beam_width: int = 8        # Độ rộng beam search
    max_search_depth: int = 8  # Độ sâu tối đa duyệt cây
    convergence_threshold_n: int = 5  # Ngưỡng N bước theo thiết kế (N <= 5)
    learning_rate: float = 0.01

@dataclass
class MultiStageFilterConfig:
    """Cấu hình cơ chế sàng lọc Heuristic nhiều tầng (Multi-stage Filtering)"""
    # Ngưỡng rẽ nhánh Entropy (Đo bằng hàm toán học thuần túy trên CPU)
    # Ví dụ: 49095killseo (số lặp liên tiếp) -> Entropy ~0.715 < 2.0 (Vào nhánh nông)
    #         2b4e5b6e (xen kẽ cao)          -> Entropy ~3.770 >= 2.0 (Vào nhánh sâu)
    entropy_cutoff_threshold: float = 2.0
    # Độ sâu tìm kiếm của nhánh cây nông (Shallow Search)
    shallow_max_depth: int = 3
    shallow_beam_width: int = 5

@dataclass
class SystemConfig:
    """Cấu hình toàn hệ thống"""
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    estimator: EstimatorConfig = field(default_factory=EstimatorConfig)
    filtering: MultiStageFilterConfig = field(default_factory=MultiStageFilterConfig)
    candidate_pool_size: int = 50
    classification_threshold: float = 0.40
    rate_limit_capacity: int = 100       # Token bucket capacity
    rate_limit_refill_rate: float = 50.0 # Tokens per second
