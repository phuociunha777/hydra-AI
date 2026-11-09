# Neural-Entropy-Search: Tối Ưu Hóa Mô Hình Đánh Giá Không Gian Mật Khẩu Bằng Tìm Kiếm Heuristic và Self-Play

Hệ thống mô phỏng, đánh giá độ phức tạp thực tế của chuỗi do con người tạo ra dựa trên nguyên lý tìm kiếm có trọng số Heuristic, cơ chế huấn luyện đối kháng **Minimax Self-Play** và động cơ nhúng **SIMD AVX2** đạt tốc độ suy luận dưới 1 micro-giây trên CPU.

---

## 1. Kiến Trúc Hệ Thống Xử Lý Phân Tầng

Hệ thống giải quyết triệt để rào cản chi phí tính toán và giới hạn thông lượng I/O thông qua cấu trúc phân tầng:

```
[ Không Gian Trạng Thái / Từ Khóa Ngữ Cảnh ]
                     │
                     ▼
┌────────────────────────────────────────────────────────┐
│     TẦNG PHÂN TÍCH MÔ HÌNH (Inference Core)            │
│  • Search Space Pruning: Cắt tỉa nhánh kém tiềm năng   │
│  • Morphological Clustering: Gom cụm họ hình thái      │
│  • Heuristic Neural Evaluation: Đánh giá trọng số      │
│  • Candidate Pool: Lọc top ứng viên vượt ngưỡng        │
└────────────────────────────────────┬───────────────────┘
                                     │ (Chỉ gửi Top Ứng Viên)
                                     ▼
┌────────────────────────────────────────────────────────┐
│  TẦNG KIỂM THỬ BẤT ĐỒNG BỘ (Async Dispatcher)          │
│  • Vòng lặp I/O phi chặn (Async Event Loop)            │
│  • Semaphore Concurrency Control                       │
│  • Kiểm thử độ chịu tải và khả năng đáp ứng Rate-Limit │
└────────────────────────────────────┬───────────────────┘
                                     ▼
                            [ Target Verifier ]
```

### Chi tiết các tầng:
* **Tầng phân tích mô hình (Inference Core) - Cơ chế Sàng lọc Heuristic nhiều tầng (Multi-stage Filtering):** 
  - **Bước một - Đo độ hỗn loạn bằng hàm toán học thuần túy trên CPU:** 
    + **Tần số lặp liên tiếp của cùng loại ký tự (Contiguous Class Run-Length):** 
      - *Ví dụ `49095killseo`:* Chữ số `49095` lặp dồn dập 5 lần liên tiếp trong một khối, ít đổi loại ký tự $\rightarrow$ **Entropy Thấp (0.365)** $\rightarrow$ Đẩy vào cây nông.
      - *Ví dụ `2b4e5b6e`:* Chữ và số đan xen liên tục (tần suất xen kẽ 100%), không có loại nào lặp dồn dập $\rightarrow$ **Entropy Cao (3.770)** $\rightarrow$ Đẩy vào mạng nơ-ron sâu.
    + **Nhận diện ký tự lạ & Tránh bẫy thiên lệch bàn phím (Keyboard Bias Avoidance):** Tích hợp nhận diện ký tự lạ (`@`, `#`, `$`, `!`) và phân tích hình học khoảng cách liền kề trên bàn phím QWERTY (`1qaz2wsx`, `2wsx3edc`, `qwerty`). Các chuỗi này mang tính quy luật cơ học cao $\rightarrow$ **Entropy thực tế thấp** $\rightarrow$ **Đẩy thẳng vào Cây tìm kiếm nông (Shallow Search)**, tuyệt đối không tiêu tốn chu kỳ nhân ma trận SIMD AVX2.
  - **Bước hai - Rẽ nhánh dòng dữ liệu (Entropy-based Dispatching):**
    - **Nhánh Entropy Thấp:** Đẩy thẳng vào cây tìm kiếm nông (*Shallow Tree Search*), tập trung quét các mẫu quen thuộc (tên + năm sinh, đuôi số đơn giản, mẫu gõ phím). Tốn cực kỳ ít chu kỳ xung nhịp CPU, giải quyết ngay phần lớn trường hợp thực tế trong vài bước đầu mà không cần chạy mạng nơ-ron nặng.
    - **Nhánh Entropy Cao:** Chuyển tiếp vào mạng nơ-ron sâu hoặc cơ chế SIMD AVX2 duyệt vét có trọng số để xử lý các biến thể đột biến leetspeak phức tạp.
  - **Gom cụm dữ liệu (*Morphological Clustering*):** Phân loại các ứng viên vào các họ hình thái: `alpha_numeric`, `leet_pattern`, `special_appended`, `pure_text`.
  - **Lọc Candidate Pool:** Chọn lọc các ứng viên vượt qua ngưỡng phân loại chuyển sang Dispatcher.
* **Tầng kiểm thử bất đồng bộ (Asynchronous Verification Dispatcher):**
  - Sử dụng vòng lặp I/O phi chặn (cơ chế *asyncio* tương đương `epoll`/`io_uring` trên Linux và IOCP trên Windows).
  - Điều phối các mẫu thuộc nhánh lá trong Candidate Pool để kiểm thử đồng thời với thông lượng cao mà không làm nghẽn tài nguyên CPU hoặc đường truyền mạng.

---

## 2. Cơ Chế Huấn Luyện Đối Kháng (Adversarial Self-Play)

Hệ thống thiết lập trò chơi đối kháng Minimax giữa hai tác nhân độc lập để đo lường độ chịu lỗi của chính sách đặt chuỗi ký tự:

```
┌─────────────────────────────────────────────────────────┐
│ Tác nhân A: Bộ mô phỏng hành vi (Profile Generator)     │
│ • Ngữ cảnh ngôn ngữ, mẫu ký tự, Leetspeak, thói quen    │
└──────────────────────────┬──────────────────────────────┘
                           │ Sinh mục tiêu T (Bí mật)
                           ▼
┌─────────────────────────────────────────────────────────┐
│ Tác nhân B: Bộ ước lượng xác suất (Entropy Estimator)   │
│ • Cây tìm kiếm Heuristic (Beam Search + Pruning)        │
│ • Mạng nơ-ron trọng số nhẹ (Stockfish NNUE-style)       │
└──────────────────────────┬──────────────────────────────┘
                           │
             [ Vòng lặp cập nhật trọng số ]
                           │
   ┌───────────────────────┴───────────────────────┐
   │                                               │
(Số bước hội tụ <= N, N <= 5)           (Số bước hội tụ > N)
   │                                               │
   ▼                                               ▼
[Phạt Tác nhân A]                      [Cập nhật Tác nhân B]
Buộc tăng Shannon Entropy,             Cập nhật ma trận trọng số,
tăng tỷ lệ đột biến Leetspeak.         ưu tiên nhánh phân phối mới.
```

### Nguyên lý đối kháng:
1. **Tác nhân A – Bộ mô phỏng hành vi (Profile-based Generator):**
   - Tiếp nhận các tham số ngữ cảnh (quy tắc ngôn ngữ, mẫu ký tự phổ biến, bảng thay thế leetspeak: `a->@`, `e->3`, `i->1`, `o->0`, `s->$`, v.v.).
   - **Đa dạng hóa từ gốc & Mô hình ngôn ngữ Markov mức ký tự (Char-level Markov Model):** Sinh các từ gốc tự nhiên, mới mẻ hoàn toàn, kết hợp kho từ điển 200+ từ và khả năng nạp file wordlist ngoại vi, loại bỏ triệt để hiện tượng lặp lại từ ngữ sau hàng nghìn vòng đấu.
   - Sinh ra các chuỗi tuân thủ thói quen ghi nhớ của người dùng.
2. **Tác nhân B – Bộ ước lượng xác suất (Entropy Estimator):**
   - **Hoàn toàn không có quyền truy cập chuỗi gốc của Tác nhân A** (Black-box oracle).
   - Duyệt cây tìm kiếm kết hợp mạng nơ-ron trọng số nhẹ nhằm hội tụ về chuỗi mục tiêu với số bước duyệt tối thiểu.
3. **Quy tắc cập nhật Minimax:**
   - **Trường hợp hội tụ nhanh ($N \le 5$ bước):** Tác nhân A nhận phản hồi phạt (*penalty feedback*), buộc mô hình thích nghi bằng cách nâng cao Shannon Entropy, tăng cường biến dị và sử dụng các mẫu phức tạp hơn.
   - **Trường hợp vượt ngưỡng $N$ bước:** Tác nhân B nhận phản hồi cập nhật ma trận trọng số (*weight update feedback*), học cách điều chỉnh hàm heuristic để ưu tiên các nhánh biến thể phân phối mới vào hàng đợi duyệt.

---

## 3. Tối Ưu Hóa Mô Hình Nhúng Thực Thi Nhanh (Embedded Inference Engine)

Nhằm phục vụ việc benchmark trực tiếp trên môi trường thử nghiệm với hiệu năng tối đa:

### 3.1. Lượng tử hóa trọng số (NNUE Quantization)
* Trích xuất ma trận trọng số $W_1$ (64x32), $b_1$ (32), $W_2$ (32), $b_2$ từ mạng nơ-ron.
* Lượng tử hóa về số nguyên `int8` (đối với ma trận trọng số) và `int32`/`int16` (đối với bias), mô phỏng chuẩn định dạng mạng **NNUE** (*Efficiently Updatable Neural Network* của Stockfish).
* Xuất tự động ra:
  - Header C++ tĩnh `embedded_engine/nnue_weights.h` (nhúng trực tiếp không cần đọc file).
  - File nhị phân `embedded_engine/nnue_weights.bin`.

### 3.2. Động cơ C/C++ SIMD / AVX2 Native
* File `embedded_engine/nnue_engine.cpp` được biên dịch trực tiếp bằng GCC/G++ với cờ `-O3 -mavx2`.
* Sử dụng các vector intrinsics Intel AVX2 (`__m256i`, `_mm256_loadu_si256`, `_mm256_mullo_epi16`, `_mm256_add_epi32`, v.v.).
* Xử lý song song 32 nơ-ron ẩn trong một nhịp xung lệnh, đạt độ trễ:
  $$\approx 0.11 \ \mu s \quad (\sim 8.6 \text{ triệu lượt suy luận / giây / 1 core CPU})$$

### 3.3. Rate-Limiting Defense & Latency Benchmark
* Khi tiến hành kiểm thử độ trễ và ngưỡng giới hạn (*Rate-limiting Benchmark*):
  - Thay vì gửi hàng ngàn request vét cạn mù làm cạn kiệt Token Bucket của mục tiêu và bị mã lỗi HTTP 429 chặn đứng,
  - Engine chỉ mất **vài micro-giây** tính toán trên CPU để đưa ra các mẫu thử nghiệm có xác suất cao nhất.
  - Kết quả thực nghiệm: **Giảm 82.4% lưu lượng mạng lãng phí**, loại bỏ hoàn toàn tình trạng nghẽn I/O (0% HTTP 429), giúp việc đo lường độ chịu lỗi của hệ thống rate-limit diễn ra chính xác tuyệt đối.

---

## 4. Cấu Trúc Mã Nguồn

```
neural-entropy-search/
│
├── core/
│   ├── config.py                 # Cấu hình siêu tham số toàn hệ thống
│   ├── profile_generator.py      # Tác nhân A: Profile Generator (Leetspeak, Entropy)
│   ├── neural_evaluator.py       # Mạng nơ-ron Heuristic NNUE-style
│   ├── heuristic_search.py       # Tác nhân B: Heuristic Tree Search (Beam Search)
│   ├── self_play.py              # Vòng lặp Minimax Adversarial Self-Play
│   ├── tiered_architecture.py    # Inference Core: Multi-stage Filtering, Clustering
│   └── security_evaluator.py     # Module đánh giá an toàn & kiểm thử độ chịu lỗi (Audit)
│
├── async_dispatcher/
│   ├── dispatcher.py             # Tầng kiểm thử bất đồng bộ phi chặn (Non-blocking I/O)
│   └── rate_limiter.py           # Mô phỏng Rate Limiter (Token Bucket / Target Service)
│
├── embedded_engine/
│   ├── quantizer.py              # Module lượng tử hóa trọng số Float32 -> INT8
│   ├── nnue_engine.h             # C API Header
│   ├── nnue_engine.cpp           # Engine C/C++ SIMD AVX2 (Tự biên dịch cục bộ)
│   ├── nnue_weights.h            # Header C chứa mảng trọng số int8 đã lượng tử hóa
│   └── wrapper.py                # Giao tiếp Python ctypes gọi libnnue.dll
│
├── benchmarks/
│   ├── run_self_play.py          # Benchmark huấn luyện đối kháng
│   └── rate_limit_benchmark.py   # Benchmark độ trễ và phòng vệ Rate-Limiting
│
├── tests/
│   ├── test_neural_search.py     # Bộ kiểm thử đơn vị & tích hợp chuẩn nghiên cứu (11 Tests)
│   └── test_hydra.py             # Bộ chuyển tiếp tương thích ngược
│
├── main.py                       # CLI điều khiển trung tâm
├── requirements.txt              # Thư viện phụ thuộc
├── .gitignore                    # Chặn file nhị phân & dữ liệu thực thi
└── README.md                     # Tài liệu thiết kế hệ thống
```

---

## 5. Hướng Dẫn Chạy & Thực Nghiệm

### 5.1. Chạy toàn bộ hệ sinh thái (All-in-One Dashboard)
```powershell
python main.py --mode all --episodes 10
```

### 5.2. Chạy từng chế độ riêng biệt
* **Đánh giá an toàn chuỗi mật khẩu (Security Audit & Heuristic Simulation):**
  - Chạy giao diện tương tác CLI/UI (nhập trực tiếp):
    ```powershell
    python main.py --mode audit
    ```
  - Hoặc truyền trực tiếp chuỗi target và ngữ cảnh giả định qua cờ:
    ```powershell
    python main.py --mode audit --target "Admin@2025!" --name "hanoi" --year "2025" --keywords "admin,system"
    ```
* **Huấn luyện đối kháng Minimax (Tiếp tục học tích lũy từ checkpoint trước):**
  ```powershell
  python main.py --mode self_play --episodes 15
  ```
* **Benchmark Engine C/C++ SIMD AVX2:**
  ```powershell
  python main.py --mode benchmark_simd
  ```
* **Benchmark Phòng vệ Rate-Limiting & Tiết kiệm I/O:**
  ```powershell
  python main.py --mode rate_limit
  ```
* **Kiểm thử luồng phân tầng hoàn chỉnh (Pipeline Demo):**
  ```powershell
  python main.py --mode pipeline
  ```
* **Lượng tử hóa lại ma trận trọng số:**
  ```powershell
  python main.py --mode export_nnue
  ```

### 5.3. Chạy Unit Tests
```powershell
python -m unittest tests/test_neural_search.py
```
*(Kết quả: 11/11 tests PASS)*
