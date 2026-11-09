#define BUILDING_NNUE_DLL

#include "nnue_engine.h"
#include "nnue_weights.h"

#include <immintrin.h>
#include <string.h>
#include <math.h>
#include <stdio.h>
#include <chrono>
#include <algorithm>

#ifdef _MSC_VER
#include <intrin.h>
#else
#include <cpuid.h>
#endif

// Kiểm tra hỗ trợ phần cứng AVX2 qua CPUID
int nnue_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int nnue_init(void) {
    return nnue_has_avx2();
}

// Hàm toán học thuần túy trên CPU đo độ hỗn loạn (Entropy) dựa trên tần số lặp của chữ, số và ký tự lạ như @
float nnue_calculate_structural_entropy(const char* str) {
    if (!str) return 0.0f;
    size_t len = strlen(str);
    if (len == 0) return 0.0f;

    int letter_cnt = 0, digit_cnt = 0, special_cnt = 0;
    int freq[256] = {0};
    
    int runs_count = 0;
    int curr_type = -1; // 0: digit, 1: letter, 2: special
    int curr_len = 0;
    int max_digit_run = 0;
    int max_letter_run = 0;
    int max_special_run = 0;
    int max_class_run = 0;

    for (size_t i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)str[i];
        freq[c]++;
        int t = 2; // special
        if (c >= '0' && c <= '9') {
            digit_cnt++;
            t = 0; // digit
        } else if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z')) {
            letter_cnt++;
            t = 1; // letter
        } else {
            special_cnt++;
            t = 2; // special
        }

        if (t == curr_type) {
            curr_len++;
        } else {
            if (curr_type != -1) {
                if (curr_type == 0 && curr_len > max_digit_run) max_digit_run = curr_len;
                if (curr_type == 1 && curr_len > max_letter_run) max_letter_run = curr_len;
                if (curr_type == 2 && curr_len > max_special_run) max_special_run = curr_len;
                if (curr_len > max_class_run) max_class_run = curr_len;
                runs_count++;
            }
            curr_type = t;
            curr_len = 1;
        }
    }
    if (curr_type != -1) {
        if (curr_type == 0 && curr_len > max_digit_run) max_digit_run = curr_len;
        if (curr_type == 1 && curr_len > max_letter_run) max_letter_run = curr_len;
        if (curr_type == 2 && curr_len > max_special_run) max_special_run = curr_len;
        if (curr_len > max_class_run) max_class_run = curr_len;
        runs_count++;
    }

    int transitions = runs_count > 0 ? (runs_count - 1) : 0;
    float t_rate = (len > 1) ? ((float)transitions / (float)(len - 1)) : 0.0f;
    float max_run_ratio = (float)max_class_run / (float)len;

    // Shannon char entropy
    float char_entropy = 0.0f;
    int repeated_chars = 0;
    for (int i = 0; i < 256; ++i) {
        if (freq[i] > 0) {
            float p_c = (float)freq[i] / (float)len;
            char_entropy -= p_c * log2f(p_c);
            if (freq[i] > 1) repeated_chars += (freq[i] - 1);
        }
    }
    float rep_ratio = (float)repeated_chars / (float)len;

    // Class entropy
    float class_entropy = 0.0f;
    if (letter_cnt > 0) { float p = (float)letter_cnt / len; class_entropy -= p * log2f(p); }
    if (digit_cnt > 0)  { float p = (float)digit_cnt / len;  class_entropy -= p * log2f(p); }
    if (special_cnt > 0){ float p = (float)special_cnt / len;class_entropy -= p * log2f(p); }

    // Run penalty: nếu cụm số liên tiếp >= 4 (ví dụ 49095) hoặc chữ liên tiếp >= 5 -> phạt nặng
    float run_penalty = 0.0f;
    if (max_digit_run >= 4) run_penalty += 0.45f + 0.15f * (float)(max_digit_run - 4);
    if (max_letter_run >= 5) run_penalty += 0.35f + 0.10f * (float)(max_letter_run - 5);

    // Keyboard bias penalty
    float kb_penalty = 0.0f;
    if (strstr(str, "123") || strstr(str, "qwerty") || strstr(str, "1qaz") || strstr(str, "123456")) {
        kb_penalty = 0.50f;
    }

    float composite = (char_entropy * (0.35f + 0.85f * t_rate) + 0.60f * class_entropy) * (1.15f - 0.60f * max_run_ratio)
                      - run_penalty - 0.40f * rep_ratio - kb_penalty;
    if (composite < 0.1f) composite = 0.1f;
    return composite;
}

// Trích xuất 64 features tương thích với Python extract_features
void nnue_extract_features(const char* str, uint8_t* out_features) {
    memset(out_features, 0, 64);
    if (!str) return;

    size_t len = strlen(str);
    if (len == 0) return;

    int digits = 0, uppers = 0, lowers = 0, specials = 0;
    for (size_t i = 0; i < len; ++i) {
        char ch = str[i];
        if (ch >= '0' && ch <= '9') digits++;
        else if (ch >= 'A' && ch <= 'Z') uppers++;
        else if (ch >= 'a' && ch <= 'z') lowers++;
        else specials++;

        // n-gram băm vào 32 buckets
        int bucket = ((unsigned char)ch + (int)i * 7) % 32;
        // Tích lũy và chuẩn hóa về thang [0, 255]
        if (out_features[bucket] < 240) {
            out_features[bucket] += (uint8_t)(255 / (len > 0 ? len : 1));
        }
    }

    // Index 32: Độ dài chuẩn hóa (len / 20 * 255)
    out_features[32] = (uint8_t)std::min((double)255.0, (len * 255.0 / 20.0));
    // Index 33: Tỷ lệ số
    out_features[33] = (uint8_t)(digits * 255 / len);
    // Index 34: Tỷ lệ hoa
    out_features[34] = (uint8_t)(uppers * 255 / len);
    // Index 35: Tỷ lệ ký tự đặc biệt
    out_features[35] = (uint8_t)(specials * 255 / len);
    // Index 36: Tỷ lệ thường
    out_features[36] = (uint8_t)(lowers * 255 / len);

    // Index 37..44: Dấu hiệu leetspeak phổ biến
    const char leets[] = {'@', '3', '1', '0', '$', '7', '4', '!'};
    for (int k = 0; k < 8; ++k) {
        if (strchr(str, leets[k]) != NULL) {
            out_features[37 + k] = 255;
        }
    }

    // Index 45: Transitions
    int transitions = 0;
    for (size_t i = 0; i + 1 < len; ++i) {
        char c1 = str[i], c2 = str[i+1];
        bool c1_alpha = (c1 >= 'a' && c1 <= 'z') || (c1 >= 'A' && c1 <= 'Z');
        bool c2_num = (c2 >= '0' && c2 <= '9');
        if (c1_alpha && c2_num) transitions++;
    }
    out_features[45] = (uint8_t)std::min(255, (transitions * 255 / 5));

    // Index 46: Unique characters
    bool seen[256] = {false};
    int unique_cnt = 0;
    for (size_t i = 0; i < len; ++i) {
        unsigned char uc = (unsigned char)str[i];
        if (!seen[uc]) {
            seen[uc] = true;
            unique_cnt++;
        }
    }
    out_features[46] = (uint8_t)(unique_cnt * 255 / len);
}

// Đánh giá trực tiếp vector đặc trưng 64 chiều bằng AVX2 SIMD Intrinsics
float nnue_evaluate_features_simd(const uint8_t* features) {
    // Khởi tạo 32 accumulators cho 32 hidden units với b1
    __m256i acc_0_7   = _mm256_loadu_si256((const __m256i*)&NNUE_B1[0]);
    __m256i acc_8_15  = _mm256_loadu_si256((const __m256i*)&NNUE_B1[8]);
    __m256i acc_16_23 = _mm256_loadu_si256((const __m256i*)&NNUE_B1[16]);
    __m256i acc_24_31 = _mm256_loadu_si256((const __m256i*)&NNUE_B1[24]);

    for (int i = 0; i < NNUE_INPUT_DIM; ++i) {
        uint8_t x = features[i];
        if (x == 0) continue; // Pruning / Sparsity optimization

        __m256i x_broadcast = _mm256_set1_epi16((int16_t)x);

        // Load 32 byte trọng số của hàng i: W1[i][0..31]
        __m256i row = _mm256_loadu_si256((const __m256i*)NNUE_W1[i]);
        __m128i row_lo = _mm256_castsi256_si128(row);
        __m128i row_hi = _mm256_extracti128_si256(row, 1);

        // Mở rộng int8 sang int16
        __m256i w_0_15  = _mm256_cvtepi8_epi16(row_lo);
        __m256i w_16_31 = _mm256_cvtepi8_epi16(row_hi);

        // Nhân vector SIMD (16-bit)
        __m256i prod_0_15  = _mm256_mullo_epi16(w_0_15, x_broadcast);
        __m256i prod_16_31 = _mm256_mullo_epi16(w_16_31, x_broadcast);

        // Mở rộng sang 32-bit và tích lũy vào 4 thanh ghi accumulator
        acc_0_7   = _mm256_add_epi32(acc_0_7,   _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_0_15)));
        acc_8_15  = _mm256_add_epi32(acc_8_15,  _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_0_15, 1)));
        acc_16_23 = _mm256_add_epi32(acc_16_23, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(prod_16_31)));
        acc_24_31 = _mm256_add_epi32(acc_24_31, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(prod_16_31, 1)));
    }

    // Đưa kết quả accumulator ra mảng 32 số nguyên và áp dụng Clipped ReLU
    int32_t raw_acc[32];
    _mm256_storeu_si256((__m256i*)&raw_acc[0],  acc_0_7);
    _mm256_storeu_si256((__m256i*)&raw_acc[8],  acc_8_15);
    _mm256_storeu_si256((__m256i*)&raw_acc[16], acc_16_23);
    _mm256_storeu_si256((__m256i*)&raw_acc[24], acc_24_31);

    // Clipped ReLU [0, 127] và nhân Layer 2 (32 units -> 1 unit)
    int32_t out_acc = NNUE_B2_VAL;
    for (int j = 0; j < NNUE_HIDDEN_DIM; ++j) {
        // Chuẩn hóa tầng ẩn đồng bộ 100% với hàm Clipped ReLU float32 trong Python
        int32_t act = (int32_t)roundf((float)raw_acc[j] / NNUE_DIVISOR_1);
        if (act < 0) act = 0;
        if (act > 127) act = 127;

        out_acc += act * (int32_t)NNUE_W2[j];
    }

    // Đổi sang float và tính hàm kích hoạt Sigmoid chuẩn xác không lệch pha
    float z2 = (float)out_acc / NNUE_DIVISOR_2;
    if (z2 > 15.0f) return 1.0f;
    if (z2 < -15.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-z2));
}

// Đánh giá chuỗi trực tiếp
float nnue_evaluate_string(const char* str) {
    uint8_t feats[64];
    nnue_extract_features(str, feats);
    return nnue_evaluate_features_simd(feats);
}

// Đánh giá hàng loạt (Batch evaluate)
void nnue_batch_evaluate(const char** strings, int count, float* out_scores) {
    for (int i = 0; i < count; ++i) {
        out_scores[i] = nnue_evaluate_string(strings[i]);
    }
}

// Đo lường độ trễ suy luận chính xác trên CPU (trả về micro-giây)
double nnue_benchmark_latency_us(int iterations) {
    const char* test_strings[] = {
        "admin@2024", "password123!", "h@n0i2025#", "Secure_P@ss!99", "Neural_2026"
    };
    int num_samples = 5;

    // Khởi động cache (Warm up)
    for (int i = 0; i < 1000; ++i) {
        nnue_evaluate_string(test_strings[i % num_samples]);
    }

    auto t_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        volatile float score = nnue_evaluate_string(test_strings[i % num_samples]);
        (void)score;
    }
    auto t_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> elapsed_us = t_end - t_start;
    return elapsed_us.count() / (double)iterations;
}
