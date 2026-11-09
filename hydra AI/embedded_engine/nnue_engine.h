#ifndef NNUE_ENGINE_H
#define NNUE_ENGINE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

#if defined(_WIN32) || defined(__CYGWIN__)
  #ifdef BUILDING_NNUE_DLL
    #define NNUE_API __declspec(dllexport)
  #else
    #define NNUE_API __declspec(dllimport)
  #endif
#else
  #define NNUE_API __attribute__((visibility("default")))
#endif

// Khởi tạo engine và cấu hình AVX2
NNUE_API int nnue_init(void);

// Trích xuất vector đặc trưng 64 chiều từ chuỗi ký tự
NNUE_API void nnue_extract_features(const char* str, uint8_t* out_features);

// Đánh giá trực tiếp từ vector đặc trưng 64 chiều bằng AVX2 SIMD
NNUE_API float nnue_evaluate_features_simd(const uint8_t* features);

// Hàm toán học thuần túy trên CPU đo độ hỗn loạn (Entropy) dựa trên tần số lặp của chữ, số và ký tự lạ như @
NNUE_API float nnue_calculate_structural_entropy(const char* str);

// Đánh giá chuỗi ký tự (trích xuất đặc trưng + suy luận SIMD)
NNUE_API float nnue_evaluate_string(const char* str);

// Đánh giá hàng loạt mảng chuỗi
NNUE_API void nnue_batch_evaluate(const char** strings, int count, float* out_scores);

// Benchmark độ trễ suy luận trên CPU (trả về thời gian trung bình tính bằng micro-giây)
NNUE_API double nnue_benchmark_latency_us(int iterations);

// Kiểm tra tính sẵn sàng của SIMD AVX2
NNUE_API int nnue_has_avx2(void);

#ifdef __cplusplus
}
#endif

#endif // NNUE_ENGINE_H
