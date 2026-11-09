@echo off
echo [Build] Bien dich dong co SIMD AVX2 cuc bo (libnnue.dll)...
g++ -O3 -mavx2 -shared -o libnnue.dll nnue_engine.cpp
if %ERRORLEVEL% EQU 0 (
    echo [Build] Bien dich thanh cong libnnue.dll!
) else (
    echo [Build] Loi khi bien dich! Vui long kiem tra trinh bien dich g++.
)
