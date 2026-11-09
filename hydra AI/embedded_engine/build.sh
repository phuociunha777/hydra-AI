#!/bin/bash
set -e

echo "[Build] Biên dịch động cơ SIMD AVX2 cục bộ..."
if [[ "$OSTYPE" == "darwin"* ]]; then
    g++ -O3 -mavx2 -shared -fPIC -o libnnue.dylib nnue_engine.cpp
    echo "[Build] Biên dịch thành công libnnue.dylib!"
else
    g++ -O3 -mavx2 -shared -fPIC -o libnnue.so nnue_engine.cpp
    echo "[Build] Biên dịch thành công libnnue.so!"
fi
