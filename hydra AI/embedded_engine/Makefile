# Makefile for Neural-Entropy-Search AVX2 SIMD Engine
CXX = g++
CXXFLAGS = -O3 -mavx2 -shared -fPIC

# OS detection
ifeq ($(OS),Windows_NT)
    TARGET = libnnue.dll
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        TARGET = libnnue.dylib
    else
        TARGET = libnnue.so
    endif
endif

all: $(TARGET)

$(TARGET): nnue_engine.cpp nnue_engine.h nnue_weights.h
	$(CXX) $(CXXFLAGS) -o $(TARGET) nnue_engine.cpp

clean:
	rm -f *.dll *.so *.dylib *.a *.o *.bin
