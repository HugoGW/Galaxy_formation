# Builds the gravity / rendering engine. engine.py also does this automatically
# the first time the simulation is run (or when nbody.cpp is newer than the library).

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -ffast-math -fopenmp -std=c++17 -shared -fPIC
OUT      := ../build/libnbody.so

$(OUT): nbody.cpp
	mkdir -p ../build
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	rm -f $(OUT)

.PHONY: clean
