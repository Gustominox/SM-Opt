CC   := icpx

# CUDA flags (specific to nvcc)
NVFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda

SOURCES := $(wildcard src/*.cpp)
TARGETS := $(patsubst src/%.cpp, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.cpp | bin
	$(CC) $(NVFLAGS) -o $@ $<

clean:
	rm -rf bin
