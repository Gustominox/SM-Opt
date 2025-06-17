# CUDA compiler and flags.
NVCC   := nvcc
CFLAGS := -O3

SOURCES := $(wildcard src/*.cu)
TARGETS := $(patsubst src/%.cu, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.cu | bin
	$(NVCC) $(CFLAGS) -o $@ $<

clean:
	rm -rf bin
