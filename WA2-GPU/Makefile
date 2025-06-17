CC   := nvcc

# CUDA flags (specific to nvcc)
NVFLAGS := -O3 -D SIZE=$(SIZE)

SOURCES := $(wildcard src/*.cu)
TARGETS := $(patsubst src/%.cu, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.cu | bin
	$(CC) $(NVFLAGS) -o $@ $<

clean:
	rm -rf bin
