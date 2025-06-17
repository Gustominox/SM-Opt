CC   := gcc
NVCC := nvcc
ICPX := icpx
# CFLAGS := -Wall -Wextra 
CFLAGS := -Wall -Wextra -D SIZE=$(SIZE) -fopenmp
CFLAGS += -funroll-loops 
CFLAGS += -Ofast 
CFLAGS += -ftree-vectorize
CFLAGS += -mtune=native

NVFLAGS := -O3 -D SIZE=$(SIZE)
CXXFLAGS := -fsycl -fsycl-targets=nvptx64-nvidia-cuda -O3 -D SIZE=$(SIZE)

.PHONY: all clean

# Default target if OUTFILE is not provided
all:
	@echo "This Makefile is designed to be called from the perf scripts."
	@echo "Example: make SIZE=1024 OUTFILE=sparse_1024"

bin:
	mkdir -p bin

# This logic handles calls from the perf scripts
ifneq ($(OUTFILE),)

# Extract the base name of the program from OUTFILE. e.g. sparse_1024 -> sparse
STEM := $(patsubst %_$(SIZE),%,$(OUTFILE))
SOURCE := $(wildcard src/$(STEM).c src/$(STEM).cu src/$(STEM).cpp)
TARGET := bin/$(OUTFILE)

# The main build rule that the scripts will trigger.
all: $(TARGET)

$(TARGET): $(SOURCE) | bin
	@echo "Building $(TARGET) from $(SOURCE)"
ifeq ($(suffix $(SOURCE)),.c)
	$(CC) $(CFLAGS) -o $@ $<
else ifeq ($(suffix $(SOURCE)),.cu)
	$(NVCC) $(NVFLAGS) -o $@ $<
else ifeq ($(suffix $(SOURCE)),.cpp)
	$(ICPX) $(CXXFLAGS) -o $@ $<
else
	@echo "Error: No source file found for stem '$(STEM)' from OUTFILE '$(OUTFILE)'"
	@exit 1
endif

endif

clean:
	rm -rf bin