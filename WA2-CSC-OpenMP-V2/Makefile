CC   := gcc
# CFLAGS := -Wall -Wextra 
CFLAGS := -Wall -Wextra -D SIZE=$(SIZE)
CFLAGS += -funroll-loops 
CFLAGS += -Ofast 
CFLAGS += -ftree-vectorize
CFLAGS += -mtune=native
CFLAGS += -fopenmp

SOURCES := $(wildcard src/*.c)
TARGETS := $(patsubst src/%.c, bin/%, $(SOURCES))

all: $(TARGETS) 

bin:
	mkdir -p bin

bin/%: src/%.c | bin
	$(CC) $(CFLAGS) -o bin/$(OUTFILE) $<

clean:
	rm -rf bin
