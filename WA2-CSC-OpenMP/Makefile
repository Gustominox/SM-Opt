CC   := gcc
# CFLAGS := -Wall -Wextra 
CFLAGS := -Wall -Wextra 
# CFLAGS += -funroll-loops 
# CFLAGS += -Ofast 
# CFLAGS += -ftree-vectorize
# CFLAGS += -mtune=native

SOURCES := $(wildcard src/*.c)
TARGETS := $(patsubst src/%.c, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.c | bin
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf bin
