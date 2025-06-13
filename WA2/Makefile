CC   := gcc
CFLAGS := -Wall -Wextra

SOURCES := $(wildcard src/*.c)
TARGETS := $(patsubst src/%.c, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.c | bin
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf bin
