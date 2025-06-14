CC   := gcc
CFLAGS := -Wall -Wextra -D SIZE=$(SIZE)

SOURCES := $(wildcard src/*.c)
TARGETS := $(patsubst src/%.c, bin/%, $(SOURCES))

all: $(TARGETS)

bin:
	mkdir -p bin

bin/%: src/%.c | bin
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf bin
