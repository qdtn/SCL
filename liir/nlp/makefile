CC = gcc
#Using -Ofast instead of -O3 might result in faster code, but is supported only by newer GCC versions
CFLAGS = -lm -pthread -O3 -march=native -Wall -funroll-loops -Wno-unused-result

all: picky_word2vec

picky_word2vec : picky_word2vec.c
	$(CC) picky_word2vec.c -o picky_word2vec $(CFLAGS)
clean:
	rm -rf picky_word2vec