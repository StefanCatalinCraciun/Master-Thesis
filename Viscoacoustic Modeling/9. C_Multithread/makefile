CC = gcc ##clang for my macbook, gcc for my linux machine
CFLAGS = -Wall -Wextra -ansi -pedantic -std=c99 -Ofast -mavx512f -ffast-math -fopenmp #-march=native
LDFLAGS = -lm -fopenmp


SRCS = AC2D.c diff.c model.c src.c main.c rec.c
OBJS = $(SRCS:.c=.o)
HEADERS = AC2D.h diff.h model.h src.h types.h rec.h


all: wave clean

wave: $(OBJS)
	$(CC)  $(OBJS) -o wave $(LDFLAGS)

%.o: %.c $(HEADERS)
	$(CC) $(CFLAGS) -c $< -o $@

clean:
	rm  *.o
