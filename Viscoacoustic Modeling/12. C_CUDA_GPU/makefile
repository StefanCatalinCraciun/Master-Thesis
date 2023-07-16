# Makefile for CUDA SAXPY example

# Compiler
NVCC = nvcc

# Compiler flags
NVCCFLAGS = -O3

# Target executable
TARGET = wave

# Source files
SRCS = main.cu AC2D.cu diff.cu model.cu rec.cu src.cu #CUDA_kernel.cu

# Object files
OBJS = $(SRCS:.cu=.o)

all: $(TARGET) clean_objs

$(TARGET): $(OBJS)
	$(NVCC) $(NVCCFLAGS) -o $@ $^

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

clean_objs:
	rm -f $(OBJS)

clean: clean_objs
	rm -f $(TARGET)



