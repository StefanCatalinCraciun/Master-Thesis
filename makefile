CC = gcc
CFLAGS = -Ofast -march=native -mavx512f -ffast-math -fopenmp
NVCC = nvcc

OUTPUT_DIR_SINGLE = Saxpy/src/Single_Core_CPU/C/bin
EXECUTABLE_SINGLE = $(OUTPUT_DIR_SINGLE)/Single_C_For_Loop_CPU

OUTPUT_DIR_MULTI = Saxpy/src/Multiple_Core_CPU/C/bin
EXECUTABLE_MULTI = $(OUTPUT_DIR_MULTI)/Saxpy_C_For_Loop_Multithread_CPU

OUTPUT_DIR_GPU = Saxpy/src/GPU/C/bin
EXECUTABLE_GPU = $(OUTPUT_DIR_GPU)/Saxpy_C_CUDA_GPU

JULIA_SCRIPT = Saxpy/src/Multiple_Core_CPU/Julia/Saxpy_Julia_Multithread_CPU.jl 

all: $(EXECUTABLE_SINGLE) $(EXECUTABLE_MULTI) $(EXECUTABLE_GPU)

$(EXECUTABLE_SINGLE): Saxpy/src/Single_Core_CPU/C/Saxpy_C_For_Loop_CPU.c
	$(CC) $^ -o $(EXECUTABLE_SINGLE) $(CFLAGS)

$(EXECUTABLE_MULTI): Saxpy/src/Multiple_Core_CPU/C/Saxpy_C_For_Loop_Multithread_CPU.c
	$(CC) $^ -o $(EXECUTABLE_MULTI) $(CFLAGS)

$(EXECUTABLE_GPU): Saxpy/src/GPU/C/Saxpy_C_CUDA_GPU.cu
	$(NVCC) $^ -o $(EXECUTABLE_GPU)

run_single: $(EXECUTABLE_SINGLE)
	$(EXECUTABLE_SINGLE)

run_multi: $(EXECUTABLE_MULTI)
	$(EXECUTABLE_MULTI)

run_gpu: $(EXECUTABLE_GPU)
	$(EXECUTABLE_GPU)

run_julia:
	export JULIA_NUM_THREADS=16; \
	julia $(JULIA_SCRIPT)

clean:
	rm -f $(EXECUTABLE_SINGLE) $(EXECUTABLE_MULTI) $(EXECUTABLE_GPU)

.PHONY: C_CPU C_Multi Julia_Multi CUDA

C_CPU: run_single
C_Multi: run_multi
Julia_Multi: run_julia
CUDA: run_gpu