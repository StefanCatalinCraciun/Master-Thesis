#####################################################################################
##################### Saxpy Python Numba CuPy GPU ###################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 28   # No. of elements in array =2^elements
iterations = 30

#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
import cupy as cp
import numba.cuda as cuda
import timeit
np.set_printoptions(suppress = True) # deactivates scientific number format

### For shared memory
#pool = cp.cuda.MemoryPool(cp.cuda.malloc_managed)
#cp.cuda.set_allocator(pool.malloc)


@cuda.jit
def saxpy_kernel(a, x, y):
    i = cuda.grid(1)
    if i < x.size:
        y[i] = a * x[i] + y[i]


N = 2** np.arange(1,elements+1,1)
a = np.float32(3.1415)
Time=np.empty([iterations,elements])

for it in range(iterations):
    counter = 0
    for size in N:

        #x = 10 * np.random.rand(size).astype(np.float32)
        #y = 10 * np.random.rand(size).astype(np.float32) 
        x = 10 * cp.random.rand(size).astype(cp.float32)
        y = 10 * cp.random.rand(size).astype(cp.float32) 
        
        # Set up the grid and block dimensions
        threads_per_block = 1024
        blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block

        start = timeit.default_timer()
        
        saxpy_kernel[blocks_per_grid, threads_per_block](a, x, y)
        cp.cuda.Device().synchronize()

        end = timeit.default_timer()

        Time[it, counter] = (end - start)*1000

        #print(counter)
        counter = counter + 1
    print(it)
print("Done loop")

        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("Saxpy/Benchmark_Results/Saxpy_Python_Numba_Cupy_GPU.csv", Time_average, delimiter=",")