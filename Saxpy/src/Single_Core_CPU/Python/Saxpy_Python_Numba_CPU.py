#####################################################################################
##################### Saxpy Python Numba CPU ########################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire

elements = 30    # No. of elements in array = 2^elements
iterations = 20  # No. of iterations

#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
import time
from numba import jit
import numba
np.set_printoptions(suppress = True) # deactivates scientific number format

@jit(nopython=True)
def saxpy_numba(a, x, y):
    size = x.shape[0]
    for i in numba.prange(size):
        y[i] = a * x[i] + y[i]
    return y

N = 2** np.arange(1,elements+1,1)
a = np.float32(3.1415)
Time=np.empty([iterations,elements])

for it in range(iterations):
    counter = 0
    for size in N:
        x = 10*np.random.rand(size).astype(np.float32)
        y = 10*np.random.rand(size).astype(np.float32)

        start = time.time()
        
        y = saxpy_numba(a, x, y)

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        #print(counter)
        counter = counter + 1
    print(it)
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("Saxpy/Benchmark_Results/Saxpy_Python_Numba_CPU.csv", Time_average, delimiter=",")