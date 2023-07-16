#####################################################################################
##################### Saxpy Python Cython Multithread ###############################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire

elements = 30    # No. of elements in array = 2^elements
iterations = 20  # No. of iterations
threads = 16      # No. of threads

#### python setup.py build_ext --inplace  ### to build the cython file
#####################################################################################
#####################################################################################
#####################################################################################

#%load_ext autoreload
#%autoreload 2
#Import modules
import sys
sys.path.insert(1, 'Saxpy/src/Multiple_Core_CPU/Python/Saxpy_Cython')
import numpy as np
import time
from saxpy_cython import*


np.set_printoptions(suppress = True) # deactivates scientific number format

N = 2** np.arange(1,elements+1,1)
a = np.float32(3.1415)
Time=np.empty([iterations,elements])

for it in range(iterations):
    counter = 0
    for size in N:
        x = 10*np.random.rand(size).astype(np.float32)
        y = 10*np.random.rand(size).astype(np.float32)

        start = time.time()

        y = saxpy_cython_multithread(a,x,y,threads)

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        #print(counter)
        counter = counter + 1
    print(it)
print("Done loop")

        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("Saxpy/Benchmark_Results/Saxpy_Python_Cython_Multithread_CPU16.csv", Time_average, delimiter=",")