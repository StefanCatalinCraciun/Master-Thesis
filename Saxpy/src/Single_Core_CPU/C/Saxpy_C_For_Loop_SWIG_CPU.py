#####################################################################################
##################### Saxpy SWIG CPU ################################################
#####################################################################################
###swig -python wrapper.i && python setup.py build_ext --inplace

# Set the number of elements in the arrays and how many interations you desire

elements = 30    # No. of elements in array = 2^elements
iterations = 20  # No. of iterations

#####################################################################################
#####################################################################################
#####################################################################################

#Import modules
import numpy as np
import time
import sys
sys.path.insert(1, 'Saxpy/src/Single_Core_CPU/C/SWIG')
import wrapper
np.set_printoptions(suppress = True) # deactivates scientific number format

N = 2** np.arange(1,elements+1,1)
a = 3.1415
Time=np.empty([iterations,elements])

for it in range(iterations):
    counter = 0
    for size in N:
        x = 10*np.random.rand(size).astype(np.float32)
        y = 10*np.random.rand(size).astype(np.float32)

        start = time.time()

        wrapper.saxpy_serial(x,y,a)

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        #print(counter)
        counter = counter + 1
    print(it)
print("Done loop")
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("Saxpy/Benchmark_Results/Saxpy_C_For_Loop_SWIG_CPU.csv", Time_average, delimiter=",")