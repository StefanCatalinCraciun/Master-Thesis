#####################################################################################
##################### Saxpy Python For Loop CPU #####################################
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

        for i in range(size):
            y[i] = a * x[i] + y[i]

        end = time.time()
        
        Time[it,counter] = (end - start)*1000
        #print(counter)
        counter = counter + 1
    print(it)
print("Done loop")
        
Time_average = np.mean(Time, axis=0)
print(Time_average)

# Saving the array
np.savetxt("Saxpy/Benchmark_Results/Saxpy_Python_For_Loop_CPU.csv", Time_average, delimiter=",")
