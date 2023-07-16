#####################################################################################
##################### Saxpy Julia Kernel GPU #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 29   # No. of elements in array =2^elements


#####################################################################################
#####################################################################################
#####################################################################################

using BenchmarkTools
using Statistics
using DelimitedFiles
using Printf
using CUDA

function saxpy_gpu_kernel!(a,x,y)
    # calculate index number
    i = (blockIdx().x - 1)* blockDim().x + threadIdx().x
    # calculate saxpy
    if i <= length(y)
        # eliminate array bounds check
        @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
end

#N = 536_870_912
N = 2 .^ (1:elements)
const a = Float32(3.1415)
global x = y = CUDA.Array{Float32}(undef, )

# Preallocate array to store benchmark times
times = zeros(Float64, length(N))

global counter = 1

for size in N
    global nthreads = CUDA.attribute(
    device(),
    CUDA.DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK
    )   
    global nblocks = cld(size, nthreads)
    
    global x = CUDA.rand(Float32, size)
    global y = CUDA.rand(Float32, size)
        
    time = @benchmark CUDA.@sync @cuda(
        threads = nthreads,
        blocks = nblocks,
        saxpy_gpu_kernel!(a,x,y)
    )
    
    
    times[counter] = mean(time.times)
    #println(mean(time.times))

    @show counter
    global counter += 1
end

# Save the results to a CSV file
writedlm("Saxpy/Benchmark_Results/Saxpy_Julia_Kernel_GPU.csv",  times, ',')

for t in times
    @printf("%.3f\n", t)
end

x = nothing
y = nothing
GC.gc(true)
CUDA.reclaim()
CUDA.memory_status()

println(CUDA.memory_status())