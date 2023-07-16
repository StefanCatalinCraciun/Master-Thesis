#####################################################################################
##################### Saxpy Julia GPU M1 Kernel #####################################
#####################################################################################

# Set the number of elements in the arrays and how many interations you desire
elements = 29 # No. of elements in array =2^elements


#####################################################################################
#####################################################################################
#####################################################################################

using BenchmarkTools
using Statistics
using DelimitedFiles
using Printf
using Metal



Metal.versioninfo()

function saxpy_kernel(a,x,y)
    i = thread_position_in_grid_1d()
    if i <= length(x)
      @inbounds y[i] = a * x[i] + y[i]
    end
    return nothing
  end
  

#N = 536_870_912
N = 2 .^ (1:elements)
const a = Float32(3.1415)
global x = y =  MtlArray{Float32}(undef, )

# Preallocate array to store benchmark times
times = zeros(Float64, length(N))

global counter = 1

for size in N
    
    global x = MtlArray(rand(Float32, size))
    global y = MtlArray(rand(Float32, size))

    global threads = 1024
    global groups = cld(size, threads)
    #groups = div(size + threads - 1, threads)

    #@metal threads = threads groups = groups saxpy_kernel(a,x,y)
    time = @benchmark @metal threads=threads groups=groups saxpy_kernel(a,x,y)    
        
    times[counter] = mean(time.times)
    #println(mean(time.times))

    @show counter
    global counter += 1
end

# Save the results to a CSV file
writedlm("Saxpy/Benchmark_Results/Saxpy_Julia_GPU_M1_Kernel.csv",  times, ',')

for t in times
    @printf("%.3f\n", t)
end

x = nothing
y = nothing
GC.gc(true)

