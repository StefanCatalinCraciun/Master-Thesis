#####################################################################################
##################### Saxpy Julia For Loop CPU ######################################
#####################################################################################

# Set the number of elements in the arrays you desire
elements = 30   # No. of elements in array =2^elements
# @btime uses multiple iterations by default and gives back the median

#####################################################################################
#####################################################################################
#####################################################################################

using BenchmarkTools
using Statistics
using DelimitedFiles
using Printf

function saxpy(a,x,y)
    for i in 1:length(x)
        y[i] = a * x[i] + y[i] 
    end
end


N = 2 .^ (1:elements)
const a = Float32(3.1415)
global x = y = Array{Float32}(undef, )

# Preallocate array to store benchmark times
times = zeros(Float64, length(N))

global counter = 1

for size in N
    
    global x = 10*rand(Float32, size)
    global y = 10*rand(Float32, size)
        
    #@btime saxpy(a,x,y) #just prints the output

    time = @benchmark saxpy(a,x,y)
    times[counter] = mean(time.times)
    #println(mean(time.times))

    @show counter
    global counter += 1
end

# Save the results to a CSV file
writedlm("Saxpy/Benchmark_Results/Saxpy_Julia_For_Loop_CPU.csv",  times, ',')

for t in times
    @printf("%.3f\n", t)
end

