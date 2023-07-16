"""
Receiver object

This script defines a class for creating a receiver object that allows to records seismograms, the presure wavefield
and 1 Trace (wich is also compared to an analytical solution) and saves the results as binary files.

@author: Stefan Catalin Craciun
"""

# Import Libraries
# ----------------
using PyPlot
using LinearAlgebra
using DSP: conv

# Import Classes
# ----------------
include("Model.jl")
include("Src.jl")

# ----------------
# Receiver Class
# ----------------
mutable struct Rec
    nr::Int
    rx::Array{Int, 1}
    ry::Array{Int, 1}
    nt::Int
    p::Array{Float32, 2}
    resamp::Int
    sresamp::Int
    pit::Int
    counter::Int
    gx::Int
    gy::Int
    nx::Int
    ny::Int
    dx::Float32
    Nb::Int
    dt::Float32
    wavefield::Array{Float32, 3}
    trace::Array{Float32, 1}
    trace_analytical::Array{Float32, 1}
end

function Rec(rx::Array{Int, 1}, ry::Array{Int, 1}, nt::Int, resamp::Int, sresamp::Int, nx::Int, ny::Int, dx::Float32, Nb::Int, dt::Float32, gx::Int=0, gy::Int=0)
    nr = length(rx)
    p = zeros(nr, Int(nt / resamp) + 1)
    wavefield = zeros(nx, ny, Int(nt / sresamp) + 1)
    trace = zeros(nt)
    trace_analytical = zeros(nt)
    Rec(nr, rx, ry, nt, p, resamp, sresamp, 1, 1, gx, gy, nx, ny, dx, Nb, dt, wavefield, trace, trace_analytical)
end

function rec_trace!(rec::Rec, it::Int, snp::Array{Float32, 2})
    rec.trace[it] = snp[rec.gx, rec.gy]
    return "OK"
end


function rec_trace_analytical!(rec::Rec, src::Src, model::Model)
    # Analytical solution
    time = collect(range(0, rec.nt * rec.dt, length=rec.nt)) # Time vector
    G = time .* 0 # Green's function

    # Calculate source-receiver distance
    r = sqrt((rec.dx * rec.gx - rec.dx * src.Sx[1])^2 + (rec.dx * rec.gy - rec.dx * src.Sy[1])^2)
    
    for a in 1:rec.nt # Calculate Green's function (Heaviside function)
        if (time[a] - r / sqrt(model.Kappa[1, 1] / model.Rho[1, 1])) >= 0
            G[a] = 1.0 / (2 * pi  * model.Rho[1, 1] * sqrt(model.Kappa[1, 1] / model.Rho[1, 1])^2) * (1.0 / sqrt(time[a]^2 - (r / sqrt(model.Kappa[1, 1] / model.Rho[1, 1]))^2))
        end
    end
   
    Gc = conv(G, src.Src .* rec.dt) # Convolve Green's function with source time function
    Gc = Gc[1:rec.nt]
    # Calculate analytical solution
    for a in 2:rec.nt - 1
        rec.trace_analytical[a] = (Gc[a + 1] - Gc[a - 1]) / (2.0 * rec.dt)
    end

    return "OK"
end

function rec_seismogram!(rec::Rec, it::Int, snp::Array{Float32, 2})
    if rec.pit > rec.nt - 1
        return "ERR"
    end

    if it % rec.resamp == 0
        for pos in 1:rec.nr
            ixr = Int(rec.rx[pos])
            iyr = Int(rec.ry[pos])
            rec.p[pos, rec.pit] = snp[iyr, ixr]
        end
        rec.pit += 1
    end

    return "OK"
end

function rec_wavefield!(rec::Rec, it::Int, snp::Array{Float32, 2})
    if rec.sresamp <= 0
        return "OK"
    end

    if it % rec.sresamp == 0
        rec.wavefield[:, :, rec.counter] = snp
        rec.counter += 1
    end

    return "OK"
end

function trace_save!(rec::Rec)
    output_file = "Visualization/trace.bin"
    
    metadata = [length(rec.trace), rec.dt]
    
    open(output_file, "w") do f
        write(f, metadata)
        write(f, rec.trace)
        println("Trace data saved to output_file")
    end
    return "OK"
end

function trace_analytical_save!(rec::Rec)
    output_file = "Visualization/trace_analytical.bin"
    
    metadata = [length(rec.trace_analytical), rec.dt]
    
    open(output_file, "w") do f
        write(f, metadata)
        write(f, rec.trace_analytical)
        println("Analytical solution data saved to output_file")
    end
    return "OK"
end

function seismogram_save!(rec::Rec)
    output_file = "Visualization/seismogram.bin"
    
    metadata = [size(rec.p)..., rec.dt, rec.dx, rec.Nb]
    
    open(output_file, "w") do f
        write(f, metadata)
        write(f, rec.p)
        println("Seismogram data saved to output_file")
    end
    return "OK"
end

function wavefield_save!(rec::Rec)
    output_file = "Visualization/wavefield.bin"
    
    metadata = [size(rec.wavefield)..., rec.dx, rec.Nb]
    println(metadata)
    
    open(output_file, "w") do f
        write(f, metadata)
        write(f, rec.wavefield)
        println("Wavefield data saved to output_file")
    end
    return "OK"
end
