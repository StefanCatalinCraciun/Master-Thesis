"""
Source object

This script defines a class for creating a source object with a given source and spatial coordinates.
It also includes a method for generating a Ricker wavelet.

@author: Stefan Catalin Craciun
"""

using LinearAlgebra

struct Src
    Src::Array{Float32, 1}
    Sx::Array{Float32, 1}
    Sy::Array{Float32, 1}
    Ns::Int
end

function Src(source::Array{Float32, 1}, sx::Array{Float32, 1}, sy::Array{Float32, 1})
    Ns = length(sx)
    return Src(source, sx, sy, Ns)
end

function SrcRicker(t0::Float32, f0::Float32, nt::Int, dt::Float32)
    t = (0:nt - 1) * dt .- t0     # Time axis values
    w0 = 2.0 * pi * f0           # Omega (angular frequency)
    arg = w0 .* t
    src = (1.0 .- 0.5 .* arg .^ 2) .* exp.(-0.25 .* arg .^ 2) # Ricker wavelet formula
    return src
end
