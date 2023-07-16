"""
Source object

This script defines a class for creating a source object with a given source and spatial coordinates.
It also includes a method for generating a Ricker wavelet.

@author: Stefan Catalin Craciun
"""

using LinearAlgebra

struct Src
    Src::Array{Float64, 1}
    Sx::Array{Float64, 1}
    Sy::Array{Float64, 1}
    Ns::Int
end

function Src(source::Array{Float64, 1}, sx::Array{Float64, 1}, sy::Array{Float64, 1})
    Ns = length(sx)
    return Src(source, sx, sy, Ns)
end

function SrcRicker(t0::Float64, f0::Float64, nt::Int, dt::Float64)
    t = (0:nt - 1) * dt .- t0     # Time axis values
    w0 = 2.0 * pi * f0           # Omega (angular frequency)
    arg = w0 .* t
    src = (1.0 .- 0.5 .* arg .^ 2) .* exp.(-0.25 .* arg .^ 2) # Ricker wavelet formula
    return src
end
