"""
AC2D object

This script defines a AC2D mutable struct for solving the acoustic wave equation in 2D.

@author: Stefan Catalin Craciun
"""


# Import Classes
# ----------------
include("Model.jl")
include("Src.jl")
include("Rec.jl")
include("Differentiator.jl")
# ----------------
# AC2D Type
# ----------------
mutable struct Ac2d
    p::Array{Float64, 2}
    vx::Array{Float64, 2}
    vy::Array{Float64, 2}
    exx::Array{Float64, 2}
    eyy::Array{Float64, 2}
    gammax::Array{Float64, 2}
    gammay::Array{Float64, 2}
    thetax::Array{Float64, 2}
    thetay::Array{Float64, 2}
    ts::Int64
end

function Ac2d(model::Model)
    p = zeros(model.Nx, model.Ny)
    vx = zeros(model.Nx, model.Ny)
    vy = zeros(model.Nx, model.Ny)
    exx = zeros(model.Nx, model.Ny)
    eyy = zeros(model.Nx, model.Ny)
    gammax = zeros(model.Nx, model.Ny)
    gammay = zeros(model.Nx, model.Ny)
    thetax = zeros(model.Nx, model.Ny)
    thetay = zeros(model.Nx, model.Ny)
    ts = 1
    return Ac2d(p, vx, vy, exx, eyy, gammax, gammay, thetax, thetay, ts)
end

function Ac2dSolve(ac2d::Ac2d, model::Model, src::Src, rec::Rec, nt::Int64, l::Int64)
    @fastmath begin
        diff = Differentiator(l)
        oldperc = 0.0
        ns = ac2d.ts
        ne = ns + nt

        @inbounds for i in ns:ne-1
            DiffDxplus(diff, ac2d.p, ac2d.exx, model.Dx)
            Ac2dvx(ac2d, model)
            DiffDyplus(diff, ac2d.p, ac2d.eyy, model.Dx)
            Ac2dvy(ac2d, model)

            DiffDxminus(diff, ac2d.vx, ac2d.exx, model.Dx)
            DiffDyminus(diff, ac2d.vy, ac2d.eyy, model.Dx)

            Ac2dstress(ac2d, model)

            @inbounds for k in 1:src.Ns
                sx = round(Int, src.Sx[k])
                sy = round(Int, src.Sy[k])
                ac2d.p[sx, sy] += model.Dt * (src.Src[i] / (model.Dx * model.Dx * model.Rho[sx, sy] )) 
            end

            perc = 1000.0 * (i / (ne - ns - 1))
            if perc - oldperc >= 10.0
                iperc = round(Int, perc) ÷ 10
                if iperc % 10 == 0
                    println(iperc)
                end
                oldperc = perc
            end

        

            rec_wavefield!(rec, i, ac2d.p)
            rec_seismogram!(rec, i, ac2d.p)
            rec_trace!(rec, i, ac2d.p)
        end
    end

    return "OK"
end


using LoopVectorization
using LinearAlgebra


function Ac2dvx(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    @fastmath begin
        @tturbo for i in 1:nx
            for j in 1:ny
                ac2d.vx[i, j] = model.Dt * (1.0 / model.Rho[i, j]) * ac2d.exx[i, j] + ac2d.vx[i, j] + model.Dt * ac2d.thetax[i, j] * model.Drhox[i, j]
                ac2d.thetax[i, j] = model.Eta1x[i, j] * ac2d.thetax[i, j] + model.Eta2x[i, j] * ac2d.exx[i, j]
            end
        end
    end
end

function Ac2dvy(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    @fastmath begin
        @tturbo for i in 1:nx
            for j in 1:ny
                ac2d.vy[i, j] = model.Dt * (1.0 / model.Rho[i, j]) * ac2d.eyy[i, j] + ac2d.vy[i, j] + model.Dt * ac2d.thetay[i, j] * model.Drhoy[i, j]
                ac2d.thetay[i, j] = model.Eta1y[i, j] * ac2d.thetay[i, j] + model.Eta2y[i, j] * ac2d.eyy[i, j]
            end
        end
    end
end

function Ac2dstress(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    @fastmath begin
        @tturbo for i in 1:nx
            for j in 1:ny
                ac2d.p[i, j] = model.Dt * model.Kappa[i, j] * (ac2d.exx[i, j] + ac2d.eyy[i, j]) + ac2d.p[i, j] + model.Dt * (ac2d.gammax[i, j] * model.Dkappax[i, j] + ac2d.gammay[i, j] * model.Dkappay[i, j])
                ac2d.gammax[i, j] = model.Alpha1x[i, j] * ac2d.gammax[i, j] + model.Alpha2x[i, j] * ac2d.exx[i, j]
                ac2d.gammay[i, j] = model.Alpha1y[i, j] * ac2d.gammay[i, j] + model.Alpha2y[i, j] * ac2d.eyy[i, j]
            end
        end
    end
end
