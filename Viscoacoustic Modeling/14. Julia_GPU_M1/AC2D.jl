"""
AC2D object

This script defines a AC2D mutable struct for solving the acoustic wave equation in 2D.

@author: Stefan Catalin Craciun
"""

using Metal
using Unrolled
#Metal.versioninfo()


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
    p::Array{Float32, 2}
    vx::Array{Float32, 2}
    vy::Array{Float32, 2}
    exx::Array{Float32, 2}
    eyy::Array{Float32, 2}
    gammax::Array{Float32, 2}
    gammay::Array{Float32, 2}
    thetax::Array{Float32, 2}
    thetay::Array{Float32, 2}
    ts::Int32
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



function Ac2dSolve(ac2d::Ac2d, model::Model, src::Src, rec::Rec, nt::Int32, l::Int32)
    @fastmath begin
        diff = Differentiator(l)
        oldperc = 0.0
        ns = ac2d.ts
        ne = ns + nt
        sx = round(Int, src.Sx[1])
        sy = round(Int, src.Sy[1])
        
        # Define the Metal arrays
        p_gpu = MtlArray(ac2d.p)
        vx_gpu = MtlArray(ac2d.vx)
        vy_gpu = MtlArray(ac2d.vy)
        exx_gpu = MtlArray(ac2d.exx)
        eyy_gpu = MtlArray(ac2d.eyy)
        gammax_gpu = MtlArray(ac2d.gammax)
        gammay_gpu = MtlArray(ac2d.gammay)
        thetax_gpu = MtlArray(ac2d.thetax)
        thetay_gpu = MtlArray(ac2d.thetay)
        rho_gpu = MtlArray(model.Rho)
        kappa_gpu = MtlArray(model.Kappa)
        drhox_gpu = MtlArray(model.Drhox)
        drhoy_gpu = MtlArray(model.Drhoy)
        dkappax_gpu = MtlArray(model.Dkappax)
        dkappay_gpu = MtlArray(model.Dkappay)
        eta1x_gpu = MtlArray(model.Eta1x)
        eta1y_gpu = MtlArray(model.Eta1y)
        eta2x_gpu = MtlArray(model.Eta2x)
        eta2y_gpu = MtlArray(model.Eta2y)
        alpha1x_gpu = MtlArray(model.Alpha1x)
        alpha1y_gpu = MtlArray(model.Alpha1y)
        alpha2x_gpu = MtlArray(model.Alpha2x)
        alpha2y_gpu = MtlArray(model.Alpha2y)
        src_gpu = MtlArray(src.Src)
        w_gpu = MtlArray(diff.w)
        
        
        
        # Determine the block and grid size
        #threads = 512
        #groups = cld(model.Nx*model.Ny, threads)
        threads = (8, 8)
        groups = (div(model.Nx + threads[1] - 1, threads[1]), div(model.Ny + threads[2] - 1, threads[2]))

        println("START________________________________")
        # Perform necessary operations on the GPU
        @inbounds for i in ns:ne-1
            if i == 2
                global t1 = time()
            end

            #DiffDxplus(diff, ac2d.p, ac2d.exx, model.Dx)
            #Ac2dvx(ac2d, model)

            #DiffDyplus(diff, ac2d.p, ac2d.eyy, model.Dx)
            #Ac2dvy(ac2d, model)

            #DiffDxminus(diff, ac2d.vx, ac2d.exx, model.Dx)
            #DiffDyminus(diff, ac2d.vy, ac2d.eyy, model.Dx)
            @metal threads = threads groups = groups DiffDxplus_kernel!(exx_gpu, p_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
            @metal threads = threads groups = groups Ac2dvx_kernel!(vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, model.Dt, model.Nx, model.Ny)
            
            @metal threads = threads groups = groups DiffDyplus_kernel!(eyy_gpu, p_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
            @metal threads = threads groups = groups Ac2dvy_kernel!(vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, model.Dt, model.Nx, model.Ny)

            @metal threads = threads groups = groups DiffDxminus_kernel!(exx_gpu, vx_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
            @metal threads = threads groups = groups DiffDyminus_kernel!(eyy_gpu, vy_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)

            @metal threads = threads groups = groups Ac2dstress_kernel!(p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, model.Dt, model.Nx, model.Ny)
            @metal update_pressure_kernel!(p_gpu, rho_gpu, src.Src[i], model.Dt, model.Dx, sx, sy)
            
            
            perc = 1000.0 * (i / (ne - ns - 1))
            if perc - oldperc >= 10.0
                iperc = round(Int, perc) ÷ 10
                if iperc % 10 == 0
                    println(iperc)
                end
                oldperc = perc
            end
            
            
            ac2d.p = Array(p_gpu)
            rec_wavefield!(rec, i, ac2d.p)
            rec_seismogram!(rec, i, ac2d.p)
            rec_trace!(rec, i, ac2d.p)
            

        end
        t2 = time()
        println("Solver wall clock time: ", t2 - t1)

        p_gpu = nothing
        vx_gpu = nothing
        vy_gpu = nothing
        exx_gpu = nothing
        eyy_gpu = nothing
        thetax_gpu = nothing
        thetay_gpu = nothing
        rho_gpu = nothing
        drhox_gpu = nothing
        drhoy_gpu = nothing
        gammax_gpu = nothing
        gammay_gpu = nothing
        kappa_gpu = nothing
        dkappax_gpu = nothing
        dkappay_gpu = nothing
        eta1x_gpu = nothing
        eta1y_gpu = nothing
        eta2x_gpu = nothing
        eta2y_gpu = nothing
        alpha1x_gpu = nothing
        alpha1y_gpu = nothing
        alpha2x_gpu = nothing
        alpha2y_gpu = nothing
        src_gpu = nothing
        w_gpu = nothing
    end
        
    return "OK"
end

####### metal kernels #######

function Ac2dvx_kernel!(vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, dt, nx, ny)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y
        arg = Float32(1.0)

        if x <= nx && y <= ny
            @inbounds vx_gpu[x, y] = dt * (arg / rho_gpu[x, y]) * exx_gpu[x, y] + vx_gpu[x, y] + dt * thetax_gpu[x, y] * drhox_gpu[x, y]
            @inbounds thetax_gpu[x, y] = eta1x_gpu[x, y] * thetax_gpu[x, y] + eta2x_gpu[x, y] * exx_gpu[x, y]
        end
    end
    return
end

function Ac2dvy_kernel!(vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, dt, nx, ny)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y
        arg = Float32(1.0)

        if x <= nx && y <= ny
            @inbounds vy_gpu[x, y] = dt * (arg / rho_gpu[x, y]) * eyy_gpu[x, y] + vy_gpu[x, y] + dt * thetay_gpu[x, y] * drhoy_gpu[x, y]
            @inbounds thetay_gpu[x, y] = eta1y_gpu[x, y] * thetay_gpu[x, y] + eta2y_gpu[x, y] * eyy_gpu[x, y]
        end
    end
    return
end

function Ac2dstress_kernel!(p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, dt, nx, ny)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y

        if x <= nx && y <= ny
            @inbounds p_gpu[x, y] = dt * kappa_gpu[x, y] * (exx_gpu[x, y] + eyy_gpu[x, y]) + p_gpu[x, y] + dt * (gammax_gpu[x, y] * dkappax_gpu[x, y] + gammay_gpu[x, y] * dkappay_gpu[x, y])
            @inbounds gammax_gpu[x, y] = alpha1x_gpu[x, y] * gammax_gpu[x, y] + alpha2x_gpu[x, y] * exx_gpu[x, y]
            @inbounds gammay_gpu[x, y] = alpha1y_gpu[x, y] * gammay_gpu[x, y] + alpha2y_gpu[x, y] * eyy_gpu[x, y]
        end
    end
    return
end


function DiffDxplus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y

        if x <= nx && y <= ny
            if x <= l
                sum = Float32(0.0)
                for k in 1:x
                    @inbounds sum = sum - w_gpu[k] * A_gpu[x - (k - 1), y]
                end
                for k in 1:l
                    @inbounds sum = sum + w_gpu[k] * A_gpu[x + k, y]
                end
                @inbounds dA_gpu[x, y] = sum / dx
            elseif x >= l+1 && x <= nx-l
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum = sum + w_gpu[k] * (-A_gpu[x - (k - 1), y] + A_gpu[x + k, y])
                end
                @inbounds dA_gpu[x, y] = sum / dx
            elseif x >= nx-l+1 && x <= nx
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum = sum - w_gpu[k] * A_gpu[x - (k - 1), y]
                end
                for k in 1:(nx - x)
                    @inbounds sum = sum + w_gpu[k] * A_gpu[x + k, y]
                end
                @inbounds dA_gpu[x, y] = sum / dx
            end
        end
    end
    return
end

function DiffDyplus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y

        if x <= nx
            if y >= 1 && y <= l
                sum = Float32(0.0)
                for k in 1:y
                    @inbounds sum -= w_gpu[k] * A_gpu[x, y - (k - 1)]
                end
                for k in 1:l
                    @inbounds sum += w_gpu[k] * A_gpu[x, y + k]
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif y >= l + 1 && y <= ny - l
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum += w_gpu[k] * (-A_gpu[x, y - (k - 1)] + A_gpu[x, y + k])
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif y >= ny - l + 1 && y <= ny
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum -= w_gpu[k] * A_gpu[x, y - (k - 1)]
                end
                for k in 1:(ny - y)
                    @inbounds sum += w_gpu[k] * A_gpu[x, y + k]
                end
                @inbounds dA_gpu[x, y] = sum / dx
            end
        end
    end

    return
end

function DiffDxminus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y

        if y <= ny
            if x >= 1 && x <= l
                sum = Float32(0.0)
                for k in 1:x
                    if x - k >= 1
                        @inbounds sum -= w_gpu[k] * A_gpu[x - k, y]
                    end
                end
                for k in 1:l
                    @inbounds sum += w_gpu[k] * A_gpu[x + (k - 1), y]
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif x >= l + 1 && x <= nx - l
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum += w_gpu[k] * (-A_gpu[x - k, y] + A_gpu[x + (k - 1), y])
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif x >= nx - l + 1 && x <= nx
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum -= w_gpu[k] * A_gpu[x - k, y]
                end
                for k in 1:(nx - x)
                    @inbounds sum += w_gpu[k] * A_gpu[x + (k - 1), y]
                end
                @inbounds dA_gpu[x, y] = sum / dx
            end
        end
    end
    return
end

function DiffDyminus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    @fastmath begin
        x = thread_position_in_grid_2d().x
        y = thread_position_in_grid_2d().y

        if x <= nx
            if y >= 1 && y <= l
                sum = Float32(0.0)
                for k in 1:y
                    if y - k >= 1
                        @inbounds sum -= w_gpu[k] * A_gpu[x, y - k]
                    end
                end
                for k in 1:l
                    @inbounds sum += w_gpu[k] * A_gpu[x, y + (k - 1)]
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif y >= l + 1 && y <= ny - l
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum += w_gpu[k] * (-A_gpu[x, y - k] + A_gpu[x, y + (k - 1)])
                end
                @inbounds dA_gpu[x, y] = sum / dx

            elseif y >= ny - l + 1 && y <= ny
                sum = Float32(0.0)
                for k in 1:l
                    @inbounds sum -= w_gpu[k] * A_gpu[x, y - k]
                end
                for k in 1:(ny - y)
                    @inbounds sum += w_gpu[k] * A_gpu[x, y + (k - 1)]
                end
                @inbounds dA_gpu[x, y] = sum / dx
            end
        end
    end

    return
end

function update_pressure_kernel!(p_gpu, rho_gpu, src_val, Dt, Dx, sx, sy)
    p_gpu[sx, sy] += Dt * (src_val / (Dx * Dx * rho_gpu[sx, sy]))
    return nothing
end




function Ac2dvx(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    for i in 1:nx
        for j in 1:ny
            ac2d.vx[i, j] = model.Dt * (1.0 / model.Rho[i, j]) * ac2d.exx[i, j] + ac2d.vx[i, j] + model.Dt * ac2d.thetax[i, j] * model.Drhox[i, j]
            ac2d.thetax[i, j] = model.Eta1x[i, j] * ac2d.thetax[i, j] + model.Eta2x[i, j] * ac2d.exx[i, j]
        end
    end
end


function Ac2dvy(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    for i in 1:nx
        for j in 1:ny
            ac2d.vy[i, j] = model.Dt * (1.0 / model.Rho[i, j]) * ac2d.eyy[i, j] + ac2d.vy[i, j] + model.Dt * ac2d.thetay[i, j] * model.Drhoy[i, j]
            ac2d.thetay[i, j] = model.Eta1y[i, j] * ac2d.thetay[i, j] + model.Eta2y[i, j] * ac2d.eyy[i, j]
        end
    end
end

function Ac2dstress(ac2d::Ac2d, model::Model)
    nx = model.Nx
    ny = model.Ny

    for i in 1:nx
        for j in 1:ny
            ac2d.p[i, j] = model.Dt * model.Kappa[i, j] * (ac2d.exx[i, j] + ac2d.eyy[i, j]) + ac2d.p[i, j] + model.Dt * (ac2d.gammax[i, j] * model.Dkappax[i, j] + ac2d.gammay[i, j] * model.Dkappay[i, j])
            ac2d.gammax[i, j] = model.Alpha1x[i, j] * ac2d.gammax[i, j] + model.Alpha2x[i, j] * ac2d.exx[i, j]
            ac2d.gammay[i, j] = model.Alpha1y[i, j] * ac2d.gammay[i, j] + model.Alpha2y[i, j] * ac2d.eyy[i, j]
        end
    end
end