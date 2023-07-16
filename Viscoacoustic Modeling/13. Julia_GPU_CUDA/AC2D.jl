"""
AC2D object

This script defines a AC2D mutable struct for solving the acoustic wave equation in 2D.

@author: Stefan Catalin Craciun
"""

using CUDA

#CUDA.versioninfo()


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
    diff = Differentiator(l)
    oldperc = 0.0
    ns = ac2d.ts
    ne = ns + nt
    sx = round(Int, src.Sx[1])
    sy = round(Int, src.Sy[1])

    # Define the CUDA arrays
    p_gpu = CuArray(ac2d.p)
    vx_gpu = CuArray(ac2d.vx)
    vy_gpu = CuArray(ac2d.vy)
    exx_gpu = CuArray(ac2d.exx)
    eyy_gpu = CuArray(ac2d.eyy)
    gammax_gpu = CuArray(ac2d.gammax)
    gammay_gpu = CuArray(ac2d.gammay)
    thetax_gpu = CuArray(ac2d.thetax)
    thetay_gpu = CuArray(ac2d.thetay)
    rho_gpu = CuArray(model.Rho)
    kappa_gpu = CuArray(model.Kappa)
    drhox_gpu = CuArray(model.Drhox)
    drhoy_gpu = CuArray(model.Drhoy)
    dkappax_gpu = CuArray(model.Dkappax)
    dkappay_gpu = CuArray(model.Dkappay)
    eta1x_gpu = CuArray(model.Eta1x)
    eta1y_gpu = CuArray(model.Eta1y)
    eta2x_gpu = CuArray(model.Eta2x)
    eta2y_gpu = CuArray(model.Eta2y)
    alpha1x_gpu = CuArray(model.Alpha1x)
    alpha1y_gpu = CuArray(model.Alpha1y)
    alpha2x_gpu = CuArray(model.Alpha2x)
    alpha2y_gpu = CuArray(model.Alpha2y)
    src_gpu = CuArray(src.Src)
    w_gpu = CuArray(diff.w)
    
    
    # Determine the block and grid size
    blockDim = (16, 16)
    gridDim = (div(model.Nx + blockDim[1] - 1, blockDim[1]), div(model.Ny + blockDim[2] - 1, blockDim[2]))

    println("START________________________________")
    # Perform necessary operations on the GPU
    for i in ns:ne-1
        #DiffDxplus(diff, ac2d.p, ac2d.exx, model.Dx)
        #DiffDxplus_test(ac2d.p, ac2d.exx, diff.w, model.Dx, model.Nx, model.Ny, l)
        if i == 2
            global t1 = time()
        end
        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        DiffDxplus_kernel!(exx_gpu, p_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
        )

        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        Ac2dvx_kernel!(vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, model.Dt, model.Nx, model.Ny)
        )

        #DiffDyplus(diff, ac2d.p, ac2d.eyy, model.Dx)
        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        DiffDyplus_kernel!(eyy_gpu, p_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
        )


        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        Ac2dvy_kernel!(vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, model.Dt, model.Nx, model.Ny)
        )

        #DiffDxminus(diff, ac2d.vx, ac2d.exx, model.Dx)
        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        DiffDxminus_kernel!(exx_gpu, vx_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
        )

        #DiffDyminus(diff, ac2d.vy, ac2d.eyy, model.Dx)
        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        DiffDyminus_kernel!(eyy_gpu, vy_gpu, w_gpu, model.Dx, model.Nx, model.Ny, l)
        )

        
        CUDA.@sync @cuda(
        threads = blockDim,
        blocks = gridDim,
        Ac2dstress_kernel!(p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, model.Dt, model.Nx, model.Ny)
        )

        #Ac2dstress(ac2d, model)
        
     
        CUDA.@sync @cuda update_pressure_kernel!(p_gpu, rho_gpu, src.Src[i], model.Dt, model.Dx, sx, sy)

        #ac2d.p[sx, sy] += model.Dt * (src.Src[i] / (model.Dx * model.Dx * model.Rho[sx, sy]))
        
        
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
    return "OK"
end

####### CUDA kernels #######

function Ac2dvx_kernel!(vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, dt, nx, ny)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx && y <= ny
        @inbounds vx_gpu[x, y] = dt * (1.0 / rho_gpu[x, y]) * exx_gpu[x, y] + vx_gpu[x, y] + dt * thetax_gpu[x, y] * drhox_gpu[x, y]
        @inbounds thetax_gpu[x, y] = eta1x_gpu[x, y] * thetax_gpu[x, y] + eta2x_gpu[x, y] * exx_gpu[x, y]
    end
    return
end

function Ac2dvy_kernel!(vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, dt, nx, ny)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx && y <= ny
        @inbounds vy_gpu[x, y] = dt * (1.0 / rho_gpu[x, y]) * eyy_gpu[x, y] + vy_gpu[x, y] + dt * thetay_gpu[x, y] * drhoy_gpu[x, y]
        @inbounds thetay_gpu[x, y] = eta1y_gpu[x, y] * thetay_gpu[x, y] + eta2y_gpu[x, y] * eyy_gpu[x, y]
    end
    return
end

function Ac2dstress_kernel!(p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, dt, nx, ny)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx && y <= ny
        @inbounds p_gpu[x, y] = dt * kappa_gpu[x, y] * (exx_gpu[x, y] + eyy_gpu[x, y]) + p_gpu[x, y] + dt * (gammax_gpu[x, y] * dkappax_gpu[x, y] + gammay_gpu[x, y] * dkappay_gpu[x, y])
        @inbounds gammax_gpu[x, y] = alpha1x_gpu[x, y] * gammax_gpu[x, y] + alpha2x_gpu[x, y] * exx_gpu[x, y]
        @inbounds gammay_gpu[x, y] = alpha1y_gpu[x, y] * gammay_gpu[x, y] + alpha2y_gpu[x, y] * eyy_gpu[x, y]
    end
    return
end


function DiffDxplus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx && y <= ny
        if x <= l
            sum = 0.0
            for k in 1:x
                sum = sum - w_gpu[k] * A_gpu[x - (k - 1), y]
            end
            for k in 1:l
                sum = sum + w_gpu[k] * A_gpu[x + k, y]
            end
            dA_gpu[x, y] = sum / dx
        elseif x >= l+1 && x <= nx-l
            sum = 0.0
            for k in 1:l
                sum = sum + w_gpu[k] * (-A_gpu[x - (k - 1), y] + A_gpu[x + k, y])
            end
            dA_gpu[x, y] = sum / dx
        else
            sum = 0.0
            for k in 1:l
                sum = sum - w_gpu[k] * A_gpu[x - (k - 1), y]
            end
            for k in 1:(nx - x)
                sum = sum + w_gpu[k] * A_gpu[x + k, y]
            end
            dA_gpu[x, y] = sum / dx
        end
    end
    return
end

function DiffDyplus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx
        if y >= 1 && y <= l
            sum = 0.0
            for k in 1:y
                sum -= w_gpu[k] * A_gpu[x, y - (k - 1)]
            end
            for k in 1:l
                sum += w_gpu[k] * A_gpu[x, y + k]
            end
            dA_gpu[x, y] = sum / dx

        elseif y >= l + 1 && y <= ny - l
            sum = 0.0
            for k in 1:l
                sum += w_gpu[k] * (-A_gpu[x, y - (k - 1)] + A_gpu[x, y + k])
            end
            dA_gpu[x, y] = sum / dx

        elseif y >= ny - l + 1 && y <= ny
            sum = 0.0
            for k in 1:l
                sum -= w_gpu[k] * A_gpu[x, y - (k - 1)]
            end
            for k in 1:(ny - y)
                sum += w_gpu[k] * A_gpu[x, y + k]
            end
            dA_gpu[x, y] = sum / dx
        end
    end

    return
end

function DiffDxminus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if y <= ny
        if x >= 1 && x <= l
            sum = 0.0
            for k in 1:x
                if x - k >= 1
                    sum -= w_gpu[k] * A_gpu[x - k, y]
                end
            end
            for k in 1:l
                sum += w_gpu[k] * A_gpu[x + (k - 1), y]
            end
            dA_gpu[x, y] = sum / dx

        elseif x >= l + 1 && x <= nx - l
            sum = 0.0
            for k in 1:l
                sum += w_gpu[k] * (-A_gpu[x - k, y] + A_gpu[x + (k - 1), y])
            end
            dA_gpu[x, y] = sum / dx

        elseif x >= nx - l + 1 && x <= nx
            sum = 0.0
            for k in 1:l
                sum -= w_gpu[k] * A_gpu[x - k, y]
            end
            for k in 1:(nx - x)
                sum += w_gpu[k] * A_gpu[x + (k - 1), y]
            end
            dA_gpu[x, y] = sum / dx
        end
    end

    return
end

function DiffDyminus_kernel!(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l)
    x = threadIdx().x + blockDim().x * (blockIdx().x - 1)
    y = threadIdx().y + blockDim().y * (blockIdx().y - 1)

    if x <= nx
        if y >= 1 && y <= l
            sum = 0.0
            for k in 1:y
                if y - k >= 1
                    sum -= w_gpu[k] * A_gpu[x, y - k]
                end
            end
            for k in 1:l
                sum += w_gpu[k] * A_gpu[x, y + (k - 1)]
            end
            dA_gpu[x, y] = sum / dx

        elseif y >= l + 1 && y <= ny - l
            sum = 0.0
            for k in 1:l
                sum += w_gpu[k] * (-A_gpu[x, y - k] + A_gpu[x, y + (k - 1)])
            end
            dA_gpu[x, y] = sum / dx

        elseif y >= ny - l + 1 && y <= ny
            sum = 0.0
            for k in 1:l
                sum -= w_gpu[k] * A_gpu[x, y - k]
            end
            for k in 1:(ny - y)
                sum += w_gpu[k] * A_gpu[x, y + (k - 1)]
            end
            dA_gpu[x, y] = sum / dx
        end
    end

    return
end

function update_pressure_kernel!(p_gpu::CuDeviceMatrix{Float64, 1}, rho_gpu::CuDeviceMatrix{Float64, 1}, src_val::Float64, Dt::Float64, Dx::Float64, sx::Int64, sy::Int64)
    p_gpu[sx, sy] += Dt * (src_val / (Dx * Dx * rho_gpu[sx, sy]))
    return nothing
end
