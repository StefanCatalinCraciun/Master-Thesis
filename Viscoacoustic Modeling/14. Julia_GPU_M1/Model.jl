"""
Model object

This script defines a mutable struct for creating a model object that contains the velocity, density, and Q models and computes the viscoelastic parameters.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
using Printf

# ----------------
# Model Class
# ----------------
mutable struct Model
    """
    Initializes a mutable struct with default parameters and attributes set to None.
    
    """
    Nx::Union{Nothing, Int32}
    Ny::Union{Nothing, Int32}
    Nb::Union{Nothing, Int32}
    W0::Union{Nothing, Float32}
    Q::Union{Nothing, Array{Float32, 2}}
    Kappa::Union{Nothing, Array{Float32, 2}}
    Dkappax::Union{Nothing, Array{Float32, 2}}
    Dkappay::Union{Nothing, Array{Float32, 2}}
    Drhox::Union{Nothing, Array{Float32, 2}}
    Drhoy::Union{Nothing, Array{Float32, 2}}
    Rho::Union{Nothing, Array{Float32, 2}}
    Alpha1x::Union{Nothing, Array{Float32, 2}}
    Alpha1y::Union{Nothing, Array{Float32, 2}}
    Alpha2x::Union{Nothing, Array{Float32, 2}}
    Alpha2y::Union{Nothing, Array{Float32, 2}}
    Eta1x::Union{Nothing, Array{Float32, 2}}
    Eta1y::Union{Nothing, Array{Float32, 2}}
    Eta2x::Union{Nothing, Array{Float32, 2}}
    Eta2y::Union{Nothing, Array{Float32, 2}}
    dx::Union{Nothing, Array{Float32, 1}}
    dy::Union{Nothing, Array{Float32, 1}}
    Dx::Union{Nothing, Float32}
    Dt::Union{Nothing, Float32}

    function Model()
        new(nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing,
            nothing, nothing)
    end

end

function ModelNew(model::Model, vp, rho, Q, Dx, Dt, W0, Nb, Rheol)
    """
    ModelNew creates a new model.

    Parameters:
    - vp :  P-wave velocity model (2D numpy array)
    - rho:  Density (2D numpy array)
    - Q  :  Q-values (2D numpy array)
    - Dx :  Grid interval in x- and y-directions (float)
    - Dt :  Modeling time sampling interval (float)
    - W0 :  Q-model peak angular frequency (float)
    - Nb :  Width of border attenuation zone (in grid points) (int)
    - Rheol : Type of Q-model (str)

              Rheol = 'MAXWELL' (Maxwell solid)
              Rheol = 'SLS' (Standard linear solid)

    Returns:
    - Model structure

    Model creates the parameters needed by the Ac2d object
    to perform 2D acoustic modeling.
    """

    if Rheol == "MAXWELL"
        model = Model_maxwell(model, vp, rho, Q, Dx, Dt, W0, Nb)
    elseif Rheol == "SLS"
        model = Model_sls(model, vp, rho, Q, Dx, Dt, W0, Nb)
    else
        @error "Unknown Q-model"
        # Bailing out
        exit(1)
    end

    return model
end

function Model_maxwell(model::Model, vp, rho, Q, Dx, Dt, W0, Nb)
    
    model.Dx = Dx
    model.Dt = Dt

    model.Nx = size(vp, 1)
    model.Ny = size(vp, 2)

    model.Nb = Nb
    model.W0 = W0

    model.Rho = zeros(model.Nx, model.Ny)
    model.Q = zeros(model.Nx, model.Ny)
    model.Kappa = zeros(model.Nx, model.Ny)

    model.Dkappax = zeros(model.Nx, model.Ny)
    model.Dkappay = zeros(model.Nx, model.Ny)
    model.Drhox = zeros(model.Nx, model.Ny)
    model.Drhoy = zeros(model.Nx, model.Ny)

    model.Alpha1x = zeros(model.Nx, model.Ny)
    model.Alpha1y = zeros(model.Nx, model.Ny)
    model.Alpha2x = zeros(model.Nx, model.Ny)
    model.Alpha2y = zeros(model.Nx, model.Ny)
    model.Eta1x = zeros(model.Nx, model.Ny)
    model.Eta1y = zeros(model.Nx, model.Ny)
    model.Eta2x = zeros(model.Nx, model.Ny)
    model.Eta2y = zeros(model.Nx, model.Ny)

    model.dx = zeros(model.Nx)
    model.dy = zeros(model.Ny)

    for i in 1:model.Nx
        for j in 1:model.Ny
            model.Kappa[i, j] = rho[i, j] * vp[i, j] * vp[i, j]
            model.Rho[i, j] = rho[i, j]
            model.Q[i, j] = Q[i, j]
        end
    end

    Modeld(model.dx, model.Dx, model.Nb)
    Modeld(model.dy, model.Dx, model.Nb)

    for i in 1:model.Nx
        for j in 1:model.Ny
            Qmin = 1.1
            tau0min = Qmin / model.W0
            tau0min = Qmin / tau0min

            Qmax = model.Q[Nb, j]
            tau0max = Qmax / model.W0
            tau0max = 1.0 / tau0max

            tau0x = tau0min + (tau0max - tau0min) * model.dx[i]

            Qmax = model.Q[i, Nb]
            tau0max = Qmax / model.W0
            tau0max = 1.0 / tau0max

            tau0y = tau0min + (tau0max - tau0min) * model.dy[j]

            argx = model.dx[i]
            argy = model.dy[j]

            model.Alpha1x[i, j] = exp(-argx) * exp(-model.Dt * tau0x)
            model.Alpha1y[i, j] = exp(-argy) * exp(-model.Dt * tau0y)
            model.Alpha2x[i, j] = -model.Dt * tau0x
            model.Alpha2y[i, j] = -model.Dt * tau0y
            model.Eta1x[i, j] = exp(-argx) * exp(-model.Dt * tau0x)
            model.Eta1y[i, j] = exp(-argy) * exp(-model.Dt * tau0y)
            model.Eta2x[i, j] = -model.Dt * tau0x
            model.Eta2y[i, j] = -model.Dt * tau0y
            
            model.Dkappax[i, j] = model.Kappa[i, j]
            model.Dkappay[i, j] = model.Kappa[i, j]
            model.Drhox[i, j] = 1.0 / model.Rho[i, j]
            model.Drhoy[i, j] = 1.0 / model.Rho[i, j]
        end
    end
return model
end

function Model_sls(model::Model, vp, rho, Q, Dx, Dt, W0, Nb)
    model.Dx = Dx
    model.Dt = Dt

    model.Nx = size(vp, 1)
    model.Ny = size(vp, 2)

    model.Nb = Nb
    model.W0 = W0

    model.Rho = zeros(model.Nx, model.Ny)
    model.Q = zeros(model.Nx, model.Ny)
    model.Kappa = zeros(model.Nx, model.Ny)

    model.Dkappax = zeros(model.Nx, model.Ny)
    model.Dkappay = zeros(model.Nx, model.Ny)
    model.Drhox = zeros(model.Nx, model.Ny)
    model.Drhoy = zeros(model.Nx, model.Ny)

    model.Alpha1x = zeros(model.Nx, model.Ny)
    model.Alpha1y = zeros(model.Nx, model.Ny)
    model.Alpha2x = zeros(model.Nx, model.Ny)
    model.Alpha2y = zeros(model.Nx, model.Ny)
    model.Eta1x = zeros(model.Nx, model.Ny)
    model.Eta1y = zeros(model.Nx, model.Ny)
    model.Eta2x = zeros(model.Nx, model.Ny)
    model.Eta2y = zeros(model.Nx, model.Ny)

    model.dx = zeros(model.Nx)
    model.dy = zeros(model.Ny)

    for i in 1:model.Nx
        for j in 1:model.Ny
            model.Kappa[i, j] = rho[i, j] * vp[i, j] * vp[i, j]
            model.Rho[i, j] = rho[i, j]
            model.Q[i, j] = Q[i, j]
        end
    end

    Modeld(model.dx, model.Dx, model.Nb)
    Modeld(model.dy, model.Dx, model.Nb)

    for i in 1:model.Nx
        for j in 1:model.Ny
            tau0 = 1.0 / model.W0
            Qmin = 1.1

            tauemin = (tau0 / Qmin) * (sqrt(Qmin * Qmin + 1.0) + 1.0)
            tauemin = 1.0 / tauemin
            tausmin = (tau0 / Qmin) * (sqrt(Qmin * Qmin + 1.0) - 1.0)
            tausmin = 1.0 / tausmin
            Qmax = model.Q[Nb, j]

            tauemax = (tau0 / Qmin) * (sqrt(Qmax * Qmax + 1.0) + 1.0)
            tauemax = 1.0 / tauemax
            tausmax = (tau0 / Qmin) * (sqrt(Qmax * Qmax + 1.0) - 1.0)
            tausmax = 1.0 / tausmax
            tauex = tauemin + (tauemax - tauemin) * model.dx[i]
            tausx = tausmin + (tausmax - tausmin) * model.dx[i]
            Qmax = model.Q[i, Nb]

            tauemax = (tau0 / Qmin) * (sqrt(Qmax * Qmax + 1.0) + 1.0)
            tauemax = 1.0 / tauemax
            tausmax = (tau0 / Qmin) * (sqrt(Qmax * Qmax + 1.0) - 1.0)
            tausmax = 1.0 / tausmax

            # Interpolate relaxation times
            tauey = tauemin + (tauemax - tauemin) * model.dy[j]
            tausy = tausmin + (tausmax - tausmin) * model.dy[j]

            # In the equations below the relaxation times taue and taus
            # are inverses (1 / taue, 1 / taus)
            # Compute alpha and eta coefficients
            argx = model.dx[i]    # Temp variables
            argy = model.dy[j]    # Temp variables

            # An extra tapering factor of exp(-(x / L)^2)
            # is used to taper some coefficients
            model.Alpha1x[i, j] = exp(-argx) * exp(-model.Dt * tausx)
            model.Alpha1y[i, j] = exp(-argy) * exp(-model.Dt * tausy)
            model.Alpha2x[i, j] = model.Dt * tauex
            model.Alpha2y[i, j] = model.Dt * tauey
            model.Eta1x[i, j] = exp(-argx) * exp(-model.Dt * tausx)
            model.Eta1y[i, j] = exp(-argy) * exp(-model.Dt * tausy)
            model.Eta2x[i, j] = model.Dt * tauex
            model.Eta2y[i, j] = model.Dt * tauey

            # Compute the change in moduli due to
            # visco-ealsticity (is equal to zero for the elastic case)
            model.Dkappax[i, j] = model.Kappa[i, j] * (1.0 - tausx / tauex)
            model.Dkappay[i, j] = model.Kappa[i, j] * (1.0 - tausy / tauey)
            model.Drhox[i, j] = (1.0 / model.Rho[i, j]) * (1.0 - tausx / tauex)
            model.Drhoy[i, j] = (1.0 / model.Rho[i, j]) * (1.0 - tausy / tauey)
        end
    end

return model
end


function stability(model)
    """
    Model stability - checks the velocity model, the time step increment and spatial increment for stability.
    
    Parameters:
    - model: Model object
    
    Returns:
    - Stability index (Float64)
    """
    nx, ny = model.Nx, model.Ny
    stab  = 0.0
    
    for i in 1:nx
        for j in 1:ny
            vp = sqrt(model.Kappa[i, j] / model.Rho[i, j])
            stab = (vp * model.Dt) / model.Dx
            
            if stab > 1.0 / sqrt(2.0)
                println("Stability index too large! $stab")
            end
        end
    end
    
    return stab
end

function Modeld(d, dx, nb)
    """
    Modeld creates a 1D profile function tapering the left
    and right borders. 
    
    Parameters:
    - d  : Input 1D float array
    - dx : Grid spacing
    - nb : Width of border zone 
    
    Returns:
    - OK if no error, ERR in all other cases.
    """
    n = length(d)

    for i in 1:n
        d[i] = 1.0
    end

    # Taper left border
    if nb != 0
        for i in 1:nb
            d[i] = d[i] * ((float(i)*dx)/(float(nb)*dx))^2
        end
    end

    # Taper right border 
    if nb != 0
        for i in (n-nb):(n-1)
            d[i] = d[i] * ((float(n-1-i)*dx)/(float(nb)*dx))^2
        end
    end
    
    return "OK"
end
