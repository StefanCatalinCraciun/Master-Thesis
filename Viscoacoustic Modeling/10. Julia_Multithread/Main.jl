"""

Acoustic 2D modeling script.

This script demonstrates how to use the Model, Src, Rec, and AC2D structures
to perform a 2D acoustic wave simulation.

@author: Stefan Catalin Craciun
"""

using BenchmarkTools
# Import Libraries
# ----------------
using PyPlot
using Dates
#using PyCall
#using LinearAlgebra
#@pyimport numpy as np
#@pyimport matplotlib.pyplot as plt

# Import Classes
# ----------------
include("Model.jl")
include("Src.jl")
include("Rec.jl")
include("Differentiator.jl")
include("AC2D.jl")

# Main Function
function main()
    #------------------------------
    #Set all modeling parameters
    #------------------------------
    nx      = 101     # No of grdipoints in x-direction
    ny      = 101      # No of gridpoints in y-direction
    dx      = 10            # Grid interval
    dt      = 0.0005        # Time sampling interval
    nt      = 1000          # No of time steps
    q0      = 10e5          # Quality factor for attenuation (the higher, the less attenuation)
    Nb      = 15 #35        # No of grid points for the absorbing boundary
    #rheol = "SLS"
    rheol = "MAXWELL"      # Rheology model (SLS or MAXWELL)
    resamp  = 1            # Resampling factor (relative to no of timesteps) for seismogram. Leave as 1m as it messes the plot timescale.
    sresamp = 5            # Resampling factor for wavefield animation
    l       = 8            # Length of differentiator

    ### Homogeneous Model Elastic Parameters ###
    vp0     = 2000         # Velocity of Homogeneous medium 
    rho0    = 2000         # Density of Homogeneous medium
    
    ### 1 Trace Parameters ###
    # Do not put same as origin of source
    #gx = nx/2 
    gx = convert(Int, round(nx/2))         # x-position of receiver
    gy = convert(Int, round(ny/2)) + 30                                # y-position of receiver
    
    #----------------------------------------------
    #Create source position and source wavelet
    #----------------------------------------------
    #Source Position
    sx=zeros(1)             # you can set up multiple sources
    sy=zeros(1)      
      
    sx[1]=round(nx/2)              # x-position of source
    sy[1]=round(ny/2)              # y-position of source

    wavelet = zeros(Float64, nt) # Initialize source pulse
    
    # Ricker wavelet parameters
    f0      = 25.0              # Dominant frequency 
    w0      = 2.0 * pi * f0     # Dominant angular frequency
    #t0= 2. / f0
    t0      = 0.04              # Pulse delay
    Length  = nt                # Length of wavelet

    # Create a Ricker wavelet
    ricker_wavelet = SrcRicker(t0, f0, Length, dt)

    # Create source object
    src = Src(ricker_wavelet, sx, sy)
    
    
    # Plot the Ricker wavelet
    time_source = [i * dt for i in 0:(Length - 1)]
    stop = Int(round(Length / 4))
    plot(time_source[1:stop], src.Src[1:stop], "b-", lw=3, label="Source Pulse")
    xlabel("Time (s)")
    ylabel("Amplitude")
    title("Ricker Wavelet")
    xlim(0, time_source[stop])
    legend()
    grid(true)
    savefig("Visualization/Source Pulse.svg", format="svg", bbox_inches="tight")
    show()

    
    #----------------------------------------------
    #Create receiver positions
    #----------------------------------------------
    nr = nx#100                   # Number of receivers
    rx = zeros(nr)
    ry = zeros(nr)

    for i in 1:nr
        rx[i] = 40             # x-position of receiver
        ry[i] = i              # y-position of receiver
    end

    # Create receiver object
    rec = Rec(Int64.(rx), Int64.(ry), nt, resamp, sresamp, nx, ny, Float64(dx), Nb, dt, gx, gy)

    #----------------------------------------
    # Create model
    #----------------------------------------


    # Get the velocity model
    vp = ones(nx, ny) * vp0
    #vp[51:end,:] .= 3000
    imshow(vp', cmap="jet")
    xlabel("X Gridpoints")
    ylabel("Y Gridpoints")
    title("Velocity Model")
    xlim(0, nx - 1)
    ylim(ny - 1, 0)
    grid(true)
    # Add a colorbar legend
    cbar = colorbar()
    cbar.set_label("Velocity (m/s)")
    savefig("Visualization/Velocity Model.svg", format="svg", bbox_inches="tight")
    show()

    # Get the density model
    rho = ones(nx, ny) * rho0
    imshow(rho', cmap="jet")
    xlabel("X Gridpoints")
    ylabel("Y Gridpoints")
    title("Density Model")
    xlim(0, nx - 1)
    ylim(ny - 1, 0)
    grid(true)
    # Add a colorbar legend
    cbar = colorbar()
    cbar.set_label("Density (kg/m3)")
    savefig("Visualization/Density Model.svg", format="svg", bbox_inches="tight")
    show()
  
    # Get the density model
    Q = ones(nx, ny) * q0
    imshow(Q', cmap="jet")
    xlabel("X Gridpoints")
    ylabel("Y Gridpoints")
    title("Q Model")
    xlim(0, nx - 1)
    ylim(ny - 1, 0)
    grid(true)
    # Add a colorbar legend
    cbar = colorbar()
    cbar.set_label("Quality")
    savefig("Visualization/Q Model.svg", format="svg", bbox_inches="tight")
    show()

    # Create a new model
    model = Model()
    model = ModelNew(model, vp, rho, Q, dx, dt, w0, Nb, rheol)
    # Compute stability index
    si = stability(model)
    println("Stability index: $si")

    
    #--------------------------------------
    # Create FD solver
    #--------------------------------------
    # Create solver object
    ac2d = Ac2d(model)
    
    #--------------------------------------
    # Run solver
    #--------------------------------------
    t1 = time()

    Ac2dSolve(ac2d, model, src, rec, nt, l)

    t2 = time()
    println("Solver wall clock time: ", t2 - t1)
           
  
    #--------------------------------------
    # Compute analytic solution
    #--------------------------------------
    rec_trace_analytical!(rec, src, model)

    #--------------------------------------
    # Save Recording
    #--------------------------------------
    wavefield_save!(rec)
    seismogram_save!(rec)
    trace_save!(rec)
    trace_analytical_save!(rec)
    
    #imshow(rec.wavefield[:, :, 600], cmap="jet")
    #show()

    println(size(rec.wavefield))


    #println()
    #println(length(ricker_wavelet))
end



main()
