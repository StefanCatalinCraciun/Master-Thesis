"""

Acoustic 2D modeling script.

This script demonstrates how to use the Model, Src, Rec, and AC2D classes
to perform a 2D acoustic wave simulation.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import time
import numpy as np
import matplotlib.pyplot as plt

# Import Classes
# ----------------
from Model import Model
from Src import Src
from Rec import Rec
from AC2D import Ac2d

# Main Function
def main():
    #------------------------------
    #Set all modeling parameters
    #------------------------------
    nx      = 101     # No of grdipoints in x-direction
    ny      = 101     # No of gridpoints in y-direction
    dx      = 10           # Grid interval
    dt      = 0.0005;      # Time sampling interval
    nt      = 1000         # No of time steps
    q0      = 10e5          # Quality factor for attenuation (the higher, the less attenuation)
    Nb      = 15 #35       # No of grid points for the absorbing boundary
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
    gx = int(nx/2)         # x-position of receiver
    gy = 80                # x-position of receiver
    
    #----------------------------------------------
    #Create source position and source wavelet
    #----------------------------------------------
    #Source Position
    sx=np.zeros(1)         # you can set up multiple sources
    sy=np.zeros(1)          
    sx[0]=nx/2             # x-position of source
    sy[0]=ny/2             # y-position of source

    wavelet = np.zeros(nt) # Initialize source pulse
    
    # Ricker wavelet parameters
    f0      = 25.0              # Dominant frequency 
    w0      = 2.0 * np.pi * f0  # Dominant angular frequency
    #t0= 2. / f0
    t0      = 0.04              # Pulse delay
    Length  = nt                # Length of wavelet

    # Create a Ricker wavelet
    ricker_wavelet = Src.SrcRicker(wavelet, t0, f0, Length, dt)
    
    #Create source object
    src=Src(ricker_wavelet,sx,sy)
    
    # Plot the Ricker wavelet
    time_source = [i * dt for i in range(Length)]
    stop = int(Length/4)
    plt.plot(time_source[:stop], src.Src[:stop],'b-',lw=3,label="Source Pulse")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.title("Ricker Wavelet")
    plt.xlim(0, time_source[stop])
    plt.legend()
    plt.grid(True)
    plt.savefig("Visualization/Source Pulse.svg", format="svg", bbox_inches = 'tight')
    plt.show()
    
    
    #----------------------------------------------
    #Create receiver positions
    #----------------------------------------------
    nr = 100                    # No of receivers
    rx=np.zeros((nr))           
    ry=np.zeros((nr))
    for i in range(nr):
        rx[i] = 20              # x-position of receiver
        ry[i] = i               # y-position of receiver
        
    
    #Create receiver object
    rec= Rec(rx,ry,nt,resamp,sresamp,nx,ny,dx,Nb,dt,gx,gy);
    
    '''
    # create a function that reads a bin file that containt the velocity model and returns a numpy array that is like the vp array below
    vp = np.fromfile('marmousi_II_marine.vp', dtype=np.float32)
    #reshape the array to the shape of the model
    vp = vp.reshape((500,174))
    plt.imshow(vp.T, cmap='jet')
    plt.show()
    
    # create a function that reads a bin file that containt the velocity model and returns a numpy array that is like the vp array below
    rho = np.fromfile('marmousi_II_marine.rho', dtype=np.float32)
    #reshape the array to the shape of the model
    rho = rho.reshape((500,174))
    plt.imshow(rho.T, cmap='jet')
    plt.show()
    '''

    #----------------------------------------
    # Create model
    #----------------------------------------

    # Get the velocity model
    vp = np.ones((nx,ny)) * vp0
    #vp[50:,:] = 3000
    plt.imshow(vp.T, cmap='jet')
    plt.xlabel("X Gridpoints")
    plt.ylabel("Y Gridpoints")
    plt.title("Velocity Model")
    plt.xlim(0, nx-1)
    plt.ylim(ny-1, 0)
    plt.grid(True)
    # Add a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label("Velocity (m/s)")
    plt.savefig("Visualization/Velocity Model.svg", format="svg", bbox_inches = 'tight')
    plt.show()
  
    #Get the density model
    rho = np.ones((nx,ny)) * rho0
    plt.imshow(rho.T, cmap='jet')
    plt.xlabel("X Gridpoints")
    plt.ylabel("Y Gridpoints")
    plt.title("Density Model")
    plt.xlim(0, nx-1)
    plt.ylim(ny-1, 0)
    plt.grid(True)
    # Add a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label("Density (kg/m3)")
    plt.savefig("Visualization/Density Model.svg", format="svg", bbox_inches = 'tight')
    plt.show()
    
    #Get the Q model
    Q = np.ones((nx,ny)) * q0
    plt.imshow(Q.T, cmap='jet')
    plt.xlabel("X Gridpoints")
    plt.ylabel("Y Gridpoints")
    plt.title("Q Model")
    plt.xlim(0, nx-1)
    plt.ylim(ny-1, 0)
    plt.grid(True)
    # Add a colorbar legend
    cbar = plt.colorbar()
    cbar.set_label("Quality")
    plt.savefig("Visualization/Q Model.svg", format="svg", bbox_inches = 'tight')
    plt.show()

    # Create a new model
    model = Model.ModelNew(Model(vp),vp, rho, Q, dx, dt, w0, Nb, rheol)
    # Compute stability index
    #si = model.stability()          
    #print(f"Stability index: {si}")
    
    
    #--------------------------------------
    #Create FD solver
    #--------------------------------------
    #Create solver object
    ac2d = Ac2d(model)
    
    #--------------------------------------
    #Run solver
    #--------------------------------------
    t1 = time.perf_counter()
    
    Ac2d.Ac2dSolve(ac2d,model,src,rec,nt,l)
    
    t2 = time.perf_counter()
    print("Solver wall clock time: ", t2 - t1)
    
    #--------------------------------------
    #Compute analytic solution
    #--------------------------------------
    rec.rec_trace_analytical(src, model)

    #--------------------------------------
    #Save Recording
    #--------------------------------------
    rec.wavefield_save()
    rec.seismogram_save()
    rec.trace_save()
    rec.trace_analytical_save()
    

if __name__ == "__main__":
    main()
