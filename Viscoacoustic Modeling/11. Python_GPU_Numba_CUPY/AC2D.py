"""
AC2D object

This script defines a AC2D class for solving the acoustic wave equation in 2D.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import matplotlib.pyplot as plt
import numpy as np

from numba import jit, float64, int64
from numba.experimental import jitclass
from numba import prange
import numba
import cupy as cp
import numba.cuda as cuda
import timeit

# Import Classes  
# ----------------
from Differentiator import * 
from Model import * 
from Src import * 
from Rec import * 

                            

spec = [
    ('p', float64[:,:]),
    ('vx', float64[:,:]),
    ('vy', float64[:,:]),
    ('exx', float64[:,:]),
    ('eyy', float64[:,:]),
    ('gammax', float64[:,:]),
    ('gammay', float64[:,:]),
    ('thetax', float64[:,:]),
    ('thetay', float64[:,:]),
    ('ts', int64),
]
# ----------------
# AC2D Class
# ----------------
#@jitclass(spec)
class Ac2d:
    def __init__(self, Model):
        """
        Initializes an Ac2d object with the given model object.

        Parameters
        ----------
        Model : Model object
                The model object containing the simulation parameters.
        """

        self.p = np.zeros((Model.Nx, Model.Ny))                     # Stress
        self.vx = np.zeros((Model.Nx, Model.Ny))                    # Particle velocity in x-direction
        self.vy = np.zeros((Model.Nx, Model.Ny))                    # Particle velocity in y-direction
        self.exx = np.zeros((Model.Nx, Model.Ny))                   # Time derivative of Strain in x-direction
        self.eyy = np.zeros((Model.Nx, Model.Ny))                   # Time derivative of Strain in y-direction
        
        self.gammax = np.zeros((Model.Nx, Model.Ny), dtype=float)   
        self.gammay = np.zeros((Model.Nx, Model.Ny), dtype=float)
        self.thetax = np.zeros((Model.Nx, Model.Ny), dtype=float)
        self.thetay = np.zeros((Model.Nx, Model.Ny), dtype=float)
        
        self.ts = 0
    
    def Ac2dSolve(Ac2d, Model, Src, Rec, nt, l):
        """
        Ac2dSolve computes the solution of the acoustic wave equation.
        The acoustic equation of motion are integrated using Virieux's (1986) stress-velocity scheme.
        (See the notes.tex file in the Doc directory).
        
            vx(t+dt)   = dt/rhox d^+x[ sigma(t)] + dt fx + vx(t)
                       + thetax D[1/rhox]
            vy(t+dt)   = dt/rhoy d^+y sigma(t) + dt fy(t) + vy(t)
                       + thetay D[1/rhoy]

            dp/dt(t+dt) = dt Kappa[d^-x dexx/dt + d-y deyy/dt + dt dq/dt(t) 
                        + dt [gammax Dkappa + gammay Dkappa]
                        + p(t)

            dexx/dt     =  d^-_x v_x 
            deyy/dt     =  d^-_z v_y 

            gammax(t+dt) = alpha1x gammax(t) + alpha2x dexx/dt 
            gammay(t+dt) = alpha1y gammay(t) + alpha2y deyy/dt 

            thetax(t+dt) = eta1x thetax(t) + eta2x d^+x p
            thetay(t+dt) = eta1y thetay(t) + eta2y d^+y p
            
            Parameters:
                Ac2d : Solver object
                Model: Model object
                Src  : Source object
                Rec  : Receiver object
                nt   : Number of timesteps to do starting with current step 
                l    : The differentiator operator length
        
        """
        
        Diff = Differentiator(l)                            # Create differentiator object
        oldperc = 0.0                                       # Old percentage for printing progress
        ns = Ac2d.ts                                        # Start timestep
        ne = ns + nt                                        # Stop timestep

        sx = int(round(Src.Sx[0]))
        sy = int(round(Src.Sy[0]))

        # Define the CuPy arrays
        p_gpu = cp.asarray(Ac2d.p)
        vx_gpu = cp.asarray(Ac2d.vx)
        vy_gpu = cp.asarray(Ac2d.vy)
        exx_gpu = cp.asarray(Ac2d.exx)
        eyy_gpu = cp.asarray(Ac2d.eyy)
        gammax_gpu = cp.asarray(Ac2d.gammax)
        gammay_gpu = cp.asarray(Ac2d.gammay)
        thetax_gpu = cp.asarray(Ac2d.thetax)
        thetay_gpu = cp.asarray(Ac2d.thetay)
        rho_gpu = cp.asarray(Model.Rho)
        kappa_gpu = cp.asarray(Model.Kappa)
        drhox_gpu = cp.asarray(Model.Drhox)
        drhoy_gpu = cp.asarray(Model.Drhoy)
        dkappax_gpu = cp.asarray(Model.Dkappax)
        dkappay_gpu = cp.asarray(Model.Dkappay)
        eta1x_gpu = cp.asarray(Model.Eta1x)
        eta1y_gpu = cp.asarray(Model.Eta1y)
        eta2x_gpu = cp.asarray(Model.Eta2x)
        eta2y_gpu = cp.asarray(Model.Eta2y)
        alpha1x_gpu = cp.asarray(Model.Alpha1x)
        alpha1y_gpu = cp.asarray(Model.Alpha1y)
        alpha2x_gpu = cp.asarray(Model.Alpha2x)
        alpha2y_gpu = cp.asarray(Model.Alpha2y)
        src_gpu = cp.asarray(Src.Src)
        w_gpu = cp.asarray(Diff.w)


        # Set up the grid and block dimensions
        #threads_per_block = 1024
        #blocks_per_grid = (x.size + (threads_per_block - 1)) // threads_per_block

        # Calculate the grid and block dimensions
        block_dim = (16, 16)
        grid_dim = (int(np.ceil(Model.Nx / block_dim[0])), int(np.ceil(Model.Ny / block_dim[1])))

        print("START________________________________")
        cp.cuda.Stream.null.synchronize()
        
        # Loop over timesteps
        for i in range(ns, ne):
            if i==1:
                start = timeit.default_timer()
            # Compute spatial derivative of stress
            # Use exx and eyy as temp storage
            DiffDxplus_kernel[grid_dim, block_dim](exx_gpu, p_gpu, w_gpu, Model.Dx, Model.Nx, Model.Ny, l)     # Forward differentiation x-axis
            Ac2dvx_kernel[grid_dim, block_dim](vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, Model.Dt, Model.Nx, Model.Ny)                             # Compute vx

            DiffDyplus_kernel[grid_dim, block_dim](eyy_gpu, p_gpu, w_gpu, Model.Dx, Model.Nx, Model.Ny, l)     # Forward differentiation y-axis
            Ac2dvy_kernel[grid_dim, block_dim](vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, Model.Dt, Model.Nx, Model.Ny)                              # Compute vy

            # Compute time derivative of strains
            #Diff.DiffDxminus(Ac2d.vx, Ac2d.exx, Model.Dx)   # Compute exx
            DiffDxminus_kernel[grid_dim, block_dim](exx_gpu, vx_gpu, w_gpu, Model.Dx, Model.Nx, Model.Ny, l)

            #Diff.DiffDyminus(Ac2d.vy, Ac2d.eyy, Model.Dx)   # Compute eyy
            DiffDyminus_kernel[grid_dim, block_dim](eyy_gpu, vy_gpu, w_gpu, Model.Dx, Model.Nx, Model.Ny, l)

            # Update stress
            #Ac2d.Ac2dstress(Model)
            Ac2dstress_kernel[grid_dim, block_dim](p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, Model.Dt, Model.Nx, Model.Ny)


            # Add source
            update_pressure_kernel(p_gpu, rho_gpu, Src.Src[i], Model.Dt, Model.Dx, sx, sy)
            #Ac2d.p[sx, sy] += Model.Dt * (Src.Src[i] / (Model.Dx * Model.Dx * Model.Rho[sx,sy]))    


            # Print progress
            perc = 1000.0 * (float(i) / float(ne - ns - 1)) # Current Percentage of completion
            if perc - oldperc >= 10.0:
                iperc = int(perc) // 10                     
                if iperc % 10 == 0:
                    print(iperc)                            # Percentage finished
                oldperc = perc


            Ac2d.p = cp.asnumpy(p_gpu)
            # Record Wavefield
            Rec.rec_wavefield(i, Ac2d.p)
            # Record Seismogram
            Rec.rec_seismogram(i, Ac2d.p)
            # Record 1 Trace
            Rec.rec_trace(i, Ac2d.p)
            
        cp.cuda.Stream.null.synchronize()
        end = timeit.default_timer() 
        
        print("Execution Time: ", (end - start))
        #print(f"Size of one element: {Rec.wavefield.itemsize} bytes")
        #print(f"Size of the whole array: {Rec.wavefield.nbytes} bytes")
        cp.cuda.MemoryPool().free_all_blocks()
        return "OK"

        
def update_pressure_kernel(p_gpu, rho_gpu, src_val, Dt, Dx, sx, sy):
    p_gpu[sx, sy] += Dt * (src_val / (Dx * Dx * rho_gpu[sx, sy]))

@cuda.jit
def Ac2dvx_kernel(vx_gpu, exx_gpu, thetax_gpu, rho_gpu, drhox_gpu, eta1x_gpu, eta2x_gpu, dt, nx, ny):
    x, y = cuda.grid(2)

    if x < nx and y < ny:
        vx_gpu[x, y] = dt * (1.0 / rho_gpu[x, y]) * exx_gpu[x, y] + vx_gpu[x, y] + dt * thetax_gpu[x, y] * drhox_gpu[x, y]
        thetax_gpu[x, y] = eta1x_gpu[x, y] * thetax_gpu[x, y] + eta2x_gpu[x, y] * exx_gpu[x, y]

@cuda.jit
def Ac2dvy_kernel(vy_gpu, eyy_gpu, thetay_gpu, rho_gpu, drhoy_gpu, eta1y_gpu, eta2y_gpu, dt, nx, ny):
    x, y = cuda.grid(2)

    if x < nx and y < ny:
        vy_gpu[x, y] = dt * (1.0 / rho_gpu[x, y]) * eyy_gpu[x, y] + vy_gpu[x, y] + dt * thetay_gpu[x, y] * drhoy_gpu[x, y]
        thetay_gpu[x, y] = eta1y_gpu[x, y] * thetay_gpu[x, y] + eta2y_gpu[x, y] * eyy_gpu[x, y]


@cuda.jit
def Ac2dstress_kernel(p_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, kappa_gpu, dkappax_gpu, dkappay_gpu, alpha1x_gpu, alpha1y_gpu, alpha2x_gpu, alpha2y_gpu, dt, nx, ny):
    x, y = cuda.grid(2)

    if x < nx and y < ny:
        p_gpu[x, y] = dt * kappa_gpu[x, y] * (exx_gpu[x, y] + eyy_gpu[x, y]) + p_gpu[x, y] + dt * (gammax_gpu[x, y] * dkappax_gpu[x, y] + gammay_gpu[x, y] * dkappay_gpu[x, y])
        gammax_gpu[x, y] = alpha1x_gpu[x, y] * gammax_gpu[x, y] + alpha2x_gpu[x, y] * exx_gpu[x, y]
        gammay_gpu[x, y] = alpha1y_gpu[x, y] * gammay_gpu[x, y] + alpha2y_gpu[x, y] * eyy_gpu[x, y]

@cuda.jit
def DiffDyminus_kernel(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l):
    i, j = cuda.grid(2)

    if i < nx and j < ny:
        if j < l:
            sum = 0.0
            for k in range(1, j + 1):
                sum -= w_gpu[k - 1] * A_gpu[i, j - k]
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * A_gpu[i, j + (k - 1)]
            dA_gpu[i, j] = sum / dx

        elif j >= l and j < ny - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * (-A_gpu[i, j - k] + A_gpu[i, j + (k - 1)])
            dA_gpu[i, j] = sum / dx

        elif j >= ny - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum -= w_gpu[k - 1] * A_gpu[i, j - k]
            for k in range(1, ny - j + 1):
                sum += w_gpu[k - 1] * A_gpu[i, j + (k - 1)]
            dA_gpu[i, j] = sum / dx

@cuda.jit
def DiffDxminus_kernel(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l):
    i, j = cuda.grid(2)

    if i < nx and j < ny:
        if i < l:
            sum = 0.0
            for k in range(1, i + 1):
                sum -= w_gpu[k - 1] * A_gpu[i - k, j]
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * A_gpu[i + (k - 1), j]
            dA_gpu[i, j] = sum / dx

        elif i >= l and i < nx - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * (-A_gpu[i - k, j] + A_gpu[i + (k - 1), j])
            dA_gpu[i, j] = sum / dx

        elif i >= nx - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum -= w_gpu[k - 1] * A_gpu[i - k, j]
            for k in range(1, nx - i + 1):
                sum += w_gpu[k - 1] * A_gpu[i + (k - 1), j]
            dA_gpu[i, j] = sum / dx

@cuda.jit
def DiffDyplus_kernel(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l):
    i, j = cuda.grid(2)

    if i < nx and j < ny:
        if j < l:
            sum = 0.0
            for k in range(1, j + 2):
                sum -= w_gpu[k - 1] * A_gpu[i, j - (k - 1)]
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * A_gpu[i, j + k]
            dA_gpu[i, j] = sum / dx

        elif j >= l and j < ny - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * (-A_gpu[i, j - (k - 1)] + A_gpu[i, j + k])
            dA_gpu[i, j] = sum / dx

        elif j >= ny - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum -= w_gpu[k - 1] * A_gpu[i, j - (k - 1)]
            for k in range(1, ny - j):
                sum += w_gpu[k - 1] * A_gpu[i, j + k]
            dA_gpu[i, j] = sum / dx


@cuda.jit
def DiffDxplus_kernel(dA_gpu, A_gpu, w_gpu, dx, nx, ny, l):
    i, j = cuda.grid(2)

    if i < nx and j < ny:

        if i < l:
            sum = 0.0
            for k in range(1, i + 2):
                sum -= w_gpu[k - 1] * A_gpu[i - (k - 1), j]
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * A_gpu[i + k, j]

            dA_gpu[i, j] = sum / dx

        elif i >= l and i < nx - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum += w_gpu[k - 1] * (-A_gpu[i - (k - 1), j] + A_gpu[i + k, j])
            dA_gpu[i, j] = sum / dx

        elif i >= nx - l:
            sum = 0.0
            for k in range(1, l + 1):
                sum -= w_gpu[k - 1] * A_gpu[i - (k - 1), j]
            for k in range(1, nx - i):
                sum += w_gpu[k - 1] * A_gpu[i + k, j]

            dA_gpu[i, j] = sum / dx
