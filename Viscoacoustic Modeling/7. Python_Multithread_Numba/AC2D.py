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

# Import Classes  
# ----------------
from Differentiator import * 
from Model import * 
from Src import * 
from Rec import * 

numba.set_num_threads(16)


@jit(nopython=True,parallel=True, fastmath=True)
def Ac2dvx_numba(nx, ny, Dt, Rho, exx, vx, Drhox, thetax, Eta1x, Eta2x):
        """
        Ac2vx computes the x-component of particle velocity
        
        Parameters:
            Ac2d : Solver object 
            Model: Model object

        """
                     # Number of grid points in y-direction

        # The derivative of stress in x-direction is stored in exx
        # Scale with inverse density and advance one time step
        for i in prange(nx):
            for j in prange(ny):
                vx[i, j] = Dt * (1.0 / Rho[i, j]) * exx[i, j] + vx[i, j] + Dt * thetax[i, j] * Drhox[i, j] 
                            
                thetax[i, j]  = Eta1x[i, j] * thetax[i, j] + Eta2x[i, j] * exx[i, j]
        
@jit(nopython=True, parallel=True, fastmath=True)        
def Ac2dvy_numba(nx, ny, Dt, Rho, eyy, vy, Drhoy, thetay, Eta1y, Eta2y):
    """
    Ac2vy computes the y-component of particle velocity
    
    Parameters:
        Ac2d : Solver object 
        Model: Model object

    """
            # Number of grid points in y-direction

    # The derivative of stress in y-direction is stored in eyy
    # Scale with inverse density and advance one time step
    for i in prange(nx):
        for j in prange(ny):
            vy[i, j] = Dt * (1.0 / Rho[i, j]) * eyy[i, j] + vy[i, j] + Dt * thetay[i, j] * Drhoy[i, j]
                        
            thetay[i, j]  = Eta1y[i, j] * thetay[i, j] + Eta2y[i, j] * eyy[i, j]

                            
@jit(nopython=True, parallel=True, fastmath=True)
def Ac2dstress_numba(nx, ny, Dt, Kappa, exx, eyy, p, gammax, gammay, Dkappax, Dkappay, Alpha1x, Alpha1y, Alpha2x, Alpha2y):
    """
    Ac2dstress computes acoustic stress 
    
    Parameters:
        Ac2d : Solver object 
        Model: Model object

    """
                 # Number of grid points in y-direction

    for i in prange(nx):
        for j in prange(ny):
            p[i, j] = Dt * Kappa[i, j] * (exx[i, j] + eyy[i, j]) \
                            + p[i, j] \
                            + Dt * (gammax[i, j] * Dkappax[i, j] + gammay[i, j] * Dkappay[i, j])
                    
            gammax[i, j] = Alpha1x[i, j] * gammax[i, j] + Alpha2x[i, j] * exx[i, j]
            gammay[i, j] = Alpha1y[i, j] * gammay[i, j] + Alpha2y[i, j] * eyy[i, j]

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
        
        # Loop over timesteps
        for i in range(ns, ne):
            # Compute spatial derivative of stress
            # Use exx and eyy as temp storage
            Diff.DiffDxplus(Ac2d.p, Ac2d.exx, Model.Dx)     # Forward differentiation x-axis
            Ac2d.Ac2dvx(Model)                              # Compute vx
            Diff.DiffDyplus(Ac2d.p, Ac2d.eyy, Model.Dx)     # Forward differentiation y-axis
            Ac2d.Ac2dvy(Model)                              # Compute vy

            # Compute time derivative of strains
            Diff.DiffDxminus(Ac2d.vx, Ac2d.exx, Model.Dx)   # Compute exx
            Diff.DiffDyminus(Ac2d.vy, Ac2d.eyy, Model.Dx)   # Compute eyy

            # Update stress
            Ac2d.Ac2dstress(Model)

            # Add source
            for k in range(Src.Ns):
                sx = int(Src.Sx)#[k]                        # Source x-coordinate
                sy = int(Src.Sy)#[k]                        # Source y-coordinate
                Ac2d.p[sx, sy] += Model.Dt * (Src.Src[i] / (Model.Dx * Model.Dx * Model.Rho[sx,sy]))      

            # Print progress
            perc = 1000.0 * (float(i) / float(ne - ns - 1)) # Current Percentage of completion
            if perc - oldperc >= 10.0:
                iperc = int(perc) // 10                     
                if iperc % 10 == 0:
                    print(iperc)                            # Percentage finished
                oldperc = perc
                

            # Record Wavefield
            Rec.rec_wavefield(i, Ac2d.p)

            # Record Seismogram
            Rec.rec_seismogram(i, Ac2d.p)
            
            # Record 1 Trace
            Rec.rec_trace(i, Ac2d.p)
            
 
        #print(f"Size of one element: {Rec.wavefield.itemsize} bytes")
        #print(f"Size of the whole array: {Rec.wavefield.nbytes} bytes")
            
        return "OK"

    def Ac2dvx(Ac2d, Model):
        """
        Ac2vx computes the x-component of particle velocity
        
        Parameters:
            Ac2d : Solver object 
            Model: Model object

        """
        nx = Model.Nx               # Number of grid points in x-direction
        ny = Model.Ny               # Number of grid points in y-direction

        # The derivative of stress in x-direction is stored in exx
        # Scale with inverse density and advance one time step
        Ac2dvx_numba(nx, ny, Model.Dt, Model.Rho, Ac2d.exx, Ac2d.vx, Model.Drhox, Ac2d.thetax, Model.Eta1x, Model.Eta2x)
  
        
        
    def Ac2dvy(Ac2d, Model):
        """
        Ac2vy computes the y-component of particle velocity
        
        Parameters:
            Ac2d : Solver object 
            Model: Model object

        """
        nx = Model.Nx           # Number of grid points in x-direction
        ny = Model.Ny           # Number of grid points in y-direction

        # The derivative of stress in y-direction is stored in eyy
        # Scale with inverse density and advance one time step
        Ac2dvy_numba(nx, ny, Model.Dt, Model.Rho, Ac2d.eyy, Ac2d.vy, Model.Drhoy, Ac2d.thetay, Model.Eta1y, Model.Eta2y)

                                

    def Ac2dstress(Ac2d, Model):
        """
        Ac2dstress computes acoustic stress 
        
        Parameters:
            Ac2d : Solver object 
            Model: Model object

        """
        nx = Model.Nx               # Number of grid points in x-direction
        ny = Model.Ny               # Number of grid points in y-direction

        Ac2dstress_numba(nx, ny, Model.Dt, Model.Kappa, Ac2d.exx, Ac2d.eyy, Ac2d.p, Ac2d.gammax, Ac2d.gammay, Model.Dkappax, Model.Dkappay, Model.Alpha1x, Model.Alpha1y, Model.Alpha2x, Model.Alpha2y)
   
             
