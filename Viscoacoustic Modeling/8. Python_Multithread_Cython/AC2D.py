"""
AC2D object

This script defines a AC2D class for solving the acoustic wave equation in 2D.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import matplotlib.pyplot as plt
import numpy as np
from Cython_Functions import Ac2dvx_Cython, Ac2dvy_Cython, Ac2dstress_Cython

# Import Classes  
# ----------------
from Differentiator import * 
from Model import * 
from Src import * 
from Rec import * 
import os
import time


# ----------------
# AC2D Class
# ----------------
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
        threads = 16
        Diff = Differentiator(l)                            # Create differentiator object
        oldperc = 0.0                                       # Old percentage for printing progress
        ns = Ac2d.ts                                        # Start timestep
        ne = ns + nt                                        # Stop timestep
        
        # Loop over timesteps
        for i in range(ns, ne):
            # Compute spatial derivative of stress
            # Use exx and eyy as temp storage
            Diff.DiffDxplus(Ac2d.p, Ac2d.exx, Model.Dx, threads)     # Forward differentiation x-axis
            Ac2d.Ac2dvx(Model, threads)                              # Compute vx
            Diff.DiffDyplus(Ac2d.p, Ac2d.eyy, Model.Dx, threads)     # Forward differentiation y-axis
            Ac2d.Ac2dvy(Model, threads)                              # Compute vy

            # Compute time derivative of strains
            Diff.DiffDxminus(Ac2d.vx, Ac2d.exx, Model.Dx, threads)   # Compute exx
            Diff.DiffDyminus(Ac2d.vy, Ac2d.eyy, Model.Dx, threads)   # Compute eyy

            # Update stress
            Ac2d.Ac2dstress(Model, threads)

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


    def Ac2dvx(Ac2d, Model, threads):
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
        Ac2dvx_Cython(nx, ny, Model.Dt, Ac2d.vx, Model.Rho, Ac2d.exx, Ac2d.thetax, Model.Drhox, Model.Eta1x, Model.Eta2x, threads)
        
        '''
        for i in range(nx):
            for j in range(ny):
                Ac2d.vx[i, j] = Model.Dt * (1.0 / Model.Rho[i, j]) * Ac2d.exx[i, j] + Ac2d.vx[i, j] + Model.Dt * Ac2d.thetax[i, j] * Model.Drhox[i, j] 
                            
                Ac2d.thetax[i, j]  = Model.Eta1x[i, j] * Ac2d.thetax[i, j] + Model.Eta2x[i, j] * Ac2d.exx[i, j]
        '''
        


    def Ac2dvy(Ac2d, Model, threads):
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
        Ac2dvy_Cython(nx, ny, Model.Dt, Ac2d.vy, Model.Rho, Ac2d.eyy, Ac2d.thetay, Model.Drhoy, Model.Eta1y, Model.Eta2y, threads)

                                

    def Ac2dstress(Ac2d, Model, threads):
        """
        Ac2dstress computes acoustic stress 
        
        Parameters:
            Ac2d : Solver object 
            Model: Model object

        """
        nx = Model.Nx               # Number of grid points in x-direction
        ny = Model.Ny               # Number of grid points in y-direction
        #start_time = time.time()
        Ac2dstress_Cython(nx, ny, Model.Dt, Ac2d.p, Model.Kappa, Ac2d.exx, Ac2d.eyy, Ac2d.gammax, Model.Dkappax, Ac2d.gammay, Model.Dkappay, Model.Alpha1x, Model.Alpha2x, Model.Alpha1y, Model.Alpha2y, threads)
        #end_time = time.time()
        #print("Time taken by the parallel loop:", end_time - start_time, "seconds") 