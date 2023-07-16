"""
Receiver object

This script defines a class for creating a receiver object that allows to records seismograms, the presure wavefield
and 1 Trace (wich is also compared to an analytical solution) and saves the results as binary files.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import numpy as np
import matplotlib.pyplot as plt
# Import Classes
# ----------------
from Model import * 
from Src import * 

# ----------------
# Receiver Class
# ----------------
class Rec:
    def __init__(self, rx, ry, nt, resamp, sresamp, nx, ny, dx, Nb, dt, gx=0, gy=0):
        """
        Initializes a new receiver object with the given receiver locations and parameters.
        
        Parameters
        ----------
        rx : integer array
             X coordinates (gridpoints) of the receiver locations.
        ry : integer array
             Y coordinates (gridpoints) of the receiver locations.
        nt : int
             Number of time samples in the receiver data.
        resamp : int
             Resampling rate for the pressure recording (seismogram).
        sresamp : int
             Resampling rate for wavefield animation.
        nx : int
             Number of grid points along the X axis.
        ny : int
             Number of grid points along the Y axis.
        dx : float
             Grid spacing interval in both x and y directions.
        Nb : int
             The number of boundary grid points.
        dt : float
             Time step increment for the simulation.
        gx : int, optional
             X coordinate for just 1 receiver (1 trace), default is 0.
        gy : int, optional
             Y coordinate for just 1 receiver (1 trace), default is 0.
        """
        self.nr = len(rx)       # No of receivers
        self.rx = rx            # Receiver x coordinates (gridpoints)
        self.ry = ry            # Receiver y coordinates (gridpoints)
        self.nt = nt            # No of time samples
        # Pressure p[i,j] time sample at the receiver positions (i) and a given time sample (j) 
        self.p = np.zeros((self.nr, int(nt/resamp)+1), dtype=float)                      
        self.resamp = resamp    # Resampling rate for pressure recording (seismogram)
        self.sresamp = sresamp  # Resampling rate for wavefield animation
        self.pit = 0            # Pressure time sample counter (seismogram)
        self.counter = 0        # Wavefield time sample counter (animation)
        self.gx = gx            # X coordinate for just 1 receiver (1 trace)
        self.gy = gy            # Y coordinate for just 1 receiver (1 trace)
        self.nx = nx            # No of grid points along the X axis
        self.ny = ny            # No of grid points along the Y axis
        self.dx = dx            # Grid spacing interval in both x and y directions
        self.Nb = Nb            # No of boundary grid points (the thickness of the absorbing boundary layer)
        self.dt = dt            # Time step increment for the simulation
        self.wavefield = np.zeros((self.nx, self.ny, int(nt/sresamp)+1)) # A 3D array for storing the wavefield snapshots for animation  
        self.trace = np.zeros(nt) # A 1D array for storing the pressure values at just 1 receiver location (1 trace)
        self.trace_analytical = np.zeros(nt) # A 1D array for storing the pressure values at just 1 receiver location (1 trace) for the analytical solution   

    def rec_trace(self, it, snp):
        """
        Records the pressure values (self.trace) at just 1 receiver location.
        
        Parameters
        ----------
        it : int
            Current time step.
        snp : numpy array
            Pressure values at the current time step.
        """
        self.trace[it] = snp[self.gx,self.gy]
        return "OK"

    def rec_trace_analytical(self, Src, Model):
        """
        Records the pressure values (self.trace_analytical) at just 1 receiver location for the analytical solution.
        
        Parameters
        ----------
        Src : object
            Source object.
        Model : object
            Model object.
        """
        # -------------------
        # Analytical solution
        # -------------------
        time = np.linspace(0 * self.dt, self.nt * self.dt, self.nt)     # Time vector
        G    = time * 0.0                                               # Green's function
        
        # Calculate source-receiver distance
        r = np.sqrt((self.dx*int(self.gx) - self.dx*int(Src.Sx))**2 + (self.dx*int(self.gy) - self.dx*int(Src.Sy))**2)
        
        for a in range(self.nt): # Calculate Green's function (Heaviside function)
            if (time[a] - r / np.sqrt(Model.Kappa[0, 0]/Model.Rho[0,0])) >= 0:
                G[a] = 1. / (2 * np.pi * Model.Rho[0, 0] * np.sqrt(Model.Kappa[0, 0]/Model.Rho[0,0])**2) * (1. / np.sqrt(time[a]**2 - (r/np.sqrt(Model.Kappa[0, 0]/Model.Rho[0,0]))**2))
        Gc   = np.convolve(G, Src.Src * self.dt) # Convolve Green's function with source time function
        Gc   = Gc[0:self.nt]                    
        
        # Calculate analytical solution
        for a in range(1, self.nt - 1):
            self.trace_analytical[a] = (Gc[a+1] - Gc[a-1]) / (2.0 * self.dt)  
        
        return "OK"
    
    
    def rec_seismogram(self, it, snp):
        """
        Records the pressure values (self.p) at the specified receiver locations (seismogram).
        
        Parameters
        ----------
        it : int
            Current time step.
        snp : numpy array
            Pressure values at receiver locations for the current time step.

        Returns
        -------
        str
            "ERR" if the current time step exceeds the total number of time steps or "OK" otherwise.
        """
        if self.pit > self.nt - 1:
            return "ERR"

        if it % self.resamp == 0:
            for pos in range(self.nr):
                ixr = int(self.rx[pos])
                iyr = int(self.ry[pos])
                self.p[pos, self.pit] = snp[iyr, ixr]
            self.pit += 1
       
        return "OK"
    
    def rec_wavefield(self, it, snp):
        """
        Stores wavefield snapshots (self.wavefield) at a given time step if the time step is a multiple of the snapshot resampling rate.
        Records the wavefield.
        
        Parameters
        ----------
        it : int
            Current time step.
        snp : numpy array
            Pressure values at the current time step for wavefield snapshots
            
        Returns
        -------
        str
            "OK" after storing the wavefield snapshot.
        """
        if self.sresamp <= 0:
            return "OK"

        if it % self.sresamp == 0:
            self.wavefield[:,:,self.counter] = snp
            #plt.imshow(self.wavefield[:,:,self.counter], cmap='bwr', vmin=-5e-10, vmax=5e-10)
            #plt.imshow(self.wavefield[:,:,self.counter], cmap='bwr', vmin=-1e-15, vmax=1e-15)
            #plt.show()
            #print(self.counter)
            self.counter += 1

        return "OK"

    def trace_save(self):
        """
        Saves the 1 trace data to a binary file called trace.bin
        in the Visualization folder.
        
        Returns
        -------
        str
            "OK" after successfully saving the trace data.
        """
        output_file = "Visualization/trace.bin"             # Output file name and path
        #print(self.trace.shape) 
        
        # Save the parameters of the seismogram as metadata  nt, dt
        metadata = np.array(self.trace.shape, dtype=int)    # Save the number of time samples as metadata (nt)
        metadata = np.append(metadata, self.dt)             # Append the time step increment to the metadata array
        #print(metadata)
        
        with open(output_file, "wb") as f:                  # Open the output file in binary mode
            metadata.tofile(f)                              # Save the metadata
            # Save the wavefield data
            self.trace.tofile(f)                            # Save the trace data
            print(f"Trace data saved to {output_file}")
        return "OK"
    
    def trace_analytical_save(self):
        """
        Saves the analytical solution data to a binary file called trace_analytical.bin
        in the Visualization folder.
        
        Returns
        -------
        str
            "OK" after successfully saving the analytical solution data.
        """
        output_file = "Visualization/trace_analytical.bin"              # Output file name and path
        #print(self.trace_analytical.shape) 
        
        # Save the parameters of the seismogram as metadata  nt, dt
        metadata = np.array(self.trace_analytical.shape, dtype=int)     # Save the number of time samples as metadata (nt)
        metadata = np.append(metadata, self.dt)                         # Append the time step increment to the metadata array
        #print(metadata)
        
        with open(output_file, "wb") as f:                              # Open the output file in binary mode
            metadata.tofile(f)                                          # Save the metadata
            # Save the wavefield data
            self.trace_analytical.tofile(f)                             # Save the trace data
            print(f"Analytical solution data saved to {output_file}")
        return "OK"

    def seismogram_save(self):
        """
        Saves the seismogram data to a binary file called seismogram.bin
        in the Visualization folder.
        
        Returns
        -------
        str
            "OK" after successfully saving the seismogram data.
        """
        output_file = "Visualization/seismogram.bin"    # Output file name and path
        #print(self.p.shape)                             
        # Save the parameters of the seismogram as metadata  nr, nt, dt, dx, Nb
        metadata = np.array(self.p.shape, dtype=int)    # Convert the shape of the seismogram to an array of integers (nr, nt)
        metadata = np.append(metadata, self.dt)         # Append the time step increment to the metadata array
        metadata = np.append(metadata, self.dx)         # Append the grid spacing interval to the metadata array
        metadata = np.append(metadata, self.Nb)         # Append the number of boundary grid points to the metadata array
        #print("seismogram metdadat: ",metadata)
        
        with open(output_file, "wb") as f:              # Open the output file in binary mode
            metadata.tofile(f)                          # Save the metadata
            # Save the wavefield data
            self.p.tofile(f)                            # Save the seismogram data
            print(f"Seismogram data saved to {output_file}")
        return "OK"
   
    def wavefield_save(self):
        """
        Saves the wavefield data to a binary file called wavefield.bin
        in the Visualization folder.
        
        Returns
        -------
        str
            "OK" after successfully saving the wavefield data.
        """
        output_file = "Visualization/wavefield.bin"             # Output file name and path
        
        # Save the shape of the wavefield as metadata  nx, ny, nt, dx, Nb
        metadata = np.array(self.wavefield.shape, dtype=int)    # Convert the shape of the wavefield to an array of integers (nx, ny, nt)
        metadata = np.append(metadata, self.dx)                 # Append the grid spacing interval to the metadata array
        metadata = np.append(metadata, self.Nb)                 # Append the number of boundary grid points to the metadata array
        #print(metadata)
   
        with open(output_file, "wb") as f:                      # Open the output file in binary mode
            metadata.tofile(f)                                  # Save the metadata
            # Save the wavefield data
            self.wavefield.tofile(f)                            # Save the wavefield data
        print(f"Wavefield data saved to {output_file}")
        return "OK"
