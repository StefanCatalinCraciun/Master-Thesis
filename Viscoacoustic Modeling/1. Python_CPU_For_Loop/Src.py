"""
Source object

This script defines a class for creating a source object with a given source and spatial coordinates.
It also includes a method for generating a Ricker wavelet.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import numpy as np

# ----------------
# Source Class
# ----------------
class Src:
    def __init__(self, source, sx, sy):
        '''
        Initializes a new source object with given number of sources and spatial coordinates.

        Parameters:
        source: array-like
                Source values (Wavelet shape).
        sx: array-like
            X coordinates of the source.
        sy: array-like
            Y coordinates of the source.
        '''
        self.Src = np.array(source)     # Source values (Wavelet shape)
        self.Sx = np.array(sx)          # X coordinates of the source
        self.Sy = np.array(sy)          # Y coordinates of the source
        self.Ns = len(sx)               # Number of sources (use just 1 source)
        
    @staticmethod
    def SrcRicker(src, t0, f0, nt, dt):
        '''
        Creates a Ricker wavelet.

        Parameters:
        t0: float
            Time shift of the Ricker wavelet.
        f0: float
            Central frequency of the Ricker wavelet.
        nt: int
            Number of time samples.
        dt: float
            Time step.

        Returns:
        src: numpy array
            Ricker wavelet values.
        '''
        t = np.arange(0, nt) * dt - t0      # Time axis values
        w0 = 2.0 * np.pi * f0               # Omega (angular frequency)
        arg = w0 * t                       
        src = (1.0 - 0.5 * arg**2) * np.exp(-0.25 * arg**2) # Ricker wavelet formula
        
        return src

    def __del__(self):
        '''
        Deletes a source object.
        '''
        pass