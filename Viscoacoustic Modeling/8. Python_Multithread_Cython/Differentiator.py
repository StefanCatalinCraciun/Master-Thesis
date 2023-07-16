"""
Differentiator object

This script defines a Differentiator class for computing derivatives of 2D arrays (grids)
with respect to x and y directions. The class provides four functions for computing forward
and backward derivatives in the x and y directions.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import numpy as np
import time
import sys
sys.path.insert(1,'.')
from Cython_Functions import DiffDxminus_Cython, DiffDyminus_Cython, DiffDxplus_Cython, DiffDyplus_Cython



# ----------------
# Differentiator Class
# ----------------
class Differentiator:
    """
    This class is used to create instances of a differentiator 
    with a specified number of coefficients (l).
    """
    def __init__(self, l):
        """
        Constructor for the Differentiator class,, which creates a new differentiator object.
        The constructor takes one argument 'l', which represents the number of coefficients.

        Parameters:
        l: int
            The length of the differentiator.
        """
        
        # Set the maximum allowed number of coefficients (lmax) to 8
        self.lmax = 8

        # Limit the input value 'l' between 1 and lmax to prevent issues with array indexing
        if l < 1:
            l = 1
        if l > self.lmax:
            l = self.lmax
        
        # Store the validated value of 'l' as an instance variable
        self.l = l
        
        # Initialize a 2D array (matrix) 'coeffs' with dimensions lmax x lmax
        # and fill it with zeros to store the coefficients
        self.coeffs = np.zeros((self.lmax, self.lmax), dtype=float)
        
        # Initialize a 1D array 'w' with length l and filled with zeros
        # to store the computed weights based on the input value 'l'
        self.w = np.zeros(l, dtype=float)   # Differentiator weights 

        # Define the coefficient values for each value of 'l' from 1 to 8
        # Note that the coefficients are hard-coded for simplicity
        
        # Load coefficients
        # l=1
        self.coeffs[0, 0] = 1.0021

        # l=2
        self.coeffs[1, 0] = 1.1452
        self.coeffs[1, 1] = -0.0492

        # l=3
        self.coeffs[2, 0] = 1.2036
        self.coeffs[2, 1] = -0.0833
        self.coeffs[2, 2] = 0.0097
        
        # l=4
        self.coeffs[3, 0] = 1.2316     
        self.coeffs[3, 1] = -0.1041     
        self.coeffs[3, 2] = 0.0206     
        self.coeffs[3, 3] = -0.0035

        # l=5
        self.coeffs[4, 0] = 1.2463
        self.coeffs[4, 1] = -0.1163
        self.coeffs[4, 2] = 0.0290
        self.coeffs[4, 3] = -0.0080
        self.coeffs[4, 4] = 0.0018

        # l=6
        self.coeffs[5, 0] = 1.2542
        self.coeffs[5, 1] = -0.1213
        self.coeffs[5, 2] = 0.0344
        self.coeffs[5, 3] = -0.0170
        self.coeffs[5, 4] = 0.0038
        self.coeffs[5, 5] = -0.0011

        # l=7
        self.coeffs[6, 0] = 1.2593
        self.coeffs[6, 1] = -0.1280
        self.coeffs[6, 2] = 0.0384
        self.coeffs[6, 3] = -0.0147
        self.coeffs[6, 4] = 0.0059
        self.coeffs[6, 5] = -0.0022
        self.coeffs[6, 6] = 0.0007

        # l=8
        self.coeffs[7, 0] = 1.2626
        self.coeffs[7, 1] = -0.1312
        self.coeffs[7, 2] = 0.0412
        self.coeffs[7, 3] = -0.0170
        self.coeffs[7, 4] = 0.0076
        self.coeffs[7, 5] = -0.0034
        self.coeffs[7, 6] = 0.0014
        self.coeffs[7, 7] = -0.0005
        

        # Populate the 'w' array with the corresponding coefficients
        for k in range(l):
            self.w[k] = self.coeffs[l - 1, k]
            

    def DiffDxminus(Diff, A, dA, dx, threads):
        """
        Computes the backward derivative in the x-direction for a given 2D array A.
        
        Parameters:
        Diff        : A Diff object containing the weights w and the length of the differentiator l.
        A           : A 2D array (float) representing the input data.
        dA          : A 2D array (float) representing the output data (computed derivatives).
        dx          : A float representing the sampling interval.
        
        The function calculates the derivative at each point in the input 2D array A and stores the result
        in the output 2D array dA. The derivative is computed using the following formula:
            
        dA[i,j] = (1/dx) * sum_{k=1}^l w[k] (A[i+(k-1)dx,j] - A[(i-kdx,j])
            
        The code handles three cases: left border, outside border area, and right border of the array A.
        
        w[k] is the weights vector and l is the length of the differentiator.
        """
        #t1 = time.perf_counter()

        nx, ny = A.shape                        # Get the dimensions of the input array A

        # Left border (1 < i < l + 1)
        l = Diff.l                              # Get the length of the differentiator
        w = Diff.w                              # Get the weights vector

        DiffDxminus_Cython(nx, ny, l, w, A, dA, dx, threads)
        '''
        for i in range(0, l):
            for j in range(0, ny):
                # Calculate the weighted sum for left border elements
                sum = 0.0
                for k in range(1, i + 1):
                    sum -= w[k - 1] * A[i - k, j]
                for k in range(1, l + 1):
                    sum += w[k - 1] * A[i + (k - 1), j]
                dA[i, j] = sum / dx

        # Outside border area (l <= i < nx - l)
        for i in range(l, nx - l):
            for j in range(0, ny):
                # Calculate the weighted sum for elements outside the border area
                sum = 0.0
                for k in range(1, l + 1):
                    sum += w[k - 1] * (-A[i - k, j] + A[i + (k - 1), j])
                dA[i, j] = sum / dx
                    
        # Right border (nx - l <= i < nx)
        for i in range(nx - l, nx):
            for j in range(0, ny):
                # Calculate the weighted sum for right border elements
                sum = 0.0
                for k in range(1, l + 1):
                    sum -= w[k - 1] * A[i - k, j]
                for k in range(1, nx - i + 1):
                    sum += w[k - 1] * A[i + (k - 1), j]
                dA[i, j] = sum / dx
                
        '''

        #t2 = time.perf_counter()
        #print("Solver wall clock time: ", t2 - t1)

    def DiffDxplus(Diff, A, dA, dx, threads):
        """
        Computes the forward derivative in the x-direction for a given 2D array A.
        
        Parameters:
        Diff        : A Diff object containing the weights w and the length of the differentiator l.
        A           : A 2D array (float) representing the input data.
        dA          : A 2D array (float) representing the output data (computed derivatives).
        dx          : A float representing the sampling interval.
        
        The function calculates the derivative at each point in the input 2D array A and stores the result
        in the output 2D array dA. The derivative is computed using the following formula:
                
        dA[i,j] = (1/dx) * sum_{k=1}^l w[k] (A[i+kdx,j] - A[(i-(k-1)dx,j])
        
        The code handles three cases: left border, between left and right border, and right border of the array A.
        
        w[k] is the weights vector and l is the length of the differentiator.
        """
        nx, ny = A.shape                    # Get the dimensions of the input array A
        
        l = Diff.l                          # Get the length of the differentiator
        w = Diff.w                          # Get the weights vector
        
        DiffDxplus_Cython(nx, ny, l, w, A, dA, dx, threads)
        '''
        # Left border (1 < i < l + 1)
        for i in range(0, l):
            for j in range(0, ny):
                sum = 0.0
                for k in range(1, i + 2):
                    sum -= w[k - 1] * A[i - (k - 1), j]
                for k in range(1, l + 1):
                    sum += w[k - 1] * A[i + k, j]
                    
                dA[i, j] = sum / dx

        # Between left and right border
        for i in range(l, nx - l):
            for j in range(0, ny):
                sum = 0.0
                for k in range(1, l + 1):
                    sum += w[k - 1] * (-A[i - (k - 1), j] + A[i + k, j])
                dA[i, j] = sum / dx

        # Right border
        for i in range(nx - l, nx):
            for j in range(0, ny):
                sum = 0.0
                for k in range(1, l + 1):
                    sum -= w[k - 1] * A[i - (k - 1), j]
                for k in range(1, nx - i):
                    sum += w[k - 1] * A[i + k, j]
                
                dA[i, j] = sum / dx
        '''
                
    def DiffDyminus(Diff, A, dA, dx, threads):
        """
        Computes the backward derivative in the y-direction for a given 2D array A.
        
        Parameters:
        Diff            : A Diff object containing the weights w and the length of the differentiator l.
        A               : A 2D array (float) representing the input data.
        dA              : A 2D array (float) representing the output data (computed derivative).
        dx              : A float representing the sampling interval.
        
        The function calculates the derivative at each point in the input 2D array A and stores the result
        in the output 2D array dA. The derivative is computed using the following formula:
            
        dA[i,j] = (1/dx) * sum_{k=1}^l w[k] (A[i,j+(k-1)dx] - A[i,j-kdx])
            
        The code handles three cases: top border, outside border area, and bottom border of the array A.
        
        w[k] is the weights vector and l is the length of the differentiator.
        """
        nx, ny = A.shape                    # Get the dimensions of the input array A
        
        l = Diff.l                          # Get the length of the differentiator
        w = Diff.w                          # Get the weights vector
        
        DiffDyminus_Cython(nx, ny, l, w, A, dA, dx, threads)
        '''
        # Top border (1 < i < l + 1)
        for i in range(0, nx):
            for j in range(0, l):
                sum = 0.0
                for k in range(1, j + 1):
                    sum -= w[k - 1] * A[i, j - k]
                for k in range(1, l + 1):
                    sum += w[k - 1] * A[i, j + (k - 1)]
                dA[i, j] = sum / dx
                
        # Outside border area
        for i in range(0, nx):
            for j in range(l, ny - l):
                sum = 0.0
                for k in range(1, l + 1):
                    sum += w[k - 1] * (-A[i, j - k] + A[i, j + (k - 1)])
                dA[i, j] = sum / dx
                
        # Bottom border
        for i in range(0, nx):
            for j in range(ny - l, ny):
                sum = 0.0
                for k in range(1, l + 1):
                    sum -= w[k - 1] * A[i, j - k]
                for k in range(1, ny - j + 1):
                    sum += w[k - 1] * A[i, j + (k - 1)]
                dA[i, j] = sum / dx
        '''

    def DiffDyplus(Diff, A, dA, dx, threads):
        """
        Computes the forward derivative in the y-direction for a given 2D array A.
        
        Parameters:
        Diff            : A Diff object containing the weights w and the length of the differentiator l.
        A               : A 2D array (float) representing the input data.
        dA              : A 2D array (float) representing the output data (computed derivative).
        dx              : A float representing the sampling interval.
        
        The function calculates the derivative at each point in the input 2D array A and stores the result
        in the output 2D array dA. The derivative is computed using the following formula:
            
        dA[i,j] = (1/dx) * sum_{k=1}^l w[k] (A[i,j+kdx] - A[i,j-(k-1)dx])
            
        The code handles three cases: top border, outside border area, and bottom border of the array A.
        w[k] is the weights vector and l is the length of the differentiator.
        """
        nx, ny = A.shape                # Get the dimensions of the input array A
        
        l = Diff.l                      # Get the length of the differentiator
        w = Diff.w                      # Get the weights vector
        
        DiffDyplus_Cython(nx, ny, l, w, A, dA, dx, threads)
        '''
        # Top border (1 < j < l + 1)
        for i in range(0, nx):
            for j in range(0, l):
                sum = 0.0
                for k in range(1, j + 2):
                    sum -= w[k - 1] * A[i, j - (k - 1)]
                for k in range(1, l + 1):
                    sum += w[k - 1] * A[i, j + k]
                dA[i, j] = sum / dx
                
        # Outside border area
        for i in range(0, nx):
            for j in range(l, ny - l):
                sum = 0.0
                for k in range(1, l + 1):
                    sum += w[k - 1] * (-A[i, j - (k - 1)] + A[i, j + k])
                dA[i, j] = sum / dx
                
        # Bottom border
        for i in range(0, nx):
            for j in range(ny - l, ny):
                sum = 0.0
                for k in range(1, l + 1):
                    sum -= w[k - 1] * A[i, j - (k - 1)]
                for k in range(1, ny - j):
                    sum += w[k - 1] * A[i, j + k]
                dA[i, j] = sum / dx
        '''