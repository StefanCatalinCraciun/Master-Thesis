"""
Model object

This script defines a class for creating a model object that contains the velocity, density and Q models and computes the viscoelastic parameters.

@author: Stefan Catalin Craciun
"""

# Import Libraries 
# ----------------
import sys
import numpy as np
import math

# ----------------
# Model Class
# ----------------
class Model:
    def _init_(self):
        """
        Initializes a new model object with default parameters and attributes set to None.
        
        """
        self.Nx = None
        self.Ny = None
        self.Nb = None
        self.W0 = None
        self.Q = None
        self.Kappa = None
        self.Dkappax = None
        self.Dkappay = None
        self.Drhox = None
        self.Drhoy = None
        self.Rho = None
        self.Alpha1x = None
        self.Alpha1y = None
        self.Alpha2x = None
        self.Alpha2y = None
        self.Eta1x = None
        self.Eta1y = None
        self.Eta2x = None
        self.Eta2y = None
        self.dx = None
        self.dy = None
        self.Dx = None
        self.Dt = None
        
    def ModelNew(self, vp, rho, Q, Dx, Dt, W0, Nb, Rheol):
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
    
        if Rheol == "MAXWELL":
            model = self.Model_maxwell(vp, rho, Q, Dx, Dt, W0, Nb)
        elif Rheol == "SLS":
            model = self.Model_sls(vp, rho, Q, Dx, Dt, W0, Nb)
        else:
            sys.stderr.write("Uknown Q-model\n")
            sys.stderr.flush()
            # Bailing out
            sys.exit()
            
        return model

    def Model_maxwell (self, vp, rho, Q, Dx, Dt, W0, Nb):
        """
        Modelmaxwell creates a new model.
  
        Parameters: 
        - vp :  P-wave velocity model (2D numpy array)
        - rho:  Density (2D numpy array)
        - Q  :  Q-values (2D numpy array)
        - Dx :  Grid interval in x- and y-directions (float)
        - Dt :  Modeling time sampling interval (float)
        - W0 :  Q-model peak angular frequency (float)
        - Nb :  Width of border attenuation zone (in grid points) (int)
  
        Returns:
        - Model structure
  
        ModelNew creates the parameters needed by the Ac2d object
        to perform 2D acoustic modeling.
        The main parameters are density (ρ) and bulk modulus κ which are
        calculated from the wave velocity and density.
        In addition, the visco-elastic coefficients α₁, α₂, η₁ and η₂ are computed.
        
        The model is defined by several 2D arrays, with the x-coordinate
        as the first index, and the y-coordinate as the second index.
        A position in the model (x,y) maps to the arrays as [i,j]
        where x=Dx*i, y=Dx*j;
        
        The absorbing boundaries is comparable to the CPML method
        but constructed using a visco-elastic medium with
        relaxation specified by a Maxwell body, with 
        a time dependent density.
        
        
                          Nx                Outer border        
        |----------------------------------------------|
        |           Qmin=1.1                           |
        |                                              |
        |           Qmax=Q(x,y=Dx*Nb)     Inner border |
        |      ----------------------------------      |
        |      |                                |      |
        |      |                                |      | Ny
        |      |      Q(x,y)                    |      |
        |      |                                |<-Nb->|
        |      |                                |      |
        |      |                                |      |
        |      ----------------------------------      |
        |                                              |
        |                                              |
        |                                              |
        |-----------------------------------------------|

        Fig 1: Organisation of the Q-model.
               The other arrays are organised in the same way.

 
        The Boundary condition is implemented by using a strongly
        absorbing medium in a border zone with width Nb.
        The border zone has the same width both in the horizontal
        and vertical directions.
        The medium in the border zone has a Q-value of Qmax
        at the inner bondary (taken from the Q-model) and
        the Q-value is gradualy reduced to Qmin at the outer boundary.
        
        In the finite-difference solver we use the Maxwell 
        solid to implement time dependent bulk modulus and density.
        The Maxwell solid model uses one parameter, tau0.
        tau0 is related to the Q-value by
        
        taue(Q0) = Q(W0)/W0

        Q0 is here the value for Q at the frequency W0.

        The coeffcients needed by the solver methods in the Ac2d object are
        alpha1x =  exp(d_x/Dt)exp(tau0x),                                  
        alpha2x =  - dx Dt/tau0x
        alpha1y =  exp(d_x/Dt)exp(tau0y),                                  
        alpha2y =  -dx Dt/tau0y
        eta1x   =  exp(d_x/Dt)exp(tau0x),                                  
        eta2x   =  -dx Dt/tau0x
        eta1y   =  exp(d_x/Dt)exp(tau0y),                                  
        eta2y   =  -dx Dt/tau0y
        
        
        tau0 is interpolated between the values given by the Q-value
        Qmax at the inner border of the model and the Qmin at the outer border.
        For the interpolation we just assume that the relaxation times
        varies proportionaly with the square of the distance from
        the inner border, according to:

            tau0x(x) = tau0xmin + (tau0xmax-tau0xmin)*d(x)
            tau0y(x) = tau0ymin + (tau0xmax-tau0ymin)*d(y)

        where:
        
            d(x) = (x/L)^2

        x is the distance from the outer border, while
        L is the length of the border.

        We also have:

            tau0xmax = tau0(Qmax)
            tau0xmin = tau0(Qmin)
            tau0ymax = tau0(Qmax)
            tau0ymin = tau0(Qmin)

        Here Qmin= 1.1, while Qmax is equal to the value
        of Q at the inner border.
        """

        model = Model()                                 # Object to instantiate

        model.Dx = Dx                                   # Grid interval in x- and y-directions
        model.Dt = Dt                                   # Time sampling interval
        
        model.Nx = vp.shape[0]                          # Model dimensions in x- and y-directions
        model.Ny = vp.shape[1]                         
        
        model.Nb = Nb                                   # Width of border attenuation zone (in grid points)
        model.W0 = W0                                   # Q-model peak angular frequency
        
        model.Rho = np.zeros((model.Nx, model.Ny))      # Density
        model.Q = np.zeros((model.Nx, model.Ny))        # Q-values
        model.Kappa = np.zeros((model.Nx, model.Ny))    # Unrelaxed bulk modulus
        
        # The following parameters are the change in the bulk modulus caused by visco-elasticity
        # A separate factor is used for the x- and y-directions due to tapering
        
        model.Dkappax = np.zeros((model.Nx, model.Ny))  # Change in bulk modulus in x-direction
        model.Dkappay = np.zeros((model.Nx, model.Ny))  # Change in bulk modulus in y-direction
        model.Drhox = np.zeros((model.Nx, model.Ny))    # Change in density in x-direction
        model.Drhoy = np.zeros((model.Nx, model.Ny))    # Change in density in y-direction
        

        # Coefficients used for updating memory functions
        
        model.Alpha1x = np.zeros((model.Nx, model.Ny))  
        model.Alpha1y = np.zeros((model.Nx, model.Ny))
        model.Alpha2x = np.zeros((model.Nx, model.Ny))
        model.Alpha2y = np.zeros((model.Nx, model.Ny))
        model.Eta1x = np.zeros((model.Nx, model.Ny))
        model.Eta1y = np.zeros((model.Nx, model.Ny))
        model.Eta2x = np.zeros((model.Nx, model.Ny))
        model.Eta2y = np.zeros((model.Nx, model.Ny))
        
        # Tapering (profile) functions for the x- and y-directions
        model.dx = np.zeros(model.Nx)   
        model.dy = np.zeros(model.Ny)
        
        # Store the model
        for i in range(model.Nx):
            for j in range(model.Ny):
                model.Kappa[i,j] = rho[i,j]*vp[i,j]*vp[i,j]
                model.Rho[i,j] = rho[i,j]
                model.Q[i,j] = Q[i,j]
                
        #Compute 1D profile functions
        self.Modeld(model.dx, model.Dx, model.Nb)
        self.Modeld(model.dy, model.Dx, model.Nb)
        
        ### Compute relaxation times ###
        for i in range(model.Nx):
            for j in range(model.Ny):
                # Compute relaxation times corresponding to Qmax and Qmin
                # Note that we compute the inverse of tau0, and use the same
                # name for the inverse, tau0=1/tau0.
                
                ### Smoothing parameters ###
                Qmin= 1.1                               # Minimum Q-value at the outer boundaries
                tau0min = Qmin / model.W0               # Minimum relaxation time; Taue value corresponding to Qmin      
                tau0min = Qmin / tau0min               
                
                Qmax = model.Q[Nb, j]                   # Maximum Q-value in boundary zone
                tau0max = Qmax / model.W0               # Maximum relaxation time; Taue value corresponding to Qmax
                tau0max = 1.0 / tau0max
                
                # Interpolate tau0 in x-direxction
                tau0x = tau0min + (tau0max - tau0min) * model.dx[i]         # Interpolated relaxation time
                
                Qmax = model.Q[i, Nb]   
                tau0max = Qmax / model.W0
                tau0max = 1.0 / tau0max
                
                # Interpolate tau0 in y-direxction
                tau0y = tau0min + (tau0max - tau0min) * model.dy[j]         # Interpolated relaxation time
                
                
                # In the equations below the relaxation time tau0 is inverse (1/tau0)
                # Compute alpha and eta coefficients
                argx = model.dx[i]                                          # Temp variables     
                argy = model.dy[j]                                          # Temp variables     
                
                # An extra tapering factor of exp(-(x/L)**2)
                # is used to taper some coefficients 
                model.Alpha1x[i, j] = math.exp(-argx) * math.exp(-model.Dt * tau0x) 
                model.Alpha1y[i, j] = math.exp(-argy) * math.exp(-model.Dt * tau0y)
                model.Alpha2x[i, j] = -model.Dt * tau0x
                model.Alpha2y[i, j] = -model.Dt * tau0y
                model.Eta1x[i, j] = math.exp(-argx) * math.exp(-model.Dt * tau0x)
                model.Eta1y[i, j] = math.exp(-argy) * math.exp(-model.Dt * tau0y)
                model.Eta2x[i, j] = -model.Dt * tau0x
                model.Eta2y[i, j] = -model.Dt * tau0y
                
                # For the Maxwell solid Dkappa = kappa and Drho = 1/rho
                # to comply with the solver algorithm
                
                model.Dkappax[i,j]   = model.Kappa[i,j]
                model.Dkappay[i,j]   = model.Kappa[i,j]
                model.Drhox[i,j]     = (1.0/model.Rho[i,j])
                model.Drhoy[i,j]     = (1.0/model.Rho[i,j])
                
                
        return model



    def Model_sls(self, vp, rho, Q, Dx, Dt, W0, Nb):
        """
        Modelsls creates a new model with Standard Linear Solid Q
  
        Parameters: 
        - vp :  P-wave velocity model (2D numpy array)
        - rho:  Density (2D numpy array)
        - Q  :  Q-values (2D numpy array)
        - Dx :  Grid interval in x- and y-directions (float)
        - Dt :  Modeling time sampling interval (float)
        - W0 :  Q-model peak angular frequency (float)
        - Nb :  Width of border attenuation zone (in grid points) (int)
  
        Returns:
        - Model structure
  
        ModelNew creates the parameters needed by the Ac2d object
        to perform 2D acoustic modeling.
        The main parameters are density ρ and bulk modulus κ which are
        calculated from the wave velocity and density.
        In addition, the visco-elastic coefficients α₁, α₂, η₁ and η₂ are computed.
        
        The model is defined by several 2D arrays, with the x-coordinate
        as the first index, and the y-coordinate as the second index.
        A position in the model (x,y) maps to the arrays as [i,j]
        where x=Dx*i, y=Dx*j;
        
        The absorbing boundaries is comparable to the CPML method
        but constructed using a visco-elastic medium with
        relaxation specified by a standard-linear solid, while 
        a time dependent density which uses a standard-linear solid
        relaxation mechanism.
        
        
                          Nx                Outer border        
        |----------------------------------------------|
        |           Qmin=1.1                           |
        |                                              |
        |           Qmax=Q(x,y=Dx*Nb)     Inner border |
        |      ----------------------------------      |
        |      |                                |      |
        |      |                                |      | Ny
        |      |      Q(x,y)                    |      |
        |      |                                |<-Nb->|
        |      |                                |      |
        |      |                                |      |
        |      ----------------------------------      |
        |                                              |
        |                                              |
        |                                              |
        |-----------------------------------------------|

        Fig 1: Organisation of the Q-model.
               The other arrays are organised in the same way.

 
        The Boundary condition is implemented by using a strongly
        absorbing medium in a border zone with width Nb.
        The border zone has the same width both in the horizontal
        and vertical directions.
        The medium in the border zone has a Q-value of Qmax
        at the inner bondary (taken from the Q-model) and
        the Q-value is gradualy reduced to Qmin at the outer boundary.
        
        In the finite-difference solver we use the standard linear 
        solid to implement time dependent bulk modulus and density.
        
        The standard linear solid model uses two parameters,
        tau_sigma (𝜏_𝜎) and tau_epsilon (𝜏_𝜀). These are related to the Q-value by

            taue(Q0) = tau0/Q0 * (√(Q^2_0+1) +1\right)
            taus(Q0) = tau0/Q0 * (√(Q^2_0+1) +1\right)
        
        Q0 is here the value for Q at the frequency W0.


        The coeffcients needed by the solver methods in the Ac2d object are
        alpha1x =  exp(d_x/Dt)exp(tausx),                                  
        alpha2x =  - dx Dt/tauex
        alpha1y =  exp(d_x/Dt)exp(tausy),                                  
        alpha2y =  -dx Dt/tauey
        eta1x   =  exp(d_x/Dt)exp(tausx),                                  
        eta2x   =  -dx Dt/tauex
        eta1y   =  exp(d_x/Dt)exp(tausy),                                  
        eta2y   =  -dx Dt/tauey
        
        Relaxation times are interpolated between the values given by the Q-value 
        Qmax at the inner border of the model and the Qmin at the outer border.
        For the interpolation we just assume that the relaxation times
        varies proportionaly with the square of the distance from
        the inner border, according to
        
            tausx(x) = tausxmin + (tausxmax-tausxmin)*d(x)
            tausy(x) = tausymin + (tausxmax-tausymin)*d(y)
            tauex(x) = tauexmin + (tauexmax-tauexmin)*d(x)
            tauey(x) = taueymin + (tauexmax-taueymin)*d(y)

        where:
        
            d(x) = (x/L)^2

        x is the distance from the outer border, while
        L is the length of the border.

        We also have:

            tausxmax = taus(Qmax)
            tausxmin = taus(Qmin)
            tausymax = taus(Qmax)
            tausymin = taus(Qmin)
            tauexmax = taue(Qmax)
            tauexmin = taue(Qmin)
            taueymax = taue(Qmax)
            taueymin = taue(Qmin)


        Here Qmin= 1.1, while Qmax is equal to the value
        of Q at the inner border.
        """
        
        model = Model()                                 # Object to instantiate

        model.Dx = Dx                                   # Grid interval in x- and y-directions
        model.Dt = Dt                                   # Modeling time sampling interval
        
        model.Nx = vp.shape[0]                          #Model dimensions in x- and y-directions
        model.Ny = vp.shape[1] 
        
        model.Nb = Nb                                   # Width of border attenuation zone (in grid points)
        model.W0 = W0                                   # Q-model peak angular frequency       
        
        model.Rho = np.zeros((model.Nx, model.Ny))      # Density
        model.Q = np.zeros((model.Nx, model.Ny))        # Q-values
        model.Kappa = np.zeros((model.Nx, model.Ny))    # Unrelaxed bulk modulus
        
        # The following parameters are the change in the bulk modulus caused by visco-elasticity
        # A separate factor is used for the x- and y-directions due to tapering
        
        model.Dkappax = np.zeros((model.Nx, model.Ny))  # Change in bulk modulus in x-direction
        model.Dkappay = np.zeros((model.Nx, model.Ny))  # Change in bulk modulus in y-direction
        model.Drhox = np.zeros((model.Nx, model.Ny))    # Change in density in x-direction
        model.Drhoy = np.zeros((model.Nx, model.Ny))    # Change in density in y-direction
        
        # Coefficients used for updating memory functions
        
        model.Alpha1x = np.zeros((model.Nx, model.Ny))
        model.Alpha1y = np.zeros((model.Nx, model.Ny))
        model.Alpha2x = np.zeros((model.Nx, model.Ny))
        model.Alpha2y = np.zeros((model.Nx, model.Ny))
        model.Eta1x = np.zeros((model.Nx, model.Ny))
        model.Eta1y = np.zeros((model.Nx, model.Ny))
        model.Eta2x = np.zeros((model.Nx, model.Ny))
        model.Eta2y = np.zeros((model.Nx, model.Ny))
        
        # Tapering (profile) functions for the x- and y-directions
        model.dx = np.zeros(model.Nx)
        model.dy = np.zeros(model.Ny)
        
        # Store the model
        for i in range(model.Nx):
            for j in range(model.Ny):
                model.Kappa[i,j] = rho[i,j]*vp[i,j]*vp[i,j]
                model.Rho[i,j] = rho[i,j]
                model.Q[i,j] = Q[i,j]
                
                
        #Compute 1D profile functions
        self.Modeld(model.dx, model.Dx, model.Nb)
        self.Modeld(model.dy, model.Dx, model.Nb)
        
        ### Compute relaxation times ###
        for i in range(model.Nx):
            for j in range(model.Ny):
                
                tau0 = 1.0/model.W0                                 # Relaxation time at peak 1/Q-value
                Qmin = 1.1;                                         # MinimumQ-value at the outer boundaries
                
                #Compute relaxation times corresponding to Qmax and Qmin
            
                tauemin = (tau0/Qmin)*(np.sqrt(Qmin*Qmin+1.0)+1.0)  # Taue values corresponding to Qmin and Qmax
                tauemin = 1.0 / tauemin
                tausmin = (tau0/Qmin)*(np.sqrt(Qmin*Qmin+1.0)-1.0)  # Taus values corresponding to Qmin and Qmax
                tausmin = 1.0 / tausmin
                
                Qmax = model.Q[Nb, j]                               # Maximum Q-value in boundary zone
         
                # Note that we compute the inverse of relaxation times, and use the same
                # name for the inverses, taus=1/taus.
                # In all formulas below this section we
                # work with the inverse of the relaxation times.
                
                tauemax = (tau0/Qmin)*(np.sqrt(Qmax*Qmax+1.0)+1.0)  
                tauemax = 1.0/tauemax
                tausmax = (tau0/Qmin)*(np.sqrt(Qmax*Qmax+1.0)-1.0)
                tausmax = 1.0/tausmax
                tauex = tauemin + (tauemax-tauemin)*model.dx[i]
                tausx = tausmin + (tausmax-tausmin)*model.dx[i]
                Qmax  = model.Q[i,Nb]
                tauemax = (tau0/Qmin)*(np.sqrt(Qmax*Qmax+1.0)+1.0)
                tauemax = 1.0/tauemax
                tausmax = (tau0/Qmin)*(np.sqrt(Qmax*Qmax+1.0)-1.0)
                tausmax = 1.0/tausmax
                
                
                # Interpolate relaxation times 
                
                tauey = tauemin + (tauemax-tauemin)*model.dy[j]
                tausy = tausmin + (tausmax-tausmin)*model.dy[j]
            
            
                # In the equations below the relaxation times taue and taus
                # are inverses (1/taue, 1/taus)
                # Compute alpha and eta coefficients
                
                argx = model.dx[i]                                  # Temp variables 
                argy = model.dy[j]                                  # Temp variables 
                
                # An extra tapering factor of exp(-(x/L)**2)
                # is used to taper some coefficients 
                
                model.Alpha1x[i, j] = math.exp(-argx) * math.exp(-model.Dt * tausx) 
                model.Alpha1y[i, j] = math.exp(-argy) * math.exp(-model.Dt * tausy)
                model.Alpha2x[i, j] = model.Dt * tauex
                model.Alpha2y[i, j] = model.Dt * tauey
                model.Eta1x[i, j] = math.exp(-argx) * math.exp(-model.Dt * tausx)
                model.Eta1y[i, j] = math.exp(-argy) * math.exp(-model.Dt * tausy)
                model.Eta2x[i, j] = model.Dt * tauex
                model.Eta2y[i, j] = model.Dt * tauey
                
                # Compute the change in moduli due to
                # visco-ealsticity (is equal to zero for the elastic case)
                
                model.Dkappax[i,j]   = model.Kappa[i,j] * (1.0-tausx/tauex)     
                model.Dkappay[i,j]   = model.Kappa[i,j] * (1.0-tausy/tauey)
                model.Drhox[i,j]     = (1.0/model.Rho[i,j]) * (1.0-tausx/tauex)
                model.Drhoy[i,j]     = (1.0/model.Rho[i,j]) * (1.0-tausy/tauey)
                
        return model

        
    def stability(model):
        """
        Model stability - checks the velocity model, the time step increment and spatial increment for stability.
        
        Parameters:
        - model: Model object
        
        Returns:
        - Stability index (float)
        """
        nx, ny = model.Nx, model.Ny
        
        for i in range(nx):
            for j in range(ny):
                vp = np.sqrt(model.Kappa[i, j] / model.Rho[i, j])
                stab = (vp * model.Dt) / model.Dx
                
            if stab > 1.0 / np.sqrt(2.0):
                print(f"Stability index too large! {stab}")
                
        return stab

    # 1D profile function
    def Modeld(self, d, dx, nb):
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
        n = len(d)

        for i in range(n):
            d[i] = 1.0

        # Taper left border
        if nb != 0:
            for i in range(nb):
                d[i] = d[i] * ((float(i)*dx)/(float(nb)*dx))**2

        # Taper right border 
        if nb != 0:
            for i in range(n-1-nb, n):
                d[i] = d[i] * ((float(n-1-i)*dx)/(float(nb)*dx))**2
        
        return "OK"
        