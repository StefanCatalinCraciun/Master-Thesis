cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void DiffDxminus_Cython(int nx, int ny, int l, double[::1] w, double[:, ::1] A, double[:, ::1] dA, int dx):
    cdef int i, j, k
    cdef float sum

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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void DiffDxplus_Cython(int nx, int ny, int l, double[::1] w, double[:, ::1] A, double[:, ::1] dA, int dx):
    cdef int i, j, k
    cdef float sum

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
                
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void DiffDyminus_Cython(int nx, int ny, int l, double[::1] w, double[:, ::1] A, double[:, ::1] dA, int dx):
    cdef int i, j, k
    cdef float sum

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
                  
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void DiffDyplus_Cython(int nx, int ny, int l, double[::1] w, double[:, ::1] A, double[:, ::1] dA, int dx):
    cdef int i, j, k
    cdef float sum
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

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void Ac2dvx_Cython(int nx, int ny, float Dt, double[:, ::1] vx, double[:, ::1] Rho, double[:, ::1] exx, double[:, ::1] thetax, double[:, ::1] Drhox, double[:, ::1] Eta1x, double[:, ::1] Eta2x):
    cdef int i, j
    cdef float sum

    for i in range(nx):
        for j in range(ny):
            vx[i, j] = Dt * (1.0 / Rho[i, j]) * exx[i, j] + vx[i, j] + Dt * thetax[i, j] * Drhox[i, j] 
                        
            thetax[i, j]  = Eta1x[i, j] * thetax[i, j] + Eta2x[i, j] * exx[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void Ac2dvy_Cython(int nx, int ny, float Dt, double[:, ::1] vy, double[:, ::1] Rho, double[:, ::1] eyy, double[:, ::1] thetay, double[:, ::1] Drhoy, double[:, ::1] Eta1y, double[:, ::1] Eta2y):
    cdef int i, j
    cdef float sum

    for i in range(nx):
        for j in range(ny):
            vy[i, j] = Dt * (1.0 / Rho[i, j]) * eyy[i, j] + vy[i, j] + Dt * thetay[i, j] * Drhoy[i, j]
                        
            thetay[i, j]  = Eta1y[i, j] * thetay[i, j] + Eta2y[i, j] * eyy[i, j]

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef void Ac2dstress_Cython(int nx, int ny, float Dt, double[:, ::1] p, double[:, ::1] Kappa, double[:, ::1] exx, double[:, ::1] eyy, double[:, ::1] gammax, double[:, ::1] Dkappax, double[:, ::1] gammay, double[:, ::1] Dkappay, double[:, ::1] Alpha1x, double[:, ::1] Alpha2x, double[:, ::1] Alpha1y, double[:, ::1] Alpha2y):
    cdef int i, j
    cdef float sum

    for i in range(nx):
        for j in range(ny):
            p[i, j] = Dt * Kappa[i, j] * (exx[i, j] + eyy[i, j]) \
                            + p[i, j] \
                            + Dt * (gammax[i, j] * Dkappax[i, j] + gammay[i, j] * Dkappay[i, j])
                    
            gammax[i, j] = Alpha1x[i, j] * gammax[i, j] + Alpha2x[i, j] * exx[i, j]
            gammay[i, j] = Alpha1y[i, j] * gammay[i, j] + Alpha2y[i, j] * eyy[i, j]
