#################################################################################################################################

#Serial
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline float[::1] saxpy_cython(float alpha, float[::1] x, float[::1] y):
    cdef int i
    cdef int n = x.shape[0]
    for i in range(n):
        y[i] = alpha * x[i] + y[i]
    return y

#Parallel
cimport cython
from cython.parallel import parallel, prange
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline float[::1] saxpy_cython_multithread(float alpha, float[::1] x, float[::1] y, int threads):
    cdef int i
    cdef int n = x.shape[0]
    for i in prange(n, nogil=True, num_threads=threads):
        y[i] = alpha * x[i] + y[i]
    return y