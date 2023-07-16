
def factorial (num):
    if num<= 1:
        return 1;
    return num*factorial(num-1)

cpdef long cp_factorial (long num):
    if num<= 1:
        return 1;
    return num*cp_factorial(num-1)

cpdef inline long cp_factorial_v2 (long num):
    if num<= 1:
        return 1;
    return num*cp_factorial_v2(num-1)



##################################################################################################################################

#Serial
cimport cython
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline float[::1] saxpy_cython(float alpha, float[::1] x, float[::1] y, float[::1] z):
    cdef int i
    cdef int n = x.shape[0]
    for i in range(n):
        z[i] = alpha * x[i] + y[i]
    return z

#Parallel
cimport cython
from cython.parallel import parallel, prange
@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
cpdef inline float[::1] saxpy_cython_multithread(float alpha, float[::1] x, float[::1] y, float[::1] z, int threads):
    cdef int i
    cdef int n = x.shape[0]
    for i in prange(n, nogil=True, num_threads=threads):
        z[i] = alpha * x[i] + y[i]
    return z