/* Need to use 1D index for accessing array elements */
void saxpy_serial(float* x, int sizex, float* y, int sizey, float a) {

  for (int i=0; i<sizex; i++) {
  	y[i] = a * x[i] + y[i];
  	}
  }

#include <omp.h>
void saxpy_multithread(float* x, int sizex, float* y, int sizey, float a, int threads) {
  int i;
  omp_set_num_threads(threads);
  #pragma omp parallel for default(none) private(i) shared(a, x, y, sizex)
  for (i=0; i<sizex; i++) {
  	y[i] = a * x[i] + y[i];
  	}
  }

