#ifndef SRC_H_
#define SRC_H_

#include <stdio.h>
#include <stdlib.h>
#define _USE_MATH_DEFINES // for C
#include <math.h>
#include <cuda.h>

typedef struct src{
    double* Src;
    int Sx;
    int Sy;
    int Ns;
}Src;


// SrcNew creates a new source object
Src* SrcNew(double* src, int sx, int sy);

// SrcDel deletes a source object
void SrcDel(Src* src);

void SrcRicker(double* src, double t0, double f0, int nt, double dt);

#endif /* SRC_H_ */
