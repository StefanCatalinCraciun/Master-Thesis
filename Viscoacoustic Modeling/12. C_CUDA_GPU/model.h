#ifndef MODEL_H_
#define MODEL_H_

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES // for C
#include <math.h>
#include "types.h"
#include <cuda.h>

#define MAXWELL 0
#define SLS 1

typedef struct model{
    int Nx,Ny;
    int Nb;
    double W0;
    double** Q;
    double** Kappa;
    double** Dkappax;
    double** Dkappay;
    double** Drhox;
    double** Drhoy;
    double** Rho;
    double** Alpha1x;
    double** Alpha1y;
    double** Alpha2x;
    double** Alpha2y;
    double** Eta1x;
    double** Eta1y;
    double** Eta2x;
    double** Eta2y;
    double* dx;
    double* dy;
    double Dx;
    double Dt;
}Model;


// Methods for the model object
// ModelNew creates a new Model obejct
Model* ModelNew(double **vp, double **rho, double **Q, double Dx, double Dt, double W0, int Nb, int Rheol, int Nx, int Ny);

void ModelDel(Model* model, int Nx);

// ModelStability computes stability index
double ModelStability(Model* model);

#endif /* MODEL_H_ */
