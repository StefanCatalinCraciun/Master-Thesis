#ifndef AC2D_H_
#define AC2D_H_

#include "model.h"
#include "rec.h"
#include "src.h"
#include "types.h"

typedef struct ac2d{
    double** p; // Stress 
    double** vx; // x-component of particle velocity
    double** vy; // y-component of particle velocity
    double** exx; // time derivative of strain x-component
    double** eyy; // time derivative of strain y-component
    double** gammax;
    double** gammay;
    double** thetax;
    double** thetay;
    int ts; // Timestep no
}AC2D;

// Ac2dNew creates a new Ac2d object
AC2D* Ac2dNew(Model* model);

void ac2cDel(AC2D* n, int Nx);

// Ac2dSolve computes the pressure at the next timestep
int Ac2dSolve(AC2D* ac2d, Model* model, Src* src, Rec* rec,int nt, int l);

#endif /* AC2D_H_ */

