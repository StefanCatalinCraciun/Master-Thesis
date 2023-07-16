#ifndef DIFF_H_
#define DIFF_H_

#include <stdio.h>
#include <stdlib.h>

typedef struct diff{
    int l; // Differentiator length
    int lmax;
    double coeffs[8][8]; 
    double* w; // Differentiator weights   
}Diff;


//Methods
Diff* DiffNew(int l);

void DiffDel(Diff *diff);

void DiffDxminus(Diff* diff, double **A, double **dA, double dx, int nx, int ny);
void DiffDyminus(Diff* diff, double **A, double **dA, double dx, int nx, int ny);
void DiffDxplus(Diff* diff, double **A, double **dA, double dx, int nx, int ny);
void DiffDyplus(Diff* diff, double **A, double **dA, double dx, int nx, int ny);

#endif /* DIFF_H_ */
