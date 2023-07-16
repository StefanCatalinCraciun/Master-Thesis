#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES // for C
#include <math.h>

#include <time.h> // Added to use the clock() function
#include <cuda.h>

#include "AC2D.h"
#include "model.h"
#include "rec.h"
#include "src.h"

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec / 1000000000.0;
}


int main(){
    
    // FD solver
    AC2D* ac2d = NULL;  
    
    // Model
    Model* model = NULL; 
    int Nx=101, Ny=101; // Model dimension in x- and y-directions.
    int Nb=15; // Border for PML attenuation
    double dt=0.0005, dx=10; // Time sampling and space sampling intervals
    double f0=25.0, t0=0.04; // Peak frequency, Pulse delay
    int l=8; // Operator length

    double W0=f0*3.1415*2.0; // Central angular frequency
    double **vp = NULL, **rho = NULL, **Q = NULL;
    double vp0=2000, rho0=2000, q0=10e5;  
    int rheol = 0;
    
    
    // Variable for receiver 
    Rec* rec = NULL;   
    int nt=1000; // No of time steps
    int resamp=1, sresamp=5; // Resampling factors for data and snapshot; Output receiver sampling, Output snapshot resampling  
    
    // Variables for source
    Src* src = NULL;   
    double ricker_wavelet[nt]; // Source pulse
    int sx, sy; // Source x,y-coordinates
    int Nr=100; // Number of receivers, number of receivers X
    int rx[Nr], ry[Nr];     // Receiver x,y-coordinates
    
    // Variable declaration for main
    int i, j; // Timestep no
    //char* tmp; // Temporary workspace
    
    double si;
    
    vp = (double**)malloc(Nx * sizeof(double*));
    rho = (double**)malloc(Nx * sizeof(double*));
    Q = (double**)malloc(Nx * sizeof(double*));
    for (int i = 0; i < Nx; i++) {
        vp[i] = (double*)malloc(Ny * sizeof(double));
        rho[i] = (double*)malloc(Ny * sizeof(double));
        Q[i] = (double*)malloc(Ny * sizeof(double));
        
        for(j=0; j < Ny; j++){
            /*if (j < Ny/2) {
                vp[i][j] = 2000;
                rho[i][j] = 2000;
                Q[i][j] = 10e5;
            }
            else {
                vp[i][j] = 3000;
                rho[i][j] = 3000;
                Q[i][j] = 10e5;
            }
            */
            vp[i][j] = vp0; /* Read the velocity model */
            rho[i][j] = rho0; /* Read the density model */
            Q[i][j] = q0; /* Read the attenuation model */
        }
    }
    
    /* Create a source */
    sx= Nx/2;
    sy= Ny/2;
    SrcRicker(ricker_wavelet, t0, f0, nt, dt);
    src=SrcNew(ricker_wavelet,sx,sy);

    // Print ricker_wavelet
    /*printf("Ricker wavelet:\n");
    for (int i = 0; i < nt; i++) {
        printf("%f ", ricker_wavelet[i]);
    }
    printf("\n");*/
    
    /* Create a model */
    rheol = MAXWELL;
    
    model = ModelNew(vp,rho,Q,dx,dt,W0,Nb, rheol, Nx, Ny);
    

    si = ModelStability(model);

    printf("Stability index: %f\n", si);
    

    /* Create a receiver */
    for(i=0; i<Nr; i=i+1){
        rx[i] = 200;
        ry[i] = i;
    }
    
    
    rec=RecNew(rx,ry,nt,resamp,sresamp,Nx,Ny,dx,Nb,dt,Nr);
    
    /* Create solver */
    ac2d = Ac2dNew(model);

    // Add timer before Ac2dSolve
    double start_time = get_time();

    /* Run solver */
    Ac2dSolve(ac2d, model, src, rec, nt,l);

    // Add timer after Ac2dSolve and print the time
    cudaDeviceSynchronize();
    double end_time = get_time();

    double elapsed= end_time - start_time;
    printf("Solver wall clock time: %f seconds\n", elapsed);


    /* Save recording */
    save_wavefield(rec);
    printf("OK : \n");

    /* Free memory */
    // Free memory when you're done
    for (int i = 0; i < Nx; i++) {
        free(vp[i]);
        free(rho[i]);
        free(Q[i]);
    }
    free(vp);
    free(rho);
    free(Q);
    SrcDel(src);
    ac2dDel(ac2d, Nx);
    ModelDel(model, Nx);
    return OK;
}

