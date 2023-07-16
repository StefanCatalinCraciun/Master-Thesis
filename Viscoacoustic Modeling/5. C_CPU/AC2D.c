#include "AC2D.h"
#include "diff.h"


// Internal functions
void Ac2dvx(AC2D* ac2d, Model* model);
void Ac2dvy(AC2D* ac2d, Model* model);
void Ac2dstress(AC2D* ac2d, Model* model);


AC2D* Ac2dNew(Model* model){
    
    int i, j;

    AC2D* ac2dini=NULL;
    ac2dini = (AC2D*)malloc(sizeof(AC2D));
    if (!ac2dini){
        fprintf (stderr, "%s\n", strerror(errno));
        return NULL;
    }

    ac2dini->p = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->vx = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->vy = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->exx = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->eyy = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->gammax = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->gammay = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->thetay = (double**) malloc(model->Nx * sizeof(double*));
    ac2dini->thetax = (double**) malloc(model->Nx * sizeof(double*));

    ac2dini->ts = 0;

    for (i=0; i < model->Nx; i++){ 
        ac2dini->p[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->vx[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->vy[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->exx[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->eyy[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->gammax[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->gammay[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->thetax[i] = (double*) malloc(model->Ny * sizeof(double));
        ac2dini->thetay[i] = (double*) malloc(model->Ny * sizeof(double));
        for (j=0; j < model->Ny; j++){ 
            ac2dini->p[i][j] = 0;
            ac2dini->vx[i][j] = 0;
            ac2dini->vy[i][j] = 0;
            ac2dini->exx[i][j] = 0;
            ac2dini->eyy[i][j] = 0;
            ac2dini->gammax[i][j] = 0;
            ac2dini->gammay[i][j] = 0;
            ac2dini->thetax[i][j] = 0;
            ac2dini->thetay[i][j] = 0;
        }
    }

    return ac2dini;
}


void ac2cDel(AC2D* ac2d, int Nx){

  int i;
    for (i = 0; i < Nx; i++) {
        free(ac2d->p[i]);
        free(ac2d->vx[i]);
        free(ac2d->vy[i]);
        free(ac2d->exx[i]);
        free(ac2d->eyy[i]);
        free(ac2d->gammax[i]);
        free(ac2d->gammay[i]);
        free(ac2d->thetax[i]);
        free(ac2d->thetay[i]);
    }        
    free(ac2d->p);
    free(ac2d->vx);
    free(ac2d->vy);
    free(ac2d->exx);
    free(ac2d->eyy);
    free(ac2d->gammax);
    free(ac2d->gammay);
    free(ac2d->thetax);
    free(ac2d->thetay);

    free(ac2d);
}


int Ac2dSolve(AC2D* ac2d, Model* model, Src* src, Rec* rec,int nt, int l){

    int ns,ne; // Start stop timesteps
    int i;
    double perc, oldperc; // Percentage finished current and old
    int iperc; // Percentage finished

    
    Diff* diff = NULL; // Differentiator object
    diff = DiffNew(l); // Create differentiator object
    oldperc = 0.0;
    ns = ac2d->ts; //Get current timestep 
    ne = ns + nt;        

    
    for(i=ns; i < ne; i++){
        // Compute spatial derivative of stress
        // Use exx and eyy as temp storage
        DiffDxplus(diff, ac2d->p, ac2d->exx, model->Dx, model->Nx,model->Ny); // Forward differentiation x-axis
        Ac2dvx(ac2d, model); // Compute vx

        DiffDyplus(diff, ac2d->p, ac2d->eyy, model->Dx, model->Nx,model->Ny); // Forward differentiation y-axis
        Ac2dvy(ac2d, model); // Compute vy

        // Compute time derivative of strains
        DiffDxminus(diff, ac2d->vx, ac2d->exx, model->Dx, model->Nx,model->Ny); //Compute exx     
        DiffDyminus(diff, ac2d->vy, ac2d->eyy, model->Dx, model->Nx,model->Ny); //Compute eyy   
         
        // Update stress
        Ac2dstress(ac2d,model);  
        // Add source
        ac2d->p[src->Sx][src->Sy] = ac2d->p[src->Sx][src->Sy] + model->Dt*(src->Src[i]/(model->Dx*model->Dx*model->Rho[src->Sx][src->Sy])); 

        // Print progress
        perc=1000.0* (double)i / (double)(ne-ns-1);
        if(perc-oldperc >= 10.0){
            iperc=(int)perc/10;
            if(iperc%10 == 0){printf("%d\n", iperc);}
            oldperc=perc;
        }
        //Record wavefield
        rec_wavefield(rec,i, ac2d->p);
        
        
    }
    
    DiffDel(diff); ////////////////////////////HERE?????????????????????????????????

    return OK;
}


void Ac2dvx(AC2D *ac2d, Model *model){

    int i,j;
    
    // The derivative of stress in x-direction is stored in exx
    // Scale with inverse density and advance one time step
    for(i=0; i < model->Nx; i++){
        for(j=0; j < model->Ny; j++){
            ac2d->vx[i][j] = model->Dt*(1.0/model->Rho[i][j])*ac2d->exx[i][j] + ac2d->vx[i][j] + model->Dt*ac2d->thetax[i][j]*model->Drhox[i][j];
            ac2d->thetax[i][j]  = model->Eta1x[i][j]*ac2d->thetax[i][j] + model->Eta2x[i][j]*ac2d->exx[i][j];
        }
    }
}


void Ac2dvy(AC2D *ac2d, Model *model){
  
    int i,j;
    
    // The derivative of stress in y-direction is stored in eyy
    // Scale with inverse density and advance one time step
    for(i=0; i < model->Nx; i++){
        for(j=0; j < model->Ny; j++){
            ac2d->vy[i][j] = model->Dt*(1.0/model->Rho[i][j])*ac2d->eyy[i][j] + ac2d->vy[i][j] + model->Dt*ac2d->thetay[i][j]*model->Drhoy[i][j];
            ac2d->thetay[i][j]  = model->Eta1y[i][j]*ac2d->thetay[i][j] + model->Eta2y[i][j]*ac2d->eyy[i][j];
        }
    }
}


void Ac2dstress(AC2D *ac2d, Model *model){
  
    int i,j;
  
    for(i=0; i < model->Nx; i++){
        for(j=0; j < model->Ny; j++){
            ac2d->p[i][j] = model->Dt*model->Kappa[i][j]*(ac2d->exx[i][j]+ac2d->eyy[i][j]) + ac2d->p[i][j] + model->Dt*(ac2d->gammax[i][j]*model->Dkappax[i][j] +ac2d->gammay[i][j]*model->Dkappay[i][j]);
            ac2d->gammax[i][j] = model->Alpha1x[i][j]*ac2d->gammax[i][j] + model->Alpha2x[i][j]*ac2d->exx[i][j];
            ac2d->gammay[i][j] = model->Alpha1y[i][j]*ac2d->gammay[i][j] + model->Alpha2y[i][j]*ac2d->eyy[i][j];
        }
  }
}
