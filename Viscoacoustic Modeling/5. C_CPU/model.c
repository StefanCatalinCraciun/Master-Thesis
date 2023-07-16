#include "model.h"

//Internal functions
Model* Modelmaxwell(double** vp, double** rho, double** Q, double Dx, double Dt, double W0, int Nb, int Nx, int Ny); // Maxwell Q-model
Model* Modelsls(double** vp, double** rho, double** Q, double Dx, double Dt, double W0, int Nb, int Nx, int Ny); // Sls Q-model
int Modeld(double* d, double dx, int nb, int n);// 1D profile function


Model *ModelNew(double **vp, double **rho, double **Q, double Dx, double Dt, double W0, int Nb, int Rheol, int Nx, int Ny){
    
    Model* m = NULL;

    if(Rheol == MAXWELL){
        m = Modelmaxwell(vp, rho, Q, Dx, Dt, W0, Nb, Nx, Ny); 
    } else if(Rheol == SLS){
        m= Modelsls(vp, rho, Q, Dx, Dt, W0, Nb, Nx, Ny);
    } else{
        fprintf (stderr, "Uknown Q-model\n");
        return NULL;
    } 
    
    return(m);
}


void ModelDel(Model* model, int Nx) {
    int i;
    for (i = 0; i < Nx; i++) {
        free(model->Q[i]);
        free(model->Kappa[i]);
        free(model->Rho[i]);
        
        free(model->Dkappax[i]);
        free(model->Dkappay[i]);
        free(model->Drhox[i]);
        free(model->Drhoy[i]);
        free(model->Alpha1x[i]);
        free(model->Alpha1y[i]);
        free(model->Alpha2x[i]);
        free(model->Alpha2y[i]);
        free(model->Eta1x[i]);
        free(model->Eta1y[i]);
        free(model->Eta2x[i]);
        free(model->Eta2y[i]);
    }
    /*free(model->Nx)
    free(model->Ny)*/
    free(model->dx);
    free(model->dy);

    free(model->Dkappax);
    free(model->Dkappay);
    free(model->Drhox);
    free(model->Drhoy);
    free(model->Alpha1x);
    free(model->Alpha1y);
    free(model->Alpha2x);
    free(model->Alpha2y);
    free(model->Eta1x);
    free(model->Eta1y);
    free(model->Eta2x);
    free(model->Eta2y);

    free(model->Q);
    free(model->Kappa);
    free(model->Rho);
    free(model);

    return;
}


double ModelStability(Model *model){
    
    int i,j;
    double vp,stab;

    for(i=0; i<model->Nx; i=i+1){
        for(j=0; j<model->Ny; j=j+1){
            vp = sqrt(model->Kappa[i][j]/model->Rho[i][j]);
            stab = (vp*model->Dt)/model->Dx;
            if(stab > 1.0/sqrt(2.0)){
                fprintf(stderr,"Stability index too large! %f\n", stab);
            }
        }
    }

    return stab;
}


Model *Modelmaxwell(double **vp, double **rho, double **Q, double Dx, double Dt, double W0, int Nb, int Nx, int Ny){   
    
    int i, j;
    

    // Smoothing parameters
    double Qmin, Qmax;       // Minimum and Maximum Q-values in boundary zone
    double tau0min,tau0max;  // Taue values corresponding to Qmin and Qmax
    double tau0x, tau0y;     // Relaxation times
    double argx, argy;            // Temp variables

    // Allocate memory for the Model struct
    Model* model = (Model*) malloc(sizeof(Model));
    

    // Set the parameters for the Model struct
    model->Dx = Dx; 
    model->Dt = Dt;
    model->W0 = W0;
    model->Nb = Nb;
    model->Nx = Nx;
    model->Ny = Ny;
    
    // Allocate memory for the arrays in the Model struct
    model->dx = (double*) malloc(model->Nx * sizeof(double));
    model->dy = (double*) malloc(model->Ny * sizeof(double));

    // Allocate memory for the double** pointers inside the Model struct
    model->Q = (double**)malloc(model->Nx * sizeof(double*));
    model->Kappa = (double**)malloc(model->Nx * sizeof(double*));
    model->Rho = (double**)malloc(model->Nx * sizeof(double*));

    model->Dkappax = (double**) malloc(model->Nx * sizeof(double*));
    model->Dkappay = (double**) malloc(model->Nx * sizeof(double*));
    model->Drhox = (double**) malloc(model->Nx * sizeof(double*));
    model->Drhoy = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha1x = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha1y = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha2x = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha2y = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta1x = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta1y = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta2x = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta2y = (double**) malloc(model->Nx * sizeof(double*));
    
    // Allocate memory for the data arrays and assign them to the double** pointers
    for (i = 0; i < model->Nx; i++) {
        model->Q[i] = (double*)malloc(model->Ny * sizeof(double));
        model->Kappa[i] = (double*)malloc(model->Ny * sizeof(double));
        model->Rho[i] = (double*)malloc(model->Ny * sizeof(double));

        model->Dkappax[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Dkappay[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Drhox[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Drhoy[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha1x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha1y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha2x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha2y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta1x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta1y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta2x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta2y[i] = (double*) malloc(model->Ny * sizeof(double));

        for (j = 0; j < model->Ny; j++) {
            model->Kappa[i][j] =  rho[i][j]*pow(vp[i][j], 2);
            model->Q[i][j] = Q[i][j];
            model->Rho[i][j] = rho[i][j];
        }
    }
    
    Modeld(model->dx, model->Dx, model->Nb, model->Nx);
    Modeld(model->dy, model->Dx, model->Nb, model->Ny);
    
    for(i=0; i<model->Nx;i=i+1){
        for(j=0; j<model->Ny;j=j+1){

            // Compute relaxation times corresponding to Qmax and Qmin
            // Note that we compute the inverse of tau0, and use the same name for the inverse, tau0=1/tau0.
            Qmin = 1.1;  // MinimumQ-value at the outer boundaries:    
            tau0min = Qmin/model->W0;
            tau0min = 1.0/tau0min;
            Qmax  = model->Q[Nb][j];
            tau0max = Qmax/model->W0;
            tau0max = 1.0/tau0max;
            // Interpolate tau0 in x-direxction
            tau0x = tau0min + (tau0max-tau0min)*model->dx[i];

            Qmax  = model->Q[i][Nb];
            tau0max = Qmax/model->W0;
            tau0max = 1.0/tau0max;

            // Interpolate tau0 in y-direxction
            tau0y = tau0min + (tau0max-tau0min)*model->dy[j];


            // In the equations below the relaxation time tau0 
            // is inverse (1/tau0)
            // Compute alpha and eta coefficients
            argx = model->dx[i];
            argy = model->dy[j];
            // An extra tapering factor of exp(-(x/L)**2)
            // is used to taper some coefficeints 
            model->Alpha1x[i][j]   = exp(-argx)*exp(-model->Dt*tau0x);
            model->Alpha1y[i][j]   = exp(-argy)*exp(-model->Dt*tau0y);
            model->Alpha2x[i][j]   = -model->Dt*tau0x;
            model->Alpha2y[i][j]   = -model->Dt*tau0y;
            model->Eta1x[i][j]     = exp(-argx)*exp(-model->Dt*tau0x);
            model->Eta1y[i][j]     = exp(-argy)*exp(-model->Dt*tau0y);
            model->Eta2x[i][j]     = -model->Dt*tau0x;
            model->Eta2y[i][j]     = -model->Dt*tau0y;
        
            // For the Maxwell solid Dkappa = kappa and Drho = 1/rho
            // to comply with the solver algorithm i ac2d.e
            model->Dkappax[i][j]   = model->Kappa[i][j];
            model->Dkappay[i][j]   = model->Kappa[i][j];
            model->Drhox[i][j]     = (1.0/model->Rho[i][j]);
            model->Drhoy[i][j]     = (1.0/model->Rho[i][j]);
        }
    } 
    
    return model;
}


Model *Modelsls(double **vp, double **rho, double **Q, double Dx, double Dt, double W0, int Nb, int Nx, int Ny){
        
    int i, j;

    // Smoothing parameters
    double Qmin, Qmax;       // Minimum and Maximum Q-values in boundary zone
    double argx, argy;            // Temp variables
    
    double tau0;         // Relaxation time at Peak 1/Q-value
    double tauemin,tauemax;  // Taue values corresponding to Qmin and Qmax
    double tausmin,tausmax;  // Taus values corresponding to Qmin and Qmax

    // Relaxation times
    double tausx, tausy;     
    double tauex, tauey;


    // Allocate memory for the Model struct
    Model* model = (Model*) malloc(sizeof(Model));

    // Set the parameters for the Model struct
    model->Dx = Dx; 
    model->Dt = Dt;
    model->W0 = W0;
    model->Nb = Nb;
    model->Nx = Nx;
    model->Ny = Ny;
    
    // Allocate memory for the arrays in the Model struct
    model->dx = (double*) malloc(model->Nx * sizeof(double));
    model->dy = (double*) malloc(model->Ny * sizeof(double));

    // Allocate memory for the double** pointers inside the Model struct
    model->Q = (double**)malloc(model->Nx * sizeof(double*));
    model->Kappa = (double**)malloc(model->Nx * sizeof(double*));
    model->Rho = (double**)malloc(model->Nx * sizeof(double*));

    model->Dkappax = (double**) malloc(model->Nx * sizeof(double*));
    model->Dkappay = (double**) malloc(model->Nx * sizeof(double*));
    model->Drhox = (double**) malloc(model->Nx * sizeof(double*));
    model->Drhoy = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha1x = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha1y = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha2x = (double**) malloc(model->Nx * sizeof(double*));
    model->Alpha2y = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta1x = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta1y = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta2x = (double**) malloc(model->Nx * sizeof(double*));
    model->Eta2y = (double**) malloc(model->Nx * sizeof(double*));

    // Allocate memory for the data arrays and assign them to the double** pointers
    for (i = 0; i < model->Nx; i++) {
        model->Q[i] = (double*)malloc(model->Ny * sizeof(double));
        model->Kappa[i] = (double*)malloc(model->Ny * sizeof(double));
        model->Rho[i] = (double*)malloc(model->Ny * sizeof(double));

        model->Dkappax[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Dkappay[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Drhox[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Drhoy[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha1x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha1y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha2x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Alpha2y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta1x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta1y[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta2x[i] = (double*) malloc(model->Ny * sizeof(double));
        model->Eta2y[i] = (double*) malloc(model->Ny * sizeof(double));

        for (j = 0; j < model->Ny; j++) {
            model->Kappa[i][j] =  rho[i][j]*pow(vp[i][j], 2);
            model->Q[i][j] = Q[i][j];
            model->Rho[i][j] = rho[i][j];
        }
    }
    
    //Compute 1D profile functions
    Modeld(model->dx, model->Dx, model->Nb, model->Nx);
    Modeld(model->dy, model->Dx, model->Nb, model->Ny);

    // Compute relaxation times
    for(i=0; i<model->Nx;i=i+1){
        for(j=0; j<model->Ny;j=j+1){
            tau0 = 1.0/model->W0;   // Relaxation time corresponding to absorption top
            Qmin = 1.1;            // MinimumQ-value at the outer boundaries

            // Compute relaxation times corresponding to Qmax and Qmin
            tauemin = (tau0/Qmin)*(sqrt(Qmin*Qmin+1.0)+1.0);
            tauemin = 1.0/tauemin;
            tausmin = (tau0/Qmin)*(sqrt(Qmin*Qmin+1.0)-1.0);
            tausmin = 1.0/tausmin;

            Qmax  = model->Q[Nb][j];
            // Note that we compute the inverse
            // of relaxation times, and use the same
            // name for the inverses, taus=1/taus.
            // In all formulas below this section we
            // work with the inverse of the relaxation times.
            tauemax = (tau0/Qmin)*(sqrt(Qmax*Qmax+1.0)+1.0);
            tauemax = 1.0/tauemax;
            tausmax = (tau0/Qmin)*(sqrt(Qmax*Qmax+1.0)-1.0);
            tausmax = 1.0/tausmax;
            tauex = tauemin + (tauemax-tauemin)*model->dx[i];
            tausx = tausmin + (tausmax-tausmin)*model->dx[i];
            Qmax  = model->Q[i][Nb];
            tauemax = (tau0/Qmin)*(sqrt(Qmax*Qmax+1.0)+1.0);
            tauemax = 1.0/tauemax;
            tausmax = (tau0/Qmin)*(sqrt(Qmax*Qmax+1.0)-1.0);
            tausmax = 1.0/tausmax;

            // Interpolate relaxation times 
            tauey = tauemin + (tauemax-tauemin)*model->dy[j];
            tausy = tausmin + (tausmax-tausmin)*model->dy[j];

            // In the equations below the relaxation times taue and taus
            // are inverses (1/taue, 1/taus)
            // Compute alpha and eta coefficients
            argx = model->dx[i];
            argy = model->dy[j];
            // An extra tapering factor of exp(-(x/L)**2)
            // is used to taper some coefficeints 
            model->Alpha1x[i][j]   = exp(-argx)*exp(-model->Dt*tausx);
            model->Alpha1y[i][j]   = exp(-argy)*exp(-model->Dt*tausy);
            model->Alpha2x[i][j]   = model->Dt*tauex;
            model->Alpha2y[i][j]   = model->Dt*tauey;
            model->Eta1x[i][j]     = exp(-argx)*exp(-model->Dt*tausx);
            model->Eta1y[i][j]     = exp(-argy)*exp(-model->Dt*tausy);
            model->Eta2x[i][j]     = model->Dt*tauex;
            model->Eta2y[i][j]     = model->Dt*tauey;
        
            // Compute the change in moduli due to
            // visco-ealsticity (is equal to zero for the elastic case)
            model->Dkappax[i][j]   = model->Kappa[i][j]*(1.0-tausx/tauex);
            model->Dkappay[i][j]   = model->Kappa[i][j]*(1.0-tausy/tauey);
            model->Drhox[i][j]     = (1.0/model->Rho[i][j])*(1.0-tausx/tauex);
            model->Drhoy[i][j]     = (1.0/model->Rho[i][j])*(1.0-tausy/tauey);
        }
    }
    return model;
}


int Modeld(double *d, double dx, int nb, int n){
    
    int i;
    
    for(i=0; i<n; i=i+1){
        d[i]=1.0;
    }
    
    // Taper left border
    for(i=0; i<nb;i=i+1){
        d[i] = d[i]*(((double)i*dx)/((double)nb*dx)
                    *((double)i)*dx/((double)nb*dx));
    }

    // taper right border
    for(i=n-1-nb; i<n;i=i+1){
        d[i] = d[i]*(((double)n-1-i)*dx)/((double)nb*dx)
                    *(((double)n-1-i)*dx)/((double)nb*dx);
    }

    return(OK);
}
