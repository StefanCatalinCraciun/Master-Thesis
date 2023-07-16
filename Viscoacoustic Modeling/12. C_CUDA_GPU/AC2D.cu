#include "AC2D.h"
#include "diff.h"

#include "CUDA_kernels.cu"



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


void ac2dDel(AC2D* ac2d, int Nx){

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

    // CUDA STARTS HERE

    // 1. Flatten the 2D array to a 1D array
    int Nx = model->Nx;
    int Ny = model->Ny;
    int size = Nx * Ny;
    double* w_flat = (double*) malloc(l * sizeof(double));
    for (int y = 0; y < l; y++) {
        w_flat[y] = diff->w[y]; 
    }

    double* Src_flat = (double*) malloc(nt * sizeof(double));
    for (int z = 0; z < nt; z++) {
        Src_flat[z] = src->Src[z];
    }

    double* p_flat = (double*) malloc(size * sizeof(double));
    double* vx_flat = (double*) malloc(size * sizeof(double));
    double* vy_flat = (double*) malloc(size * sizeof(double));
    double* exx_flat = (double*) malloc(size * sizeof(double));
    double* eyy_flat = (double*) malloc(size * sizeof(double));
    double* gammax_flat = (double*) malloc(size * sizeof(double));
    double* gammay_flat = (double*) malloc(size * sizeof(double));
    double* thetax_flat = (double*) malloc(size * sizeof(double));
    double* thetay_flat = (double*) malloc(size * sizeof(double));
    double* Rho_flat = (double*) malloc(size * sizeof(double));
    double* Kappa_flat = (double*) malloc(size * sizeof(double));
    double* Drhox_flat = (double*) malloc(size * sizeof(double));
    double* Drhoy_flat = (double*) malloc(size * sizeof(double));
    double* Eta1x_flat = (double*) malloc(size * sizeof(double));
    double* Eta2x_flat = (double*) malloc(size * sizeof(double));
    double* Eta1y_flat = (double*) malloc(size * sizeof(double));
    double* Eta2y_flat = (double*) malloc(size * sizeof(double));
    double* Dkappax_flat = (double*) malloc(size * sizeof(double));
    double* Dkappay_flat = (double*) malloc(size * sizeof(double));
    double* Alpha1x_flat = (double*) malloc(size * sizeof(double));
    double* Alpha2x_flat = (double*) malloc(size * sizeof(double));
    double* Alpha1y_flat = (double*) malloc(size * sizeof(double));
    double* Alpha2y_flat = (double*) malloc(size * sizeof(double));

    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            p_flat[i * Ny + j] = ac2d->p[i][j];
            vx_flat[i * Ny + j] = ac2d->vx[i][j];
            vy_flat[i * Ny + j] = ac2d->vy[i][j];
            exx_flat[i * Ny + j] = ac2d->exx[i][j];
            eyy_flat[i * Ny + j] = ac2d->eyy[i][j];
            gammax_flat[i * Ny + j] = ac2d->gammax[i][j];
            gammay_flat[i * Ny + j] = ac2d->gammay[i][j];
            thetax_flat[i * Ny + j] = ac2d->thetax[i][j];
            thetay_flat[i * Ny + j] = ac2d->thetay[i][j];
            Rho_flat[i * Ny + j] = model->Rho[i][j];
            Kappa_flat[i * Ny + j] = model->Kappa[i][j];
            Drhox_flat[i * Ny + j] = model->Drhox[i][j];
            Drhoy_flat[i * Ny + j] = model->Drhoy[i][j];
            Eta1x_flat[i * Ny + j] = model->Eta1x[i][j];
            Eta2x_flat[i * Ny + j] = model->Eta2x[i][j];
            Eta1y_flat[i * Ny + j] = model->Eta1y[i][j];
            Eta2y_flat[i * Ny + j] = model->Eta2y[i][j];
            Dkappax_flat[i * Ny + j] = model->Dkappax[i][j];
            Dkappay_flat[i * Ny + j] = model->Dkappay[i][j];
            Alpha1x_flat[i * Ny + j] = model->Alpha1x[i][j];
            Alpha2x_flat[i * Ny + j] = model->Alpha2x[i][j];
            Alpha1y_flat[i * Ny + j] = model->Alpha1y[i][j];
            Alpha2y_flat[i * Ny + j] = model->Alpha2y[i][j];
        }
    }

    // 2. Define the CUDA arrays 
    double* p_gpu, * vx_gpu, * vy_gpu, * exx_gpu, * eyy_gpu, * gammax_gpu, * gammay_gpu, * thetax_gpu, * thetay_gpu; // For ac2d
    double* Rho_gpu, * Kappa_gpu, * Drhox_gpu, * Drhoy_gpu, * Eta1x_gpu, * Eta2x_gpu, * Eta1y_gpu, * Eta2y_gpu, * Dkappax_gpu, * Dkappay_gpu, * Alpha1x_gpu, * Alpha2x_gpu, * Alpha1y_gpu, * Alpha2y_gpu; // For model
    double* w_gpu; // For diff
    double* Src_gpu; // For src

    // 3. Allocate memory on the GPU for the flattened array
    cudaMalloc((void**)&p_gpu, size * sizeof(double));
    cudaMalloc((void**)&vx_gpu, size * sizeof(double));
    cudaMalloc((void**)&vy_gpu, size * sizeof(double));
    cudaMalloc((void**)&exx_gpu, size * sizeof(double));
    cudaMalloc((void**)&eyy_gpu, size * sizeof(double));
    cudaMalloc((void**)&gammax_gpu, size * sizeof(double));
    cudaMalloc((void**)&gammay_gpu, size * sizeof(double));
    cudaMalloc((void**)&thetax_gpu, size * sizeof(double));
    cudaMalloc((void**)&thetay_gpu, size * sizeof(double));
    cudaMalloc((void**)&Rho_gpu, size * sizeof(double));
    cudaMalloc((void**)&Kappa_gpu, size * sizeof(double));
    cudaMalloc((void**)&Drhox_gpu, size * sizeof(double));
    cudaMalloc((void**)&Drhoy_gpu, size * sizeof(double));
    cudaMalloc((void**)&Eta1x_gpu, size * sizeof(double));
    cudaMalloc((void**)&Eta2x_gpu, size * sizeof(double));
    cudaMalloc((void**)&Eta1y_gpu, size * sizeof(double));
    cudaMalloc((void**)&Eta2y_gpu, size * sizeof(double));
    cudaMalloc((void**)&Dkappax_gpu, size * sizeof(double));
    cudaMalloc((void**)&Dkappay_gpu, size * sizeof(double));
    cudaMalloc((void**)&Alpha1x_gpu, size * sizeof(double));
    cudaMalloc((void**)&Alpha2x_gpu, size * sizeof(double));
    cudaMalloc((void**)&Alpha1y_gpu, size * sizeof(double));
    cudaMalloc((void**)&Alpha2y_gpu, size * sizeof(double));
    cudaMalloc((void**)&w_gpu, l * sizeof(double));
    cudaMalloc((void**)&Src_gpu, nt * sizeof(double));

    // 4. Copy the flattened array from the host to the GPU
    cudaMemcpy(p_gpu, p_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vx_gpu, vx_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(vy_gpu, vy_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(exx_gpu, exx_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(eyy_gpu, eyy_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gammax_gpu, gammax_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(gammay_gpu, gammay_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(thetax_gpu, thetax_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(thetay_gpu, thetay_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Rho_gpu, Rho_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Kappa_gpu, Kappa_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Drhox_gpu, Drhox_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Drhoy_gpu, Drhoy_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Eta1x_gpu, Eta1x_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Eta2x_gpu, Eta2x_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Eta1y_gpu, Eta1y_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Eta2y_gpu, Eta2y_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Dkappax_gpu, Dkappax_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Dkappay_gpu, Dkappay_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha1x_gpu, Alpha1x_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha2x_gpu, Alpha2x_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha1y_gpu, Alpha1y_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Alpha2y_gpu, Alpha2y_flat, size * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(w_gpu, w_flat, l * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(Src_gpu, Src_flat, nt * sizeof(double), cudaMemcpyHostToDevice);

    // 4. Perform necessary operations on the GPU
    
    //int blockSize = 1024;
    //int gridSize = (size + blockSize - 1) / blockSize;

    // Determine the block and grid size for a 2D problem
    dim3 blockSize(16, 16); // Adjust these values based on your GPU architecture
    dim3 gridSize((Nx + blockSize.x - 1) / blockSize.x, (Ny + blockSize.y - 1) / blockSize.y);


    for(i=ns; i < ne; i++){

        // Compute spatial derivative of stress
        // Use exx and eyy as temp storage

        // Launch the kernel
        DiffDxplus_kernel<<<gridSize, blockSize>>>(p_gpu, exx_gpu, w_gpu, model->Dx, Nx, Ny, l); // Forward differentiation x-axis
        cudaError_t err1 = cudaDeviceSynchronize();
        if (err1 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err1));
            return 1;
        }
        
        // Launch the kernel
        Ac2dvx_kernel<<<gridSize, blockSize>>>(vx_gpu, Rho_gpu, exx_gpu, thetax_gpu, Drhox_gpu, Eta1x_gpu, Eta2x_gpu, Nx, Ny, model->Dt); // Compute vx
        cudaError_t err2 = cudaDeviceSynchronize();
        if (err2 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err2));
            return 1;
        }

        // Launch the kernel
        DiffDyplus_kernel<<<gridSize, blockSize>>>(p_gpu, eyy_gpu, w_gpu, model->Dx, Nx, Ny, l); // Forward differentiation y-axis
        cudaError_t err3 = cudaDeviceSynchronize();
        if (err3 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err3));
            return 1;
        }

        Ac2dvy_kernel<<<gridSize, blockSize>>>(vy_gpu, Rho_gpu, eyy_gpu, thetay_gpu, Drhoy_gpu, Eta1y_gpu, Eta2y_gpu, Nx, Ny, model->Dt); // Compute vy
        cudaError_t err4 = cudaDeviceSynchronize();
        if (err4 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err4));
            return 1;
        }

        // Compute time derivative of strains
        // Launch the kernel
        DiffDxminus_kernel<<<gridSize, blockSize>>>(vx_gpu, exx_gpu, w_gpu, model->Dx, Nx, Ny, l); //Compute exx
        cudaError_t err5 = cudaDeviceSynchronize();
        if (err5 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err5));
            return 1;
        }

        // Launch the kernel     
        DiffDyminus_kernel<<<gridSize, blockSize>>>(vy_gpu, eyy_gpu, w_gpu, model->Dx, Nx, Ny, l); //Compute eyy 
        cudaError_t err6 = cudaDeviceSynchronize();
        if (err6 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err6));
            return 1;
        }

        // Update stress
        // Launch the kernel
        Ac2dstress_kernel<<<gridSize, blockSize>>>(p_gpu, Kappa_gpu, exx_gpu, eyy_gpu, gammax_gpu, gammay_gpu, Dkappax_gpu, Dkappay_gpu, Alpha1x_gpu, Alpha2x_gpu, Alpha1y_gpu, Alpha2y_gpu, Nx, Ny, model->Dt);
        cudaError_t err7 = cudaDeviceSynchronize();
        if (err7 != cudaSuccess) {
            printf("Error synchronizing the device: %s\n", cudaGetErrorString(err7));
            return 1;
        }
        
        // Add source
        add_source_kernel<<<gridSize, blockSize>>>(p_gpu, model->Dt, Src_gpu, model->Dx, src->Sx, src->Sy, Ny, Rho_gpu, i);
        
        // Print progress
        perc=1000.0* (double)i / (double)(ne-ns-1);
        if(perc-oldperc >= 10.0){
            iperc=(int)perc/10;
            if(iperc%10 == 0){printf("%d\n", iperc);}
            oldperc=perc;
        }

        
        // 6. Copy the result back from the GPU to the host
        cudaMemcpy(p_flat, p_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);


        for (int a = 0; a < Nx; a++) {
            for (int b = 0; b < Ny; b++) {
                ac2d->p[a][b] = p_flat[a * Ny + b];
            }
        }
        
        rec_wavefield(rec,i, ac2d->p);
        
        
    }
   
 
    // 5. Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        exit(1);
    }
    
    
    // 6. Copy the result back from the GPU to the host
    cudaMemcpy(p_flat, p_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vx_flat, vx_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(vy_flat, vy_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(exx_flat, exx_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(eyy_flat, eyy_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gammax_flat, gammax_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(gammay_flat, gammay_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(thetax_flat, thetax_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(thetay_flat, thetay_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Rho_flat, Rho_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Kappa_flat, Kappa_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Drhox_flat, Drhox_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Drhoy_flat, Drhoy_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Eta1x_flat, Eta1x_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Eta2x_flat, Eta2x_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Eta1y_flat, Eta1y_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Eta2y_flat, Eta2y_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Dkappax_flat, Dkappax_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Dkappay_flat, Dkappay_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Alpha1x_flat, Alpha1x_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Alpha2x_flat, Alpha2x_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Alpha1y_flat, Alpha1y_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(Alpha2y_flat, Alpha2y_gpu, size * sizeof(double), cudaMemcpyDeviceToHost);

    // 7. Convert the flattened array back to a 2D array
    for (int i = 0; i < Nx; i++) {
        for (int j = 0; j < Ny; j++) {
            ac2d->p[i][j] = p_flat[i * Ny + j];
            ac2d->vx[i][j] = vx_flat[i * Ny + j];
            ac2d->vy[i][j] = vy_flat[i * Ny + j];
            ac2d->exx[i][j] = exx_flat[i * Ny + j];
            ac2d->eyy[i][j] = eyy_flat[i * Ny + j];
            ac2d->gammax[i][j] = gammax_flat[i * Ny + j];
            ac2d->gammay[i][j] = gammay_flat[i * Ny + j];
            ac2d->thetax[i][j] = thetax_flat[i * Ny + j];
            ac2d->thetay[i][j] = thetay_flat[i * Ny + j];
            model->Rho[i][j] = Rho_flat[i * Ny + j];
            model->Kappa[i][j] = Kappa_flat[i * Ny + j];
            model->Drhox[i][j] = Drhox_flat[i * Ny + j];
            model->Drhoy[i][j] = Drhoy_flat[i * Ny + j];
            model->Eta1x[i][j] = Eta1x_flat[i * Ny + j];
            model->Eta2x[i][j] = Eta2x_flat[i * Ny + j];
            model->Eta1y[i][j] = Eta1y_flat[i * Ny + j];
            model->Eta2y[i][j] = Eta2y_flat[i * Ny + j];
            model->Dkappax[i][j] = Dkappax_flat[i * Ny + j];
            model->Dkappay[i][j] = Dkappay_flat[i * Ny + j];
            model->Alpha1x[i][j] = Alpha1x_flat[i * Ny + j];
            model->Alpha2x[i][j] = Alpha2x_flat[i * Ny + j];
            model->Alpha1y[i][j] = Alpha1y_flat[i * Ny + j];
            model->Alpha2y[i][j] = Alpha2y_flat[i * Ny + j];
        }
    }

    // Free the allocated memory
    free(p_flat);
    free(vx_flat);
    free(vy_flat);
    free(exx_flat);
    free(eyy_flat);
    free(gammax_flat);
    free(gammay_flat);
    free(thetax_flat);
    free(thetay_flat);
    free(Rho_flat);
    free(Kappa_flat);
    free(Drhox_flat);
    free(Drhoy_flat);
    free(Eta1x_flat);
    free(Eta2x_flat);
    free(Eta1y_flat);
    free(Eta2y_flat);
    free(Dkappax_flat);
    free(Dkappay_flat);
    free(Alpha1x_flat);
    free(Alpha2x_flat);
    free(Alpha1y_flat);
    free(Alpha2y_flat);
    free(w_flat);
    free(Src_flat);

    cudaFree(p_gpu);
    cudaFree(vx_gpu);
    cudaFree(vy_gpu);
    cudaFree(exx_gpu);
    cudaFree(eyy_gpu);
    cudaFree(gammax_gpu);
    cudaFree(gammay_gpu);
    cudaFree(thetax_gpu);
    cudaFree(thetay_gpu);
    cudaFree(Rho_gpu);
    cudaFree(Kappa_gpu);
    cudaFree(Drhox_gpu);
    cudaFree(Drhoy_gpu);
    cudaFree(Eta1x_gpu);
    cudaFree(Eta2x_gpu);
    cudaFree(Eta1y_gpu);
    cudaFree(Eta2y_gpu);
    cudaFree(Dkappax_gpu);
    cudaFree(Dkappay_gpu);
    cudaFree(Alpha1x_gpu);
    cudaFree(Alpha2x_gpu);
    cudaFree(Alpha1y_gpu);
    cudaFree(Alpha2y_gpu);
    cudaFree(w_gpu);
    cudaFree(Src_gpu);
    

    return OK;
}
