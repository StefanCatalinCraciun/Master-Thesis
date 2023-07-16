#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cuda.h>


__global__ void Ac2dvx_kernel(double* vx_flat, double* Rho_flat, double* exx_flat, double* thetax_flat, double* Drhox_flat, double* Eta1x_flat, double* Eta2x_flat, int Nx, int Ny, double Dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Nx && j < Ny) {
        vx_flat[i * Ny + j] = Dt * (1.0 / Rho_flat[i * Ny + j]) * exx_flat[i * Ny + j] + vx_flat[i * Ny + j] + Dt * thetax_flat[i * Ny + j] * Drhox_flat[i * Ny + j];
        thetax_flat[i * Ny + j] = Eta1x_flat[i * Ny + j] * thetax_flat[i * Ny + j] + Eta2x_flat[i * Ny + j] * exx_flat[i * Ny + j];
    }
}

__global__ void Ac2dvy_kernel(double* vy_flat, double* Rho_flat, double* eyy_flat, double* thetay_flat, double* Drhoy_flat, double* Eta1y_flat, double* Eta2y_flat, int Nx, int Ny, double Dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < Nx && j < Ny) {
        vy_flat[i * Ny + j] = Dt * (1.0 / Rho_flat[i * Ny + j]) * eyy_flat[i * Ny + j] + vy_flat[i * Ny + j] + Dt * thetay_flat[i * Ny + j] * Drhoy_flat[i * Ny + j];
        thetay_flat[i * Ny + j] = Eta1y_flat[i * Ny + j] * thetay_flat[i * Ny + j] + Eta2y_flat[i * Ny + j] * eyy_flat[i * Ny + j];
    }
}

__global__ void Ac2dstress_kernel(double* p_flat, double* Kappa_flat, double* exx_flat, double* eyy_flat, double* gammax_flat, double* gammay_flat, double* Dkappax_flat, double* Dkappay_flat, double* Alpha1x_flat, double* Alpha2x_flat, double* Alpha1y_flat, double* Alpha2y_flat, int Nx, int Ny, double Dt) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

    if (i < Nx && j < Ny) {
        p_flat[i * Ny + j] = Dt * Kappa_flat[i * Ny + j] * (exx_flat[i * Ny + j] + eyy_flat[i * Ny + j]) + p_flat[i * Ny + j] + Dt * (gammax_flat[i * Ny + j] * Dkappax_flat[i * Ny + j] + gammay_flat[i * Ny + j] * Dkappay_flat[i * Ny + j]);
        gammax_flat[i * Ny + j] = Alpha1x_flat[i * Ny + j] * gammax_flat[i * Ny + j] + Alpha2x_flat[i * Ny + j] * exx_flat[i * Ny + j];
        gammay_flat[i * Ny + j] = Alpha1y_flat[i * Ny + j] * gammay_flat[i * Ny + j] + Alpha2y_flat[i * Ny + j] * eyy_flat[i * Ny + j];
    }
}


__global__ void DiffDxplus_kernel(double* A, double* dA, double* w, double dx, int Nx, int Ny, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    double sum;
    int k;

    if (i < Nx && j < Ny) {
        // Top border (1 < i < l+1)
        if (i < l) {
            sum = 0.0;
            for (k = 1; k < i + 2; k++) {
                sum = sum - w[k - 1] * A[(i - (k - 1)) * Ny + j];
            }
            for (k = 1; k < l + 1; k++) {
                sum = sum + w[k - 1] * A[(i + k) * Ny + j];
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Between left and right border
        else if (i >= l && i < Nx - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = sum + w[k - 1] * (-A[(i - (k - 1)) * Ny + j] + A[(i + k) * Ny + j]);
            }
            dA[i * Ny + j] = sum / dx;
        }

        
        // Right border
        else {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = sum - w[k - 1] * A[(i - (k - 1)) * Ny + j];
            }
            for (k = 1; k < Nx - i; k++) {
                sum = sum + w[k - 1] * A[(i + k) * Ny + j];
            }
            dA[i * Ny + j] = sum / dx;
        }
    }
}



__global__ void DiffDyplus_kernel(double* A, double* dA, double* w, double dx, int Nx, int Ny, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    double sum;

    if (i < Nx && j < Ny) {
        // Top border (1 < j < l+1)
        if (j < l) {
            sum = 0.0;
            for (k = 1; k < j + 2; k++) {
                sum = -w[k - 1] * A[i * Ny + (j - (k - 1))] + sum;
            }
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * A[i * Ny + (j + k)] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Between top and bottom border
        else if (j >= l && j < Ny - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * (-A[i * Ny + (j - (k - 1))] + A[i * Ny + (j + k)]) + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Bottom border
        else if (j >= Ny - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = -w[k - 1] * A[i * Ny + (j - (k - 1))] + sum;
            }
            for (k = 1; k < Ny - j; k++) {
                sum = w[k - 1] * A[i * Ny + (j + k)] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
    }
}


__global__ void DiffDxminus_kernel(double* A, double* dA, double* w, double dx, int Nx, int Ny, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    double sum;

    if (i < Nx && j < Ny) {
        // Left border (1 < i < l+1)
        if (i < l) {
            sum = 0.0;
            for (k = 1; k < i + 1; k++) {
                sum = -w[k - 1] * A[(i - k) * Ny + j] + sum;
            }
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * A[(i + (k - 1)) * Ny + j] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Outside border area
        else if (i >= l && i < Nx - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * (-A[(i - k) * Ny + j] + A[(i + (k - 1)) * Ny + j]) + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Right border
        else if (i >= Nx - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = -w[k - 1] * A[(i - k) * Ny + j] + sum;
            }
            for (k = 1; k < (Nx - i + 1); k++) {
                sum = w[k - 1] * A[(i + (k - 1)) * Ny + j] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
    }
}


__global__ void DiffDyminus_kernel(double* A, double* dA, double* w, double dx, int Nx, int Ny, int l) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k;
    double sum;

    if (i < Nx && j < Ny) {
        // Top border (1 < j < l+1)
        if (j < l) {
            sum = 0.0;
            for (k = 1; k < j + 1; k++) {
                sum = -w[k - 1] * A[i * Ny + (j - k)] + sum;
            }
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * A[i * Ny + (j + (k - 1))] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Outside border area
        else if (j >= l && j < Ny - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = w[k - 1] * (-A[i * Ny + (j - k)] + A[i * Ny + (j + (k - 1))]) + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
        // Bottom border (Ny - l ≤ j < Ny)
        else if (j >= Ny - l) {
            sum = 0.0;
            for (k = 1; k < l + 1; k++) {
                sum = -w[k - 1] * A[i * Ny + (j - k)] + sum;
            }
            for (k = 1; k < (Ny - j + 1); k++) {
                sum = w[k - 1] * A[i * Ny + (j + (k - 1))] + sum;
            }
            dA[i * Ny + j] = sum / dx;
        }
    }
}


__global__ void add_source_kernel(double* p, double Dt, double* Src, double dx, int src_x, int src_y, int Ny, double* Rho, int i) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx == src_x && idy == src_y) {
        p[src_x * Ny + src_y] = p[src_x * Ny + src_y] + Dt * (Src[i] / (dx * dx * Rho[src_x * Ny + src_y]));
    }
}

