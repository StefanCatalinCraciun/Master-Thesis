#ifndef REC_H_
#define REC_H_

#include <stdio.h>
#include <cuda.h>

typedef struct rec{
    int nr; // No of receivers
    int *rx; // Receiver x-postions
    int *ry; // Receiver y-postions 
    int nt; // No of time samples
    double **p; // Pressure p[i,j] time sample no j at position no i 
    int resamp; // Resample factor for receivers
    int sresamp; // Resample factor for snapshots
    int pit; // Next time sample to be recorded

    int counter;
    int nx;
    int ny;
    double dx;
    int Nb;
    double dt;
    double ***wavefield;
}Rec;


// RecNew is the constructor for receivers.
// The return value is a Rec object
Rec* RecNew(int* rx, int* ry, int nt, int resamp, int sresamp, int nx, int ny, double dx, int Nb, double dt, int Nr);

// rec_wavefield records the wavefield
void rec_wavefield(Rec* rec, int it, double **snp);

// wavedfield_save saves the wavefield to a bin file
void save_wavefield(Rec* rec);

// RecReciver records data at the receivers
int RecReceiver(Rec* rec,int it, double** p);

// Recsave stores receiver recording on file
int RecSave(Rec* rec);

// RecSnap records snapshots
int RecSnap(Rec* rec,int it, double** snp);

#endif /* REC_H_ */

