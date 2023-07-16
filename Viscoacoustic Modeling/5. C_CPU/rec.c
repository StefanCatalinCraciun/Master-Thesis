#include "rec.h"
#include <stdlib.h>


Rec *RecNew(int* rx, int* ry, int nt, int resamp, int sresamp, int nx, int ny, double dx, int Nb, double dt, int Nr){
    Rec* rec = (Rec*)malloc(sizeof(Rec));
    rec->nr = Nr;
    rec->rx = rx;
    rec->ry = ry;
    rec->nt = nt;
    rec->resamp = resamp;
    rec->sresamp = sresamp;
    rec->pit = 0;
    rec->counter = 0;
    rec->nx = nx;
    rec->ny = ny;
    rec->dx = dx;
    rec->Nb = Nb;
    rec->dt = dt;
    rec->wavefield = (double***)malloc(nx*sizeof(double**));
    for (int i = 0; i < nx; i++)
    {
        rec->wavefield[i] = (double**)malloc(ny*sizeof(double*));
        for (int j = 0; j < ny; j++)
        {
            rec->wavefield[i][j] = (double*)malloc(nt*sizeof(double));
        }
    }
    return rec;
}


void rec_wavefield(Rec *rec, int it, double **snp)
{
    /*
     * Stores wavefield snapshots at a given time step if the time step is a multiple of the snapshot resampling rate.
     * Records the wavefield.
     */
    if (rec->resamp <= 0) 
        return;

    if (it % rec->sresamp == 0)
    {
        for (int i = 0; i < rec->nx; i++) {
            for (int j = 0; j < rec->ny; j++) {
                rec->wavefield[i][j][rec->counter] = snp[i][j];
            }
        }
        rec->counter += 1;
    }

}

void save_wavefield(Rec* rec)
{   
    const char *output_file = "Visualization/wavefield.bin";
 
    // Save the shape of the wavefield as metadata
    int metadata[5] = {rec->nx, rec->ny, rec->nt, (int)rec->dx, rec->Nb};
    
    printf("Metadata: ");
    for (int i = 0; i < 5; i++)
    {
        printf("%d ", metadata[i]);
    }
    printf("\n");


    FILE *f = fopen(output_file, "wb");
    if (f == NULL)
    {
        printf("Error: Unable to open the file %s for writing.\n", output_file);
        return;
    }

    // Write metadata
    fwrite(metadata, sizeof(int), 5, f);

    // Save the wavefield data
    for (int i = 0; i < rec->nx; i++)
    {
        for (int j = 0; j < rec->ny; j++)
        {
            fwrite(rec->wavefield[i][j], sizeof(double), rec->nt, f);
        }
    }

    fclose(f);
    printf("Wavefield data saved to %s\n", output_file);
}



int RecReceiver(Rec *rec, int it, double **p){

    return 0;
}

int RecSave(Rec *rec){

    return 0;
}

int RecSnap(Rec *rec, int it, double **snp){
    
    return 0;
}

