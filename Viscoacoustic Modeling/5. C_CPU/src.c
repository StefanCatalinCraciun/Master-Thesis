#include "src.h"


Src *SrcNew(double *source, int sx, int sy){
    Src *src = (Src*)malloc(sizeof(Src));
    src->Src = source;
    src->Sx = sx;
    src->Sy = sy;
    src->Ns = 0; // initialize Ns to 0, assuming you'll update it later
    return src;
}


void SrcDel(Src *src){
    if(src){
        free(src);
        src = NULL;
    }
    return;
}


void SrcRicker(double *src, double t0, double f0, int nt, double dt){
        
    double t, w0, arg;
    int i;
    
    w0 = 2.0*3.1415*f0;
    for(i=0; i<nt; i++){
        t = (double)i*dt-t0;
        arg = w0*t;
        src[i] = (1.0 - 0.5 * pow(arg, 2)) * exp(-0.25 * pow(arg,2));
    }
}
