#include "diff.h"


Diff* DiffNew(int l){

    int i,j,k;

    Diff* diff = NULL;
    diff = (Diff*) malloc(sizeof(Diff));
    
    diff->w = (double*) malloc(l*sizeof(double));
    diff->lmax = 8;

    if(l < 1){l=1;}
    if(l > diff->lmax){l=diff->lmax;}

    diff->l = l;

    // Load coefficients
    for (i=0; i<diff->lmax; i++){
        for (j=0; j<diff->lmax; j++){
            diff->coeffs[i][j] = 0.0;
        } 
    }
    // l=1
    diff->coeffs[0][0] = 1.0021;

    // l=2
    diff->coeffs[1][0] = 1.1452;
    diff->coeffs[1][1] = -0.0492;
    
    // l=3
    diff->coeffs[2][0] = 1.2036;
    diff->coeffs[2][1] = -0.0833;
    diff->coeffs[2][2] = 0.0097;

    // l=4
    diff->coeffs[3][0] = 1.2316;
    diff->coeffs[3][1] = -0.1041;
    diff->coeffs[3][2] = 0.0206;
    diff->coeffs[3][3] = -0.0035;

    // l=5
    diff->coeffs[4][0] = 1.2463;
    diff->coeffs[4][1] = -0.1163;
    diff->coeffs[4][2] = 0.0290;
    diff->coeffs[4][3] = -0.0080;
    diff->coeffs[4][4] = 0.0018;

    // l=6
    diff->coeffs[5][0] = 1.2542;
    diff->coeffs[5][1] = -0.1213;
    diff->coeffs[5][2] = 0.0344;
    diff->coeffs[5][3] = -0.017;
    diff->coeffs[5][4] = 0.0038;
    diff->coeffs[5][5] = -0.0011;

    // l=7
    diff->coeffs[6][0] = 1.2593;
    diff->coeffs[6][1] = -0.1280;
    diff->coeffs[6][2] = 0.0384;
    diff->coeffs[6][3] = -0.0147;
    diff->coeffs[6][4] = 0.0059;
    diff->coeffs[6][5] = -0.0022;
    diff->coeffs[6][6] = 0.0007;

    // l=8
    diff->coeffs[7][0] = 1.2626;
    diff->coeffs[7][1] = -0.1312;
    diff->coeffs[7][2] = 0.0412;
    diff->coeffs[7][3] = -0.0170;
    diff->coeffs[7][4] = 0.0076;
    diff->coeffs[7][5] = -0.0034;
    diff->coeffs[7][6] = 0.0014;
    diff->coeffs[7][7] = -0.0005;


    for(k=0;k<l;k++){
        diff->w[k] = diff->coeffs[l-1][k];
    }

    return diff;
}


void DiffDel(Diff *diff){
    if (diff){
        free(diff);
        diff=NULL;
    }return;
}



