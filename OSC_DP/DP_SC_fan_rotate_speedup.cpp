#include "mex.h"
#include "string.h"
#include "math.h"

#define infty 100000
void speedup(double* minimum,double* index,
        double *energy,double* leny,double lenxneig,int Ny, double lambda)
 {
    double *cand=(double*)calloc(Ny*Ny, sizeof(double));
    for (int jj=0;jj<Ny;jj++){
        
        memcpy(cand, energy, Ny*Ny*sizeof(double));
        
        for (int row=0;row<Ny;row++)
            for (int col=0;col<Ny;col++)
                *(cand+row+col*Ny)+=lambda*fabs(*(leny+row+Ny*jj)-lenxneig);
                                    
        for (int col=0;col<Ny;col++)
            *(minimum+jj+Ny*col)=infty;
        
        for (int col=0;col<Ny;col++){
            for (int row=0;row<Ny;row++){
                if (*(cand+row+col*Ny)<*(minimum+jj+Ny*col)){
                    *(minimum+jj+Ny*col)=*(cand+row+col*Ny);
                    *(index+jj+Ny*col)=row+1;
                }
            }
        }
    }
    free(cand);
}


void mexFunction
(int nl, mxArray *pl[], int nr, const mxArray *pr[])
{
    if(nr==4){
        double *energy=mxGetPr(pr[0]);
        double* leny=mxGetPr(pr[1]);
        double* lenxneig=mxGetPr(pr[2]);
        double* lambda=mxGetPr(pr[3]);
        int Ny=mxGetM(pr[0]);
        
        
        pl[0] = mxCreateDoubleMatrix(Ny,Ny, mxREAL);
        pl[1] = mxCreateDoubleMatrix(Ny,Ny, mxREAL);
        double *minimum=mxGetPr(pl[0]);
        double* index=mxGetPr(pl[1]);
        
        speedup( minimum, index, energy, leny, *lenxneig, Ny, *lambda);
        
    }
}