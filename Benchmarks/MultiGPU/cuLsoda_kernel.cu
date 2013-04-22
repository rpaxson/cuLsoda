/*
 *  cuLsoda_kernel.cu
 *  cuLsoda
 *
 */
 #ifndef _CULSODA_CU_H_
 #define _CULSODA_CU_H_
 
 #include "cuLsoda.cu.h"
 
 #define REAL double
 
 template<typename Fex, typename Jex>
__global__ void Sanders(Fex fex, int *neq, REAL *y, REAL *t, REAL *tout, int *itol, REAL *rtol, REAL *atol, int *itask, int *istate, int *iopt, REAL *rwork, int *lrw, int *iwork, int *liw, Jex jac, int *jt, struct cuLsodaCommonBlock *common, int *err, int probSize)
{
	int me = threadIdx.x + blockIdx.x * blockDim.x;
//	printf("Thread ID: %d\tProbsize: %d\n",me,probSize);
	if(me < probSize){
//	printf("neq: %d\ty[0]: %f\ty[1]: %f\ty[2]: %f\ty[3]: %f\tt: %f\ttout: %f\n",neq[me],y[4*me],y[4*me+1],y[4*me+2],y[4*me+3],t[me],tout[me]);
	err[me] = dlsoda_(fex, &neq[me], &y[4*me], &t[me], &tout[me], &itol[me], &rtol[me], &atol[me], &itask[me], &istate[me], &iopt[me], &rwork[86*me], &lrw[me], &iwork[24*me], &liw[me], jac, &jt[me], &common[me]);
	
	}
	__syncthreads();
}


#endif

