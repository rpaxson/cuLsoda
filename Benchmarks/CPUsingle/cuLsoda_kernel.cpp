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
void Sanders(Fex fex, int *neq, REAL *y, REAL *t, REAL *tout, int *itol, REAL *rtol, REAL *atol, int *itask, int *istate, int *iopt, REAL *rwork, int *lrw, int *iwork, int *liw, Jex jac, int *jt, struct cuLsodaCommonBlock *common, int *err)
{
	*err = dlsoda_(fex, neq, y, t, tout, itol, rtol, atol, itask, istate, iopt, rwork, lrw, iwork, liw, jac, jt, common);
}


#endif

