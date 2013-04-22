/*
 *  main.cpp
 */

//#define EMULATION_MODE
//#define use_export	// uncomment this if project is built using a compiler that
// supports the C++ keyword "export".  If not using such a 
// compiler, be sure not to add cuLsoda.cc to the target, or
// you will get redefinition errors.

#include <stdio.h>
#include <math.h>
#include "cuLsoda_kernel.cu"
#include <sys/timeb.h>

#define COSFUNCTION cos
#define REAL double

struct benchFex
{
	__device__ void operator()(int *neq, REAL *t, REAL *y, REAL *ydot/*, void *otherData*/)
	{
		for (int itr = 0; itr < *neq; itr++)
		{
			ydot[itr] = COSFUNCTION(itr * *t);
		}
	}
};

int main(int argc, char *argv[])   /* Main program */ 
{
	
	int probSize;
//	printf("Please input the problem size: ");
	sscanf(argv[1],"%d",&probSize);

    /* Local variables */
     REAL *t = (REAL*)malloc(sizeof(REAL)*probSize);
	 REAL *y/*[4]*/ = (REAL*)malloc(sizeof(REAL)*4*probSize);
     int *jt = (int*)malloc(sizeof(int)*probSize);
     int *neq = (int*)malloc(sizeof(int)*probSize);
	 int *liw = (int*)malloc(sizeof(int)*probSize);
	 int *lrw = (int*)malloc(sizeof(int)*probSize);
     REAL *atol/*[1]*/ = (REAL*)malloc(sizeof(REAL)*probSize);
     int *itol =(int*) malloc(sizeof(int)*probSize);
	 int *iopt =(int*) malloc(sizeof(int)*probSize);
     REAL *rtol = (REAL*)malloc(sizeof(REAL)*probSize);
     int *iout =(int*) malloc(sizeof(int)*probSize);
     REAL *tout =(REAL*) malloc(sizeof(REAL)*probSize);
     int *itask = (int*)malloc(sizeof(int)*probSize);
	 int *iwork/*[24]*/ =(int*) malloc(sizeof(int)*24*probSize);
     REAL *rwork/*[86]*/ = (REAL*)malloc(sizeof(REAL)*86*probSize);
	 int *istate = (int*)malloc(sizeof(int)*probSize);
	struct cuLsodaCommonBlock common[probSize];
	struct cuLsodaCommonBlock *Hcommon = common;
		 int *err = (int*)malloc(sizeof(int)*probSize);

	/* End Local Block */

	/* Pointers to Device versions of Local variables */
	REAL	*_Dt;
	REAL	*_Dy;	// [4]
	int	*_Djt;
	int	*_Dneq;
	int	*_Dliw;
	int	*_Dlrw;
    REAL	*_Datol;	//[1]
    int	*_Ditol;
	int	*_Diopt;
    REAL	*_Drtol;
    int	*_Diout;
    REAL	*_Dtout;
    int	*_Ditask;
	int	*_Diwork;	// [24]
    REAL	*_Drwork;	// [86]
	int	*_Distate;
	struct cuLsodaCommonBlock *_Dcommon;
	int	*_Derr;

	/* End Pointer Block */
	
	
	
	/* Method instantiations for Derivative and Jacobian functions to send to template */
	benchFex fex;
	myJex jex;

	//printf("\n\n\n\n\n\nPlease enter the desired output time: ");
	REAL outtime;
//	scanf("%lf",&outtime);
	//printf("%f\n",outtime);
	outtime = 10000.;
	int mxstpH = 200000;
	//printf("Please enter the desired Maximum Step number: ");
	//scanf("%d",&mxstpH);

	clock_t startTime, endTime;
        double procTime = 0;
	struct timeb bstart,bstop;
	double btime;


	/* Assignment of initial values to locals */
    for (int itr = 0; itr < probSize; itr++)
	{
		*(neq+itr) = 4;
		*(y+0+itr*4) = (REAL)0.;
		*(y+1+itr*4) = (REAL)0.;
		*(y+2+itr*4) = (REAL)0.;
		*(y+3+itr*4) = (REAL)0.;
		*(t+itr) = (REAL)0.;
		*(tout+itr) = (REAL) outtime;
		//	*tout = (REAL)1000000.;
		
		*(itol +itr) = 1;
		*(rtol +itr) = (REAL)0.;
		*(atol +itr) = (REAL)1.e-5;
		*(itask +itr) = 1;
		*(istate +itr) = 1;
		*(iopt +itr) = 1;
		
		*(iwork + 5  +itr*24) = mxstpH;
		*(lrw +itr) = 86;
		*(liw +itr) = 24;
		*(jt +itr) = 2;
		cuLsodaCommonBlockInit(&Hcommon[itr]);
		*(err +itr) = -1;
	}
	

	startTime = clock();
	ftime(&bstart);
	/* Allocate device memory for each of the pointers, and copy the values from local to device */
	cudaMalloc((void**)&_Dt,sizeof(REAL)*probSize);								cudaMemcpy(_Dt,t,sizeof(REAL)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dy,sizeof(REAL)*4*probSize);							cudaMemcpy(_Dy,y,sizeof(REAL)*4*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Djt,sizeof(int)*probSize);								cudaMemcpy(_Djt,jt,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dneq,sizeof(int)*probSize);							cudaMemcpy(_Dneq,neq,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dliw,sizeof(int)*probSize);							cudaMemcpy(_Dliw,liw,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dlrw,sizeof(int)*probSize);							cudaMemcpy(_Dlrw,lrw,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Datol,sizeof(REAL)*probSize);							cudaMemcpy(_Datol,atol,sizeof(REAL)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Ditol,sizeof(int)*probSize);							cudaMemcpy(_Ditol,itol,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diopt,sizeof(int)*probSize);							cudaMemcpy(_Diopt,iopt,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Drtol,sizeof(REAL)*probSize);							cudaMemcpy(_Drtol,rtol,sizeof(REAL)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diout,sizeof(int)*probSize);							cudaMemcpy(_Diout,iout,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dtout,sizeof(REAL)*probSize);							cudaMemcpy(_Dtout,tout,sizeof(REAL)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Ditask,sizeof(int)*probSize);							cudaMemcpy(_Ditask,itask,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Diwork,sizeof(int)*24*probSize);						cudaMemcpy(_Diwork,iwork,sizeof(int)*24*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Drwork,sizeof(REAL)*86*probSize);						cudaMemcpy(_Drwork,rwork,sizeof(REAL)*86*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Distate,sizeof(int)*probSize);							cudaMemcpy(_Distate,istate,sizeof(int)*probSize,cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Dcommon,sizeof(struct cuLsodaCommonBlock)*probSize);	cudaMemcpy(_Dcommon,Hcommon,sizeof(struct cuLsodaCommonBlock)*probSize, cudaMemcpyHostToDevice);
	cudaMalloc((void**)&_Derr,sizeof(int)*probSize);							cudaMemcpy(_Derr,istate,sizeof(int)*probSize,cudaMemcpyHostToDevice);

	/* End Allocation and Copy Block */
	
	int error = -1;
	int error2 = -1;
	//cudaEvent_t start,stop;
	//cudaEventCreate(&start);
	//cudaEventCreate(&stop);
	
	
	
   // for (*iout = 1; *iout <= 12; ++*iout) 
	{
		//printf("Calling Kernel Sanders\n");
		int threadsPerBlock = 32;
		int blocksPerGrid = (probSize + threadsPerBlock -1)/threadsPerBlock;
		//cudaEventRecord(start, 0);
		
		Sanders<<<blocksPerGrid,threadsPerBlock>>>(fex, _Dneq, _Dy, _Dt, _Dtout, _Ditol, _Drtol, _Datol, _Ditask, _Distate, _Diopt, _Drwork, _Dlrw, _Diwork, _Dliw, jex, _Djt, _Dcommon, _Derr,probSize);

		error = cudaGetLastError();		
		error2 = cudaThreadSynchronize();
		//cudaEventRecord(stop,0);
		//cudaEventSynchronize(stop);
		//float elapsedTime;
		//cudaEventElapsedTime(&elapsedTime, start, stop);
		
		/* Copy memory back from Device to Host */
		cudaMemcpy(t,_Dt,sizeof(REAL)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(y,_Dy,sizeof(REAL)*4*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(jt,_Djt,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(neq,_Dneq,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(liw,_Dliw,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(lrw,_Dlrw,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(atol,_Datol,sizeof(REAL)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(itol,_Ditol,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(iopt,_Diopt,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(rtol,_Drtol,sizeof(REAL)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(tout,_Dtout,sizeof(REAL)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(itask,_Ditask,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(iwork,_Diwork,sizeof(int)*24*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(rwork,_Drwork,sizeof(REAL)*86*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(istate,_Distate,sizeof(int)*probSize,cudaMemcpyDeviceToHost);
		cudaMemcpy(Hcommon,_Dcommon,sizeof(struct cuLsodaCommonBlock)*probSize, cudaMemcpyDeviceToHost);
		cudaMemcpy(err,_Derr,sizeof(int)*probSize,cudaMemcpyDeviceToHost);

		/* End Copy Block */

		endTime= clock();
	        ftime(&bstop);
		procTime = ((double)endTime -(double)startTime)/(double)CLOCKS_PER_SEC;
		btime=(double)(bstop.time-bstart.time)+0.001*((bstop.millitm-bstart.millitm));
		printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\t%f\n",probSize, *(t), y[0], y[1], y[2], y[3], procTime,btime );


		//for (int itr = 0; itr < probSize; itr++)
		//printf("At t =\t%- #12.2ey = %- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\t err = %d\n", *(t+itr), y[0+itr*4], y[1+itr*4], y[2+itr*4], y[3+itr*4],err[itr]);
		//printf("error = %d\terror2 = %d\n",error,error2);
		//printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\n",probSize, *(t), y[0], y[1], y[2], y[3],elapsedTime/1000.);

		//printf("Error = %d\t Error2 = %d\nElapsed Time = %f seconds\n",error,error2,elapsedTime/1000.);
	/*	printf("\tCM_conit = %f\n",Hcommon->CM_conit);
		printf("\tCM_crate = %f\n",Hcommon->CM_crate);
		printf("\tCM_ccmax = %f\n",Hcommon->CM_ccmax);
		printf("\tCM_el0 = %f\n",Hcommon->CM_el0);
		printf("\tCM_h__ = %f\n",Hcommon->CM_h__);
		printf("\tCM_hmin = %f\n",Hcommon->CM_hmin);
		printf("\tCM_hmxi = %f\n",Hcommon->CM_hmxi);
		printf("\tCM_hu = %f\n",Hcommon->CM_hu);
		printf("\tCM_rc = %f\n",Hcommon->CM_rc);
		printf("\tCM_tn = %f\n",Hcommon->CM_tn);
		printf("\tCM_uround = %f\n",Hcommon->CM_uround);
		printf("\tCM_pdest = %f\n",Hcommon->CM_pdest);
		printf("\tCM_pdlast = %f\n",Hcommon->CM_pdlast);
		printf("\tCM_ratio = %f\n",Hcommon->CM_ratio);
		printf("\tCM_hold = %f\n",Hcommon->CM_hold);
		printf("\tCM_rmax = %f\n",Hcommon->CM_rmax);
		printf("\tCM_tsw = %f\n",Hcommon->CM_tsw);
		printf("\tCM_pdnorm = %f\n",Hcommon->CM_pdnorm);*/
		
		error = error2 = -1;

		if (istate < 0) {
			goto L80;
		}
		

		/* L40: */
		//*tout *= 10.; 
		//cudaMemcpy(_Dtout,tout,sizeof(REAL),cudaMemcpyHostToDevice);
    }
 	//printf("Number of Steps:  %i\nNo. f-s: %i\nNo. J-s = %i\nMethod Last Used = %i\nLast switch was at t = %g\n",iwork[10],iwork[11],iwork[12],iwork[18],rwork[14]);
	
L80:
 	//printf( "STOP istate is < 0\n\n\n\n");
    return 0;
} /* MAIN__ */


