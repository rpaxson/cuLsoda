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
#include <omp.h>
#include "cuLsoda_kernel.cu"
#include <stdlib.h>
#include <time.h>
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
	int deviceCount = 4;
	//cudaGetDeviceCount(&deviceCount);
	int probSize;
	//printf("Please input the problem size: ");
	sscanf(argv[1],"%d",&probSize);
	//scanf("%d",&probSize);
	int arrayOffset = (probSize-1)/deviceCount + 1;
	
	
	
    /* Local variables */
	REAL *t = (REAL*)malloc(sizeof(REAL)* arrayOffset*deviceCount);
	REAL *y/*[4]*/ = (REAL*)malloc(sizeof(REAL)*4*arrayOffset*deviceCount);
	int *jt = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	int *neq = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	int *liw = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	int *lrw = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	REAL *atol/*[1]*/ = (REAL*)malloc(sizeof(REAL)*arrayOffset*deviceCount);
	int *itol =(int*) malloc(sizeof(int)*arrayOffset*deviceCount);
	int *iopt =(int*) malloc(sizeof(int)*arrayOffset*deviceCount);
	REAL *rtol = (REAL*)malloc(sizeof(REAL)*arrayOffset*deviceCount);
	int *iout =(int*) malloc(sizeof(int)*arrayOffset*deviceCount);
	REAL *tout =(REAL*) malloc(sizeof(REAL)*arrayOffset*deviceCount);
	int *itask = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	int *iwork/*[24]*/ =(int*) malloc(sizeof(int)*24*arrayOffset*deviceCount);
	REAL *rwork/*[86]*/ = (REAL*)malloc(sizeof(REAL)*86*arrayOffset*deviceCount);
	int *istate = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	struct cuLsodaCommonBlock common[arrayOffset*deviceCount];
	struct cuLsodaCommonBlock *Hcommon = common;
	int *err = (int*)malloc(sizeof(int)*arrayOffset*deviceCount);
	
	/* End Local Block */
	
	
	
	
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
	struct timeb startb,stopb;
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
	ftime(&startb);
#pragma omp parallel num_threads(deviceCount)
	{
		
		unsigned int cpu_thread_id = omp_get_thread_num();
		unsigned int myFirstIdx = cpu_thread_id*(arrayOffset);
	//	unsigned int myLastIdx = (myFirstIdx + arrayOffset) > probSize ? probSize - 1 : (myFirstIdx + arrayOffset);
		cudaSetDevice(cpu_thread_id);

		
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
		
		
		/* Allocate device memory for each of the pointers, and copy the values from local to device */
		cudaMalloc((void**)&_Dt,sizeof(REAL)*arrayOffset);
		cudaMemcpy(_Dt,t + myFirstIdx,sizeof(REAL)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dy,sizeof(REAL)*4*arrayOffset);
		cudaMemcpy(_Dy,y + myFirstIdx*4,sizeof(REAL)*4*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Djt,sizeof(int)*arrayOffset);
		cudaMemcpy(_Djt,jt + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dneq,sizeof(int)*arrayOffset);
		cudaMemcpy(_Dneq,neq + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dliw,sizeof(int)*arrayOffset);
		cudaMemcpy(_Dliw,liw + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dlrw,sizeof(int)*arrayOffset);
		cudaMemcpy(_Dlrw,lrw + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Datol,sizeof(REAL)*arrayOffset);
		cudaMemcpy(_Datol,atol + myFirstIdx*1,sizeof(REAL)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Ditol,sizeof(int)*arrayOffset);
		cudaMemcpy(_Ditol,itol + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Diopt,sizeof(int)*arrayOffset);
		cudaMemcpy(_Diopt,iopt + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Drtol,sizeof(REAL)*arrayOffset);
		cudaMemcpy(_Drtol,rtol + myFirstIdx,sizeof(REAL)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Diout,sizeof(int)*arrayOffset);	
		cudaMemcpy(_Diout,iout + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dtout,sizeof(REAL)*arrayOffset);
		cudaMemcpy(_Dtout,tout + myFirstIdx,sizeof(REAL)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Ditask,sizeof(int)*arrayOffset);
		cudaMemcpy(_Ditask,itask + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Diwork,sizeof(int)*24*arrayOffset);
		cudaMemcpy(_Diwork,iwork + myFirstIdx*24,sizeof(int)*24*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Drwork,sizeof(REAL)*86*arrayOffset);
		cudaMemcpy(_Drwork,rwork + myFirstIdx*86,sizeof(REAL)*86*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Distate,sizeof(int)*arrayOffset);
		cudaMemcpy(_Distate,istate + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Dcommon,sizeof(struct cuLsodaCommonBlock)*arrayOffset);
		cudaMemcpy(_Dcommon,&Hcommon[myFirstIdx],sizeof(struct cuLsodaCommonBlock)*arrayOffset, cudaMemcpyHostToDevice);
		cudaMalloc((void**)&_Derr,sizeof(int)*arrayOffset);
		cudaMemcpy(_Derr,istate + myFirstIdx,sizeof(int)*arrayOffset,cudaMemcpyHostToDevice);
		
		
		/* End Allocation and Copy Block */
		
		int error = -1;
		int error2 = -1;
		cudaEvent_t start,stop;
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		
		
		// for (*iout = 1; *iout <= 12; ++*iout) 
		//{
		//printf("Calling Kernel Sanders\n");
		int threadsPerBlock = 32;
		int blocksPerGrid = (arrayOffset + threadsPerBlock -1)/threadsPerBlock;
		//cudaEventRecord(start, 0);
		
//		#pragma omp barrier
//		if (cpu_thread_id == 0)
//		{
//			startTime = clock();
//		}

		if (cpu_thread_id == deviceCount - 1)
		{
	//	printf("last\n");
			Sanders<<<blocksPerGrid,threadsPerBlock>>>(fex, _Dneq, _Dy, _Dt, _Dtout, _Ditol, _Drtol, _Datol, _Ditask, _Distate, _Diopt, _Drwork, _Dlrw, _Diwork, _Dliw, jex, _Djt, _Dcommon, _Derr,arrayOffset*(1-deviceCount)+probSize);
		} else
		{
			Sanders<<<blocksPerGrid,threadsPerBlock>>>(fex, _Dneq, _Dy, _Dt, _Dtout, _Ditol, _Drtol, _Datol, _Ditask, _Distate, _Diopt, _Drwork, _Dlrw, _Diwork, _Dliw, jex, _Djt, _Dcommon, _Derr,arrayOffset);
		}
		error = cudaGetLastError();		
		error2 = cudaThreadSynchronize();
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		float elapsedTime;
		cudaEventElapsedTime(&elapsedTime, start, stop);
		
		/* Copy memory back from Device to Host */
		cudaMemcpy(t + myFirstIdx,_Dt,sizeof(REAL)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(y + myFirstIdx*4,_Dy,sizeof(REAL)*4*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(jt + myFirstIdx,_Djt,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(neq + myFirstIdx,_Dneq,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(liw + myFirstIdx,_Dliw,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(lrw + myFirstIdx,_Dlrw,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(atol + myFirstIdx*1,_Datol,sizeof(REAL)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(itol + myFirstIdx,_Ditol,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(iopt + myFirstIdx,_Diopt,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(rtol + myFirstIdx,_Drtol,sizeof(REAL)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(tout + myFirstIdx,_Dtout,sizeof(REAL)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(itask + myFirstIdx,_Ditask,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(iwork + myFirstIdx*24,_Diwork,sizeof(int)*24*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(rwork + myFirstIdx*86,_Drwork,sizeof(REAL)*86*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(istate + myFirstIdx,_Distate,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		cudaMemcpy(Hcommon + myFirstIdx,_Dcommon,sizeof(struct cuLsodaCommonBlock)*arrayOffset, cudaMemcpyDeviceToHost);
		cudaMemcpy(err + myFirstIdx,_Derr,sizeof(int)*arrayOffset,cudaMemcpyDeviceToHost);
		
		/* End Copy Block */
 

//		#pragma	omp barrier
//		if (cpu_thread_id== 0 )
//		{
//			endTime= clock();
//			procTime = ((double)endTime -(double)startTime)/(double)CLOCKS_PER_SEC;
//		}	printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\n",probSize, *(t), y[0], y[1], y[2], y[3], procTime );
			
		//for (int itr = 0; itr < arrayOffset; itr++)
		//printf("At t =\t%- #12.2ey = %- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\t err = %d\n", *(t+itr), y[0+itr*4], y[1+itr*4], y[2+itr*4], y[3+itr*4],err[itr]);
//		printf("error = %d\terror2 = %d\n",error,error2);
//		printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\n",arrayOffset, *(t + myFirstIdx), y[0 + myFirstIdx*4], y[1 + myFirstIdx*4], y[2 + myFirstIdx*4], y[3 + myFirstIdx*4],elapsedTime/1000.);
		//printf("%d\t%d\nError = %d\t Error2 = %d\n",cpu_thread_id,myFirstIdx,error,error2);
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
			printf( "STOP istate is < 0\n\n\n\n");		
			}
		
		
		/* L40: */
		//*tout *= 10.; 
		//cudaMemcpy(_Dtout,tout,sizeof(REAL),cudaMemcpyHostToDevice);
		//}
		//printf("Number of Steps:  %i\nNo. f-s: %i\nNo. J-s = %i\nMethod Last Used = %i\nLast switch was at t = %g\n",iwork[10],iwork[11],iwork[12],iwork[18],rwork[14]);
		
}		
	
		endTime= clock();
		ftime(&stopb);
		procTime = ((double)endTime -(double)startTime)/((double)CLOCKS_PER_SEC*deviceCount);
		btime=(double)(stopb.time-startb.time)+0.001*((stopb.millitm-startb.millitm));
		printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\t%f\n",probSize, *(t), y[0], y[1], y[2], y[3], procTime,btime );
//		printf("%f\t%f\t%f\n",(double)endTime,(double)startTime,(double)CLOCKS_PER_SEC);
	return 0;
} /* MAIN__ */


