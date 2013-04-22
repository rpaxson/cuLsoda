/*
 *  main.cpp
 */

//#define EMULATION_MODE
//#define use_export	// uncomment this if project is built using a compiler that
// supports the C++ keyword "export".  If not using such a 
// compiler, be sure not to add cuLsoda.cc to the target, or
// you will get redefinition errors.
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "cuLsoda_kernel.cpp"
#include <time.h>
#include <sys/timeb.h>

#define COSFUNCTION cos
#define REAL double

struct benchFex
{
	void operator()(int *neq, REAL *t, REAL *y, REAL *ydot/*, void *otherData*/)
	{
		for (int itr = 0; itr < *neq; itr++)
		{
			ydot[itr] = COSFUNCTION(itr * *t);
		}
	}
};

int main(int argc, char *argv[])   /* Main program */ 
{
	
	int runs;
	sscanf(argv[1],"%d",&runs);
	//scanf("%d",&runs);
	
    /* Local variables */
     REAL *t = (REAL*)malloc(sizeof(REAL));
	 REAL *y/*[4]*/ = (REAL*)malloc(sizeof(REAL)*4);
     int *jt = (int*)malloc(sizeof(int));
     int *neq = (int*)malloc(sizeof(int));
	 int *liw = (int*)malloc(sizeof(int));
	 int *lrw = (int*)malloc(sizeof(int));
     REAL *atol/*[1]*/ = (REAL*)malloc(sizeof(REAL));
     int *itol =(int*) malloc(sizeof(int));
	 int *iopt =(int*) malloc(sizeof(int));
     REAL *rtol = (REAL*)malloc(sizeof(REAL));
     int *iout =(int*) malloc(sizeof(int));
     REAL *tout =(REAL*) malloc(sizeof(REAL));
     int *itask = (int*)malloc(sizeof(int));
	 int *iwork/*[24]*/ =(int*) malloc(sizeof(int)*24);
     REAL *rwork/*[86]*/ = (REAL*)malloc(sizeof(REAL)*86);
	 int *istate = (int*)malloc(sizeof(int));
	struct cuLsodaCommonBlock common;
	struct cuLsodaCommonBlock *Hcommon = &common;
		 int *err = (int*)malloc(sizeof(int));

	/* End Local Block */
	clock_t startTime, endTime;
	double procTime = 0;
	struct timeb bstart,bstop;
	double btime;

	/* End Pointer Block */
	
	
	
	/* Method instantiations for Derivative and Jacobian functions to send to template */
	benchFex fex;
	myJex jex;

	
	/* Assignment of initial values to locals */
    *neq = 4;
	y[0] = (REAL)0.;
    y[1] = (REAL)0.;
    y[2] = (REAL)0.;
	y[3] = (REAL)0.;
    *t = (REAL)0.;
//	printf("\n\n\n\n\n\nPlease enter the desired output time: ");
    REAL outtime = 10000;
	//scanf("%lf",&outtime);
	//printf("%f\n",outtime);
	*tout = (REAL) outtime;
//	*tout = (REAL)1000000.;

	*itol = 1;
    *rtol = (REAL)0.;
    *atol = (REAL)1.e-5;
    *itask = 1;
    *istate = 1;
    *iopt = 1;
	int mxstpH;
	//printf("Please enter the desired Maximum Step number: ");
	//scanf("%d",&mxstpH);
	mxstpH = 200000;
	iwork[5] = mxstpH;
    *lrw = 86;
    *liw = 24;
    *jt = 2;
	cuLsodaCommonBlockInit(Hcommon);
	*err = -1;
	
		
	int error = -1;
	int error2 = -1;
	

	startTime = clock();
	ftime(&bstart);
	for (*iout = 0; *iout < runs; ++*iout) 
	{
			y[0] = (REAL)0.;
		y[1] = (REAL)0.;
		y[2] = (REAL)0.;
		y[3] = (REAL)0.;
		*t = (REAL)0.;
		//	printf("\n\n\n\n\n\nPlease enter the desired output time: ");
		
		*itask = 1;
		*istate = 1;
		*iopt = 1;
		iwork[5] = mxstpH;
		//printf("Calling Kernel Sanders\n");
		Sanders(fex, neq, y, t, tout, itol, rtol, atol, itask, istate, iopt, rwork, lrw, iwork, liw, jex, jt, Hcommon, err);

			//float elapsedTime;
		
	
		
				

	//	printf("At t =\t%- #12.2ey = %- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\nError = %d\t Error2 = %d\t err = %d\nElapsed Time = %f seconds\n", *t, y[0], y[1], y[2], y[3],error,error2,*err,elapsedTime/1000.);
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
		
		//error = error2 = -1;

		if (istate < 0) {
			printf( "STOP istate is < 0\n\n\n\n");
		}
		
    }
 	//printf("Number of Steps:  %i\nNo. f-s: %i\nNo. J-s = %i\nMethod Last Used = %i\nLast switch was at t = %g\n",iwork[10],iwork[11],iwork[12],iwork[18],rwork[14]);
	endTime= clock();
	ftime(&bstop);
	btime=(double)(bstop.time-bstart.time)+0.001*((bstop.millitm-bstart.millitm));
	procTime = ((double)endTime -(double)startTime)/(double)CLOCKS_PER_SEC;
L80:
 
	printf("runs\t%d\ttime\t%- #12.2ey\t%- #16.12e\t%- #16.12e\t%- #16.12e\t%- #16.12e\tElTime\t%f\t%f\n",runs, *(t), y[0], y[1], y[2], y[3], procTime,btime );
	
	return 0;
} /* MAIN__ */


