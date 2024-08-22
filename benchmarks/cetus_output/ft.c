/*
Copyright (C) 1991-2020 Free Software Foundation, Inc.
   This file is part of the GNU C Library.

   The GNU C Library is free software; you can redistribute it andor
   modify it under the terms of the GNU Lesser General Public
   License as published by the Free Software Foundation; either
   version 2.1 of the License, or (at your option) any later version.

   The GNU C Library is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
   Lesser General Public License for more details.

   You should have received a copy of the GNU Lesser General Public
   License along with the GNU C Library; if not, see
   <https:www.gnu.org/licenses/>. 
*/
/*
This header is separate from features.h so that the compiler can
   include it implicitly at the start of every compilation.  It must
   not itself include <features.h> or any other header that includes
   <features.h> because the implicit include comes before any feature
   test macros that may be defined in a source file before it first
   explicitly includes a system header.  GCC knows the name of this
   header in order to preinclude it. 
*/
/*
glibc's intent is to support the IEC 559 math functionality, real
   and complex.  If the GCC (4.9 and later) predefined macros
   specifying compiler intent are available, use them to determine
   whether the overall intent is to support these features; otherwise,
   presume an older compiler has intent to support these features and
   define these macros by default. 
*/
/*
wchar_t uses Unicode 10.0.0.  Version 10.0 of the Unicode Standard is
   synchronized with ISOIEC 10646:2017, fifth edition, plus
   the following additions from Amendment 1 to the fifth edition:
   - 56 emoji characters
   - 285 hentaigana
   - 3 additional Zanabazar Square characters
*/
/*
--------------------------------------------------------------------
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - FT

  This benchmark is an OpenMP C version of the NPB FT code.
  
  The OpenMP C versions are developed by RWCP and derived from the serial
  Fortran versions in "NPB 2.3-serial" developed by NAS.

  Permission to use, copy, distribute and modify this software for any
  purpose with or without fee is hereby granted.
  This software is provided "as is" without express or implied warranty.
  
  Send comments on the OpenMP C versions to pdp-openmp@rwcp.or.jp

  Information on OpenMP activities at RWCP is available at:

           http:pdplab.trc.rwcp.or.jppdperf/Omni/
  
  Information on NAS Parallel Benchmarks 2.3 is available at:
  
           http:www.nas.nasa.gov/NAS/NPB/

--------------------------------------------------------------------
*/
/*
--------------------------------------------------------------------

  Authors: D. Bailey
           W. Saphir

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------
*/
#include "npb-C.h"
/* global variables */
#include "global.h"
/* function declarations */
static void evolve(dcomplex u0[NZ][NY][NX], dcomplex u1[NZ][NY][NX], int t, int indexmap[NZ][NY][NX], int d[3]);
static void compute_initial_conditions(dcomplex u0[NZ][NY][NX], int d[3]);
static void ipow46(double a, int exponent, double * result);
static void setup(void );
static void compute_indexmap(int indexmap[NZ][NY][NX], int d[3]);
static void print_timers(void );
static void fft(int dir, dcomplex x1[NZ][NY][NX], dcomplex x2[NZ][NY][NX]);
static void cffts1(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18]);
static void cffts2(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18]);
static void cffts3(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18]);
static void fft_init(int n);
static void cfftz(int is, int m, int n, dcomplex x[NX][18], dcomplex y[NX][18]);
static void fftz2(int is, int l, int m, int n, int ny, int ny1, dcomplex u[NX], dcomplex x[NX][18], dcomplex y[NX][18]);
static int ilog2(int n);
static void checksum(int i, dcomplex u1[NZ][NY][NX], int d[3]);
static void verify(int d1, int d2, int d3, int nt, boolean * verified, char * class);
/*
--------------------------------------------------------------------
c FT benchmark
c-------------------------------------------------------------------
*/
int main(int argc, char * * argv)
{
	/*
	c-------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int i, ierr;
	/*
	------------------------------------------------------------------
	c u0, u1, u2 are the main arrays in the problem. 
	c Depending on the decomposition, these arrays will have different 
	c dimensions. To accomodate all possibilities, we allocate them as 
	c one-dimensional arrays and pass them to subroutines for different 
	c views
	c  - u0 contains the initial (transformed) initial condition
	c  - u1 and u2 are working arrays
	c  - indexmap maps i,j,k of u0 to the correct i^2+j^2+k^2 for the
	c    time evolution operator. 
	c-----------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c Large arrays are in common so that they are allocated on the
	c heap rather than the stack. This common block is not
	c referenced directly anywhere else. Padding is to avoid accidental 
	c cache problems, since all array sizes are powers of two.
	c-------------------------------------------------------------------
	*/
	static dcomplex u0[NZ][NY][NX];
	static dcomplex pad1[3];
	static dcomplex u1[NZ][NY][NX];
	static dcomplex pad2[3];
	static dcomplex u2[NZ][NY][NX];
	static dcomplex pad3[3];
	static int indexmap[NZ][NY][NX];
	int iter;
	int nthreads = 1;
	double total_time, mflops;
	boolean verified;
	char class;
	/*
	--------------------------------------------------------------------
	c Run the entire problem once to make sure all data is touched. 
	c This reduces variable startup costs, which is important for such a 
	c short benchmark. The other NPB 2 implementations are similar. 
	c-------------------------------------------------------------------
	*/
	int _ret_val_0;
	#pragma loop name main#0 
	for (i=0; i<7; i ++ )
	{
		timer_clear(i);
	}
	setup();
	{
		compute_indexmap(indexmap, dims[2]);
		{
			compute_initial_conditions(u1, dims[0]);
			fft_init(dims[0][0]);
		}
		fft(1, u1, u0);
	}
	/* end parallel */
	/*
	--------------------------------------------------------------------
	c Start over from the beginning. Note that all operations must
	c be timed, in contrast to other benchmarks. 
	c-------------------------------------------------------------------
	*/
	#pragma loop name main#1 
	for (i=0; i<7; i ++ )
	{
		timer_clear(i);
	}
	timer_start(0);
	if (0==1)
	{
		timer_start(1);
	}
	{
		compute_indexmap(indexmap, dims[2]);
		{
			compute_initial_conditions(u1, dims[0]);
			fft_init(dims[0][0]);
		}
		if (0==1)
		{
			timer_stop(1);
		}
		if (0==1)
		{
			timer_start(2);
		}
		fft(1, u1, u0);
		if (0==1)
		{
			timer_stop(2);
		}
		#pragma loop name main#2 
		for (iter=1; iter<=niter; iter ++ )
		{
			if (0==1)
			{
				timer_start(3);
			}
			evolve(u0, u1, iter, indexmap, dims[0]);
			if (0==1)
			{
				timer_stop(3);
			}
			if (0==1)
			{
				timer_start(2);
			}
			fft( - 1, u1, u2);
			if (0==1)
			{
				timer_stop(2);
			}
			if (0==1)
			{
				timer_start(4);
			}
			checksum(iter, u2, dims[0]);
			if (0==1)
			{
				timer_stop(4);
			}
		}
		verify(NX, NY, NZ, niter,  & verified,  & class);
	}
	/* end parallel */
	timer_stop(0);
	total_time=timer_read(0);
	if (total_time!=0.0)
	{
		mflops=(((1.0E-6*((double)NTOTAL))*((14.8157+(7.19641*log((double)NTOTAL)))+((5.23518+(7.21113*log((double)NTOTAL)))*niter)))/total_time);
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("FT", class, NX, NY, NZ, niter, nthreads, total_time, mflops, "          floating point", verified, "2.3", "24 Jun 2024", "gcc", "gcc", "(none)", "-I../common", "-O3 ", "(none)", "randdp");
	if (0==1)
	{
		print_timers();
	}
	return _ret_val_0;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void evolve(dcomplex u0[NZ][NY][NX], dcomplex u1[NZ][NY][NX], int t, int indexmap[NZ][NY][NX], int d[3])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c evolve u0 -> u1 (t time steps) in fourier space
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	#pragma loop name evolve#0 
	for (k=0; k<d[2]; k ++ )
	{
		#pragma loop name evolve#0#0 
		for (j=0; j<d[1]; j ++ )
		{
			#pragma loop name evolve#0#0#0 
			for (i=0; i<d[0]; i ++ )
			{
				((u1[k][j][i].real=(u0[k][j][i].real*ex[t*indexmap[k][j][i]])), (u1[k][j][i].imag=(u0[k][j][i].imag*ex[t*indexmap[k][j][i]])));
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void compute_initial_conditions(dcomplex u0[NZ][NY][NX], int d[3])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c Fill in array u0 with initial conditions from 
	c random number generator 
	c-------------------------------------------------------------------
	*/
	int k;
	double x0, start, an, dummy;
	static double tmp[(((NX*2)*MAXDIM)+1)];
	int i, j, t;
	start=3.14159265E8;
	/*
	--------------------------------------------------------------------
	c Jump to the starting element for our first plane.
	c-------------------------------------------------------------------
	*/
	ipow46(1.220703125E9, ((((zstart[0]-1)*2)*NX)*NY)+(((ystart[0]-1)*2)*NX),  & an);
	dummy=randlc( & start, an);
	ipow46(1.220703125E9, (2*NX)*NY,  & an);
	/*
	--------------------------------------------------------------------
	c Go through by z planes filling in one square at a time.
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_initial_conditions#0 
	for (k=0; k<dims[0][2]; k ++ )
	{
		x0=start;
		vranlc((2*NX)*dims[0][1],  & x0, 1.220703125E9, tmp);
		t=1;
		#pragma loop name compute_initial_conditions#0#0 
		for (j=0; j<dims[0][1]; j ++ )
		{
			#pragma loop name compute_initial_conditions#0#0#0 
			for (i=0; i<NX; i ++ )
			{
				u0[k][j][i].real=tmp[t ++ ];
				u0[k][j][i].imag=tmp[t ++ ];
			}
		}
		if (k!=dims[0][2])
		{
			dummy=randlc( & start, an);
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void ipow46(double a, int exponent, double * result)
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c compute a^exponent mod 2^46
	c-------------------------------------------------------------------
	*/
	double dummy, q, r;
	int n, n2;
	/*
	--------------------------------------------------------------------
	c Use
	c   a^n = a^(n2)*a^(n/2) if n even else
	c   a^n = a*a^(n-1)       if n odd
	c-------------------------------------------------------------------
	*/
	( * result)=1;
	if (exponent==0)
	{
		return ;
	}
	q=a;
	r=1;
	n=exponent;
	while (n>1)
	{
		n2=(n/2);
		if ((n2*2)==n)
		{
			dummy=randlc( & q, q);
			n=n2;
		}
		else
		{
			dummy=randlc( & r, q);
			n=(n-1);
		}
	}
	dummy=randlc( & r, q);
	( * result)=r;
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void setup(void )
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int ierr, i, j, fstatus;
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - FT Benchmark\n\n");
	niter=200;
	printf(" Size                : %3dx%3dx%3d\n", NX, NY, NZ);
	printf(" Iterations          :     %7d\n", niter);
	/*
	1004 format(' Number of processes :     ', i7)
	 1005 format(' Processor array     :     ', i3, 'x', i3)
	 1006 format(' WARNING: compiled for ', i5, ' processes. ',
	     >       ' Will not verify. ')
	*/
	#pragma loop name setup#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (i=0; i<3; i ++ )
	{
		dims[i][0]=NX;
		dims[i][1]=NY;
		dims[i][2]=NZ;
	}
	#pragma loop name setup#1 
	for (i=0; i<3; i ++ )
	{
		xstart[i]=1;
		xend[i]=NX;
		ystart[i]=1;
		yend[i]=NY;
		zstart[i]=1;
		zend[i]=NZ;
	}
	/*
	--------------------------------------------------------------------
	c Set up info for blocking of ffts and transposes.  This improves
	c performance on cache-based systems. Blocking involves
	c working on a chunk of the problem at a time, taking chunks
	c along the first, second, or third dimension. 
	c
	c - In cffts1 blocking is on 2nd dimension (with fft on 1st dim)
	c - In cffts23 blocking is on 1st dimension (with fft on 2nd and 3rd dims)
	
	c Since 1st dim is always in processor, we'll assume it's long enough 
	c (default blocking factor is 16 so min size for 1st dim is 16)
	c The only case we have to worry about is cffts1 in a 2d decomposition. 
	c so the blocking factor should not be larger than the 2nd dimension. 
	c-------------------------------------------------------------------
	*/
	fftblock=16;
	fftblockpad=18;
	if (fftblock!=16)
	{
		fftblockpad=(fftblock+3);
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void compute_indexmap(int indexmap[NZ][NY][NX], int d[3])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c compute function from local (i,j,k) to ibar^2+jbar^2+kbar^2 
	c for time evolution exponent. 
	c-------------------------------------------------------------------
	*/
	int i, j, k, ii, ii2, jj, ij2, kk;
	double ap;
	/*
	--------------------------------------------------------------------
	c basically we want to convert the fortran indices 
	c   1 2 3 4 5 6 7 8 
	c to 
	c   0 1 2 3 -4 -3 -2 -1
	c The following magic formula does the trick:
	c mod(i-1+n2, n) - n/2
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_indexmap#0 
	for (i=0; i<dims[2][0]; i ++ )
	{
		ii=((((((i+1)+xstart[2])-2)+(NX/2))%NX)-(NX/2));
		ii2=(ii*ii);
		#pragma loop name compute_indexmap#0#0 
		for (j=0; j<dims[2][1]; j ++ )
		{
			jj=((((((j+1)+ystart[2])-2)+(NY/2))%NY)-(NY/2));
			ij2=((jj*jj)+ii2);
			#pragma loop name compute_indexmap#0#0#0 
			for (k=0; k<dims[2][2]; k ++ )
			{
				kk=((((((k+1)+zstart[2])-2)+(NZ/2))%NZ)-(NZ/2));
				indexmap[k][j][i]=((kk*kk)+ij2);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c compute array of exponentials for time evolution. 
	c-------------------------------------------------------------------
	*/
	{
		ap=(((( - 4.0)*1.0E-6)*3.141592653589793)*3.141592653589793);
		ex[0]=1.0;
		ex[1]=exp(ap);
		#pragma loop name compute_indexmap#1 
		for (i=2; i<=(200*((((NX*NX)/4)+((NY*NY)/4))+((NZ*NZ)/4))); i ++ )
		{
			ex[i]=(ex[i-1]*ex[1]);
		}
	}
	/* end single */
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void print_timers(void )
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int i;
	char * tstrings[];
	#pragma loop name print_timers#0 
	for (i=0; i<7; i ++ )
	{
		if (timer_read(i)!=0.0)
		{
			printf("timer %2d(%16s( :%10.6f\n", i, tstrings[i], timer_read(i));
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void fft(int dir, dcomplex x1[NZ][NY][NX], dcomplex x2[NZ][NY][NX])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	dcomplex y0[NX][18];
	dcomplex y1[NX][18];
	/*
	--------------------------------------------------------------------
	c note: args x1, x2 must be different arrays
	c note: args for cfftsx are (direction, layout, xin, xout, scratch)
	c       xinxout may be the same and it can be somewhat faster
	c       if they are
	c-------------------------------------------------------------------
	*/
	if (dir==1)
	{
		cffts1(1, dims[0], x1, x1, y0, y1);
		/* x1 -> x1 */
		cffts2(1, dims[1], x1, x1, y0, y1);
		/* x1 -> x1 */
		cffts3(1, dims[2], x1, x2, y0, y1);
		/* x1 -> x2 */
	}
	else
	{
		cffts3( - 1, dims[2], x1, x1, y0, y1);
		/* x1 -> x1 */
		cffts2( - 1, dims[1], x1, x1, y0, y1);
		/* x1 -> x1 */
		cffts1( - 1, dims[0], x1, x2, y0, y1);
		/* x1 -> x2 */
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void cffts1(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int logd[3];
	int i, j, k, jj;
	#pragma loop name cffts1#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	#pragma loop name cffts1#1 
	for (k=0; k<d[2]; k ++ )
	{
		#pragma loop name cffts1#1#0 
		for (jj=0; jj<=(d[1]-fftblock); jj+=fftblock)
		{
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts1#1#0#0 
			for (j=0; j<fftblock; j ++ )
			{
				#pragma loop name cffts1#1#0#0#0 
				for (i=0; i<d[0]; i ++ )
				{
					y0[i][j].real=x[k][j+jj][i].real;
					y0[i][j].imag=x[k][j+jj][i].imag;
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz(is, logd[0], d[0], y0, y1);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts1#1#0#1 
			for (j=0; j<fftblock; j ++ )
			{
				#pragma loop name cffts1#1#0#1#0 
				for (i=0; i<d[0]; i ++ )
				{
					xout[k][j+jj][i].real=y0[i][j].real;
					xout[k][j+jj][i].imag=y0[i][j].imag;
				}
			}
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void cffts2(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int logd[3];
	int i, j, k, ii;
	#pragma loop name cffts2#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	#pragma loop name cffts2#1 
	for (k=0; k<d[2]; k ++ )
	{
		#pragma loop name cffts2#1#0 
		for (ii=0; ii<=(d[0]-fftblock); ii+=fftblock)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts2#1#0#0 
			for (j=0; j<d[1]; j ++ )
			{
				#pragma loop name cffts2#1#0#0#0 
				for (i=0; i<fftblock; i ++ )
				{
					y0[j][i].real=x[k][j][i+ii].real;
					y0[j][i].imag=x[k][j][i+ii].imag;
				}
			}
			/* 	    if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz(is, logd[1], d[1], y0, y1);
			/*          if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*          if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts2#1#0#1 
			for (j=0; j<d[1]; j ++ )
			{
				#pragma loop name cffts2#1#0#1#0 
				for (i=0; i<fftblock; i ++ )
				{
					xout[k][j][i+ii].real=y0[j][i].real;
					xout[k][j][i+ii].imag=y0[j][i].imag;
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void cffts3(int is, int d[3], dcomplex x[NZ][NY][NX], dcomplex xout[NZ][NY][NX], dcomplex y0[NX][18], dcomplex y1[NX][18])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int logd[3];
	int i, j, k, ii;
	#pragma loop name cffts3#0 
	for (i=0; i<3; i ++ )
	{
		logd[i]=ilog2(d[i]);
	}
	#pragma loop name cffts3#1 
	for (j=0; j<d[1]; j ++ )
	{
		#pragma loop name cffts3#1#0 
		for (ii=0; ii<=(d[0]-fftblock); ii+=fftblock)
		{
			/* 	    if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts3#1#0#0 
			for (k=0; k<d[2]; k ++ )
			{
				#pragma loop name cffts3#1#0#0#0 
				for (i=0; i<fftblock; i ++ )
				{
					y0[k][i].real=x[k][j][i+ii].real;
					y0[k][i].imag=x[k][j][i+ii].imag;
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTLOW); */
			cfftz(is, logd[2], d[2], y0, y1);
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTLOW); */
			/*           if (TIMERS_ENABLED == TRUE) timer_start(T_FFTCOPY); */
			#pragma loop name cffts3#1#0#1 
			for (k=0; k<d[2]; k ++ )
			{
				#pragma loop name cffts3#1#0#1#0 
				for (i=0; i<fftblock; i ++ )
				{
					xout[k][j][i+ii].real=y0[k][i].real;
					xout[k][j][i+ii].imag=y0[k][i].imag;
				}
			}
			/*           if (TIMERS_ENABLED == TRUE) timer_stop(T_FFTCOPY); */
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void fft_init(int n)
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c compute the roots-of-unity array that will be used for subsequent FFTs. 
	c-------------------------------------------------------------------
	*/
	int m, nu, ku, i, j, ln;
	double t, ti;
	/*
	--------------------------------------------------------------------
	c   Initialize the U array with sines and cosines in a manner that permits
	c   stride one access at each FFT iteration.
	c-------------------------------------------------------------------
	*/
	nu=n;
	m=ilog2(n);
	u[0].real=((double)m);
	u[0].imag=0.0;
	ku=1;
	ln=1;
	#pragma loop name fft_init#0 
	for (j=1; j<=m; j ++ )
	{
		t=(3.141592653589793/ln);
		#pragma loop name fft_init#0#0 
		for (i=0; i<=(ln-1); i ++ )
		{
			ti=(i*t);
			u[i+ku].real=cos(ti);
			u[i+ku].imag=sin(ti);
		}
		ku=(ku+ln);
		ln=(2*ln);
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void cfftz(int is, int m, int n, dcomplex x[NX][18], dcomplex y[NX][18])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   Computes NY N-point complex-to-complex FFTs of X using an algorithm due
	c   to Swarztrauber.  X is both the input and the output array, while Y is a 
	c   scratch array.  It is assumed that N = 2^M.  Before calling CFFTZ to 
	c   perform FFTs, the array U must be initialized by calling CFFTZ with IS 
	c   set to 0 and M set to MX, where MX is the maximum value of M for any 
	c   subsequent call.
	c-------------------------------------------------------------------
	*/
	int i, j, l, mx;
	/*
	--------------------------------------------------------------------
	c   Check if input parameters are invalid.
	c-------------------------------------------------------------------
	*/
	mx=((int)u[0].real);
	if ((((is!=1)&&(is!=( - 1)))||(m<1))||(m>mx))
	{
		printf("CFFTZ: Either U has not been initialized, or else\n""one of the input parameters is invalid%5d%5d%5d\n", is, m, mx);
		exit(1);
	}
	/*
	--------------------------------------------------------------------
	c   Perform one variant of the Stockham FFT.
	c-------------------------------------------------------------------
	*/
	#pragma loop name cfftz#0 
	for (l=1; l<=m; l+=2)
	{
		fftz2(is, l, m, n, fftblock, fftblockpad, u, x, y);
		if (l==m)
		{
			break;
		}
		fftz2(is, l+1, m, n, fftblock, fftblockpad, u, y, x);
	}
	/*
	--------------------------------------------------------------------
	c   Copy Y to X.
	c-------------------------------------------------------------------
	*/
	if ((m%2)==1)
	{
		#pragma loop name cfftz#1 
		for (j=0; j<n; j ++ )
		{
			#pragma loop name cfftz#1#0 
			for (i=0; i<fftblock; i ++ )
			{
				x[j][i].real=y[j][i].real;
				x[j][i].imag=y[j][i].imag;
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void fftz2(int is, int l, int m, int n, int ny, int ny1, dcomplex u[NX], dcomplex x[NX][18], dcomplex y[NX][18])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   Performs the L-th iteration of the second variant of the Stockham FFT.
	c-------------------------------------------------------------------
	*/
	int k, n1, li, lj, lk, ku, i, j, i11, i12, i21, i22;
	dcomplex u1, x11, x21;
	/*
	--------------------------------------------------------------------
	c   Set initial parameters.
	c-------------------------------------------------------------------
	*/
	n1=(n/2);
	if ((l-1)==0)
	{
		lk=1;
	}
	else
	{
		lk=(2<<((l-1)-1));
	}
	if ((m-l)==0)
	{
		li=1;
	}
	else
	{
		li=(2<<((m-l)-1));
	}
	lj=(2*lk);
	ku=li;
	#pragma loop name fftz2#0 
	for (i=0; i<li; i ++ )
	{
		i11=(i*lk);
		i12=(i11+n1);
		i21=(i*lj);
		i22=(i21+lk);
		if (is>=1)
		{
			u1.real=u[ku+i].real;
			u1.imag=u[ku+i].imag;
		}
		else
		{
			u1.real=u[ku+i].real;
			u1.imag=( - u[ku+i].imag);
		}
		/*
		--------------------------------------------------------------------
		c   This loop is vectorizable.
		c-------------------------------------------------------------------
		*/
		#pragma loop name fftz2#0#0 
		for (k=0; k<lk; k ++ )
		{
			#pragma loop name fftz2#0#0#0 
			for (j=0; j<ny; j ++ )
			{
				double x11real, x11imag;
				double x21real, x21imag;
				x11real=x[i11+k][j].real;
				x11imag=x[i11+k][j].imag;
				x21real=x[i12+k][j].real;
				x21imag=x[i12+k][j].imag;
				y[i21+k][j].real=(x11real+x21real);
				y[i21+k][j].imag=(x11imag+x21imag);
				y[i22+k][j].real=((u1.real*(x11real-x21real))-(u1.imag*(x11imag-x21imag)));
				y[i22+k][j].imag=((u1.real*(x11imag-x21imag))+(u1.imag*(x11real-x21real)));
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static int ilog2(int n)
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int nn, lg;
	int _ret_val_0;
	if (n==1)
	{
		_ret_val_0=0;
		return _ret_val_0;
	}
	lg=1;
	nn=2;
	while (nn<n)
	{
		nn=(nn<<1);
		lg ++ ;
	}
	return lg;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void checksum(int i, dcomplex u1[NZ][NY][NX], int d[3])
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int j, q, r, s, ierr;
	dcomplex chk, allchk;
	chk.real=0.0;
	chk.imag=0.0;
	#pragma loop name checksum#0 
	/* #pragma cetus reduction(+: chk.imag, chk.real)  */
	for (j=1; j<=1024; j ++ )
	{
		q=((j%NX)+1);
		if ((q>=xstart[0])&&(q<=xend[0]))
		{
			r=(((3*j)%NY)+1);
			if ((r>=ystart[0])&&(r<=yend[0]))
			{
				s=(((5*j)%NZ)+1);
				if ((s>=zstart[0])&&(s<=zend[0]))
				{
					((chk.real=(chk.real+u1[s-zstart[0]][r-ystart[0]][q-xstart[0]].real)), (chk.imag=(chk.imag+u1[s-zstart[0]][r-ystart[0]][q-xstart[0]].imag)));
				}
			}
		}
	}
	{
		sums[i].real+=chk.real;
		sums[i].imag+=chk.imag;
	}
	
	{
		/* complex % real */
		sums[i].real=(sums[i].real/((double)NTOTAL));
		sums[i].imag=(sums[i].imag/((double)NTOTAL));
		printf("T = %5d     Checksum = %22.12e %22.12e\n", i, sums[i].real, sums[i].imag);
	}
	return ;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void verify(int d1, int d2, int d3, int nt, boolean * verified, char * class)
{
	/*
	--------------------------------------------------------------------
	c-------------------------------------------------------------------
	*/
	int ierr, size, i;
	double err, epsilon;
	/*
	--------------------------------------------------------------------
	c   Sample size reference checksums
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   Class S size reference checksums
	c-------------------------------------------------------------------
	*/
	double vdata_real_s[(6+1)];
	double vdata_imag_s[(6+1)];
	/*
	--------------------------------------------------------------------
	c   Class W size reference checksums
	c-------------------------------------------------------------------
	*/
	double vdata_real_w[(6+1)];
	double vdata_imag_w[(6+1)];
	/*
	--------------------------------------------------------------------
	c   Class A size reference checksums
	c-------------------------------------------------------------------
	*/
	double vdata_real_a[(6+1)];
	double vdata_imag_a[(6+1)];
	/*
	--------------------------------------------------------------------
	c   Class B size reference checksums
	c-------------------------------------------------------------------
	*/
	double vdata_real_b[(20+1)];
	double vdata_imag_b[(20+1)];
	/*
	--------------------------------------------------------------------
	c   Class C size reference checksums
	c-------------------------------------------------------------------
	*/
	double vdata_real_c[(20+1)];
	double vdata_imag_c[(20+1)];
	epsilon=1.0E-12;
	( * verified)=1;
	( * class)='U';
	if ((((d1==64)&&(d2==64))&&(d3==64))&&(nt==6))
	{
		( * class)='S';
		#pragma loop name verify#0 
		for (i=1; i<=nt; i ++ )
		{
			err=((sums[i].real-vdata_real_s[i])/vdata_real_s[i]);
			if (fabs(err)>epsilon)
			{
				( * verified)=0;
				break;
			}
			err=((sums[i].imag-vdata_imag_s[i])/vdata_imag_s[i]);
			if (fabs(err)>epsilon)
			{
				( * verified)=0;
				break;
			}
		}
	}
	else
	{
		if ((((d1==128)&&(d2==128))&&(d3==32))&&(nt==6))
		{
			( * class)='W';
			#pragma loop name verify#1 
			for (i=1; i<=nt; i ++ )
			{
				err=((sums[i].real-vdata_real_w[i])/vdata_real_w[i]);
				if (fabs(err)>epsilon)
				{
					( * verified)=0;
					break;
				}
				err=((sums[i].imag-vdata_imag_w[i])/vdata_imag_w[i]);
				if (fabs(err)>epsilon)
				{
					( * verified)=0;
					break;
				}
			}
		}
		else
		{
			if ((((d1==256)&&(d2==256))&&(d3==128))&&(nt==6))
			{
				( * class)='A';
				#pragma loop name verify#2 
				for (i=1; i<=nt; i ++ )
				{
					err=((sums[i].real-vdata_real_a[i])/vdata_real_a[i]);
					if (fabs(err)>epsilon)
					{
						( * verified)=0;
						break;
					}
					err=((sums[i].imag-vdata_imag_a[i])/vdata_imag_a[i]);
					if (fabs(err)>epsilon)
					{
						( * verified)=0;
						break;
					}
				}
			}
			else
			{
				if ((((d1==512)&&(d2==256))&&(d3==256))&&(nt==20))
				{
					( * class)='B';
					#pragma loop name verify#3 
					for (i=1; i<=nt; i ++ )
					{
						err=((sums[i].real-vdata_real_b[i])/vdata_real_b[i]);
						if (fabs(err)>epsilon)
						{
							( * verified)=0;
							break;
						}
						err=((sums[i].imag-vdata_imag_b[i])/vdata_imag_b[i]);
						if (fabs(err)>epsilon)
						{
							( * verified)=0;
							break;
						}
					}
				}
				else
				{
					if ((((d1==512)&&(d2==512))&&(d3==512))&&(nt==20))
					{
						( * class)='C';
						#pragma loop name verify#4 
						for (i=1; i<=nt; i ++ )
						{
							err=((sums[i].real-vdata_real_c[i])/vdata_real_c[i]);
							if (fabs(err)>epsilon)
							{
								( * verified)=0;
								break;
							}
							err=((sums[i].imag-vdata_imag_c[i])/vdata_imag_c[i]);
							if (fabs(err)>epsilon)
							{
								( * verified)=0;
								break;
							}
						}
					}
				}
			}
		}
	}
	if (( * class)!='U')
	{
		printf("Result verification successful\n");
	}
	else
	{
		printf("Result verification failed\n");
	}
	printf("class = %1c\n",  * class);
	return ;
}
