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
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - EP

  This benchmark is an OpenMP C version of the NPB EP code.
  
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

  Author: P. O. Frederickson 
          D. H. Bailey
          A. C. Woo

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------
*/
#include "npb-C.h"
#include "npbparams.h"
/* parameters */
/* global variables */
/* commonstorage */
static double x[(2*(1<<16))];

static double q[10];
/*
--------------------------------------------------------------------
      program EMBAR
c-------------------------------------------------------------------
*/
/*

c   This is the serial version of the APP Benchmark 1,
c   the "embarassingly parallel" benchmark.
c
c   M is the Log_2 of the number of complex pairs of uniform (0, 1) random
c   numbers.  MK is the Log_2 of the size of each batch of uniform random
c   numbers.  MK can be set for convenience on a given system, since it does
c   not affect the results.

*/
int main(int argc, char * * argv)
{
	double Mops, t1, t2, t3, t4, x1, x2, sx, sy, tm, an, tt, gc;
	double dum[3];
	int np, ierr, node, no_nodes, i, ik, kk, l, k, nit, ierrcode, no_large_nodes, np_add, k_offset, j;
	int nthreads = 1;
	boolean verified;
	char size[(13+1)];
	/* character13 */
	/*
	
	c   Because the size of the problem is too large to store in a 32-bit
	c   integer for some classes, we put it into a string (for printing).
	c   Have to strip off the decimal point put in there by the floating
	c   point print statement (internal file)
	
	*/
	int _ret_val_0;
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - EP Benchmark\n");
	sprintf(size, "%12.0f", pow(2.0, 24+1));
	#pragma loop name main#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (j=13; j>=1; j -- )
	{
		if (size[j]=='.')
		{
			size[j]=' ';
		}
	}
	printf(" Number of random numbers generated: %13s\n", size);
	verified=0;
	/*
	
	c   Compute the number of "batches" of random number pairs generated 
	c   per processor. Adjust if the number of processors does not evenly 
	c   divide the total number
	
	*/
	np=(1<<(24-16));
	/*
	
	c   Call the random number generator functions and initialize
	c   the x-array to reduce the effects of paging on the timings.
	c   Also, call all mathematical functions that are used. Make
	c   sure these initializations cannot be eliminated as dead code.
	
	*/
	vranlc(0,  & dum[0], dum[1],  & dum[2]);
	dum[0]=randlc( & dum[1], dum[2]);
	#pragma loop name main#1 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (i=0; i<(2*(1<<16)); i ++ )
	{
		x[i]=( - 1.0E99);
	}
	Mops=log(sqrt(fabs(((1.0>1.0) ? 1.0 : 1.0))));
	timer_clear(1);
	timer_clear(2);
	timer_clear(3);
	timer_start(1);
	vranlc(0,  & t1, 1.220703125E9, x);
	/*   Compute AN = A ^ (2 NK) (mod 2^46). */
	t1=1.220703125E9;
	#pragma loop name main#2 
	for (i=1; i<=(16+1); i ++ )
	{
		t2=randlc( & t1, t1);
	}
	an=t1;
	tt=2.71828183E8;
	gc=0.0;
	sx=0.0;
	sy=0.0;
	#pragma loop name main#3 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (i=0; i<=(10-1); i ++ )
	{
		q[i]=0.0;
	}
	/*
	
	c   Each instance of this loop may be performed independently. We compute
	c   the k offsets separately to take into account the fact that some nodes
	c   have more numbers to generate than others
	
	*/
	k_offset=( - 1);
	{
		double t1, t2, t3, t4, x1, x2;
		int kk, i, ik, l;
		double qq[10];
		/* private copy of q[0:NQ-1] */
		#pragma loop name main#4 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (i=0; i<10; i ++ )
		{
			qq[i]=0.0;
		}
		#pragma loop name main#5 
		/* #pragma cetus reduction(+: qq[l], sx, sy)  */
		for (k=1; k<=np; k ++ )
		{
			kk=(k_offset+k);
			t1=2.71828183E8;
			t2=an;
			/*      Find starting seed t1 for this kk. */
			#pragma loop name main#5#0 
			for (i=1; i<=100; i ++ )
			{
				ik=(kk/2);
				if ((2*ik)!=kk)
				{
					t3=randlc( & t1, t2);
				}
				if (ik==0)
				{
					break;
				}
				t3=randlc( & t2, t2);
				kk=ik;
			}
			/*      Compute uniform pseudorandom numbers. */
			if (0==1)
			{
				timer_start(3);
			}
			vranlc(2*(1<<16),  & t1, 1.220703125E9, x-1);
			if (0==1)
			{
				timer_stop(3);
			}
			/*
			
			c       Compute Gaussian deviates by acceptance-rejection method and 
			c       tally counts in concentric square annuli.  This loop is not 
			c       vectorizable.
			
			*/
			if (0==1)
			{
				timer_start(2);
			}
			#pragma loop name main#5#1 
			/* #pragma cetus reduction(+: qq[l], sx, sy)  */
			for (i=0; i<(1<<16); i ++ )
			{
				x1=((2.0*x[2*i])-1.0);
				x2=((2.0*x[(2*i)+1])-1.0);
				t1=((x1*x1)+(x2*x2));
				if (t1<=1.0)
				{
					t2=sqrt((( - 2.0)*log(t1))/t1);
					t3=(x1*t2);
					/* Xi */
					t4=(x2*t2);
					/* Yi */
					l=((fabs(t3)>fabs(t4)) ? fabs(t3) : fabs(t4));
					qq[l]+=1.0;
					/* counts */
					sx=(sx+t3);
					/* sum of Xi */
					sy=(sy+t4);
					/* sum of Yi */
				}
			}
			if (0==1)
			{
				timer_stop(2);
			}
		}
		{
			#pragma loop name main#6 
			for (i=0; i<=(10-1); i ++ )
			{
				q[i]+=qq[i];
			}
		}
	}
	/* end of parallel region */
	#pragma loop name main#7 
	#pragma cetus reduction(+: gc) 
	#pragma cetus parallel 
	#pragma omp parallel for reduction(+: gc)
	for (i=0; i<=(10-1); i ++ )
	{
		gc=(gc+q[i]);
	}
	timer_stop(1);
	tm=timer_read(1);
	nit=0;
	if (24==24)
	{
		if ((fabs((sx-( - 3247.83465203474))/sx)<=1.0E-8)&&(fabs((sy-( - 6958.407078382297))/sy)<=1.0E-8))
		{
			verified=1;
		}
	}
	else
	{
		if (24==25)
		{
			if ((fabs((sx-( - 2863.319731645753))/sx)<=1.0E-8)&&(fabs((sy-( - 6320.053679109499))/sy)<=1.0E-8))
			{
				verified=1;
			}
		}
		else
		{
			if (24==28)
			{
				if ((fabs((sx-( - 4295.875165629892))/sx)<=1.0E-8)&&(fabs((sy-( - 15807.32573678431))/sy)<=1.0E-8))
				{
					verified=1;
				}
			}
			else
			{
				if (24==30)
				{
					if ((fabs((sx-40338.15542441498)/sx)<=1.0E-8)&&(fabs((sy-( - 26606.69192809235))/sy)<=1.0E-8))
					{
						verified=1;
					}
				}
				else
				{
					if (24==32)
					{
						if ((fabs((sx-47643.67927995374)/sx)<=1.0E-8)&&(fabs((sy-( - 80840.72988043731))/sy)<=1.0E-8))
						{
							verified=1;
						}
					}
				}
			}
		}
	}
	Mops=((pow(2.0, 24+1)/tm)/1000000.0);
	printf("EP Benchmark Results: \n""CPU Time = %10.4f\n""N = 2^%5d\n""No. Gaussian Pairs = %15.0f\n""Sums = %25.15e %25.15e\n""Counts:\n", tm, 24, gc, sx, sy);
	#pragma loop name main#8 
	for (i=0; i<=(10-1); i ++ )
	{
		printf("%3d %15.0f\n", i, q[i]);
	}
	c_print_results("EP", 'S', 24+1, 0, 0, nit, nthreads, tm, Mops, "Random numbers generated", verified, "2.3", "24 Jun 2024", "gcc", "gcc", "(none)", "-I../common", "-O3 ", "(none)", "randdp");
	if (0==1)
	{
		printf("Total time:     %f", timer_read(1));
		printf("Gaussian pairs: %f", timer_read(2));
		printf("Random numbers: %f", timer_read(3));
	}
	return _ret_val_0;
}
