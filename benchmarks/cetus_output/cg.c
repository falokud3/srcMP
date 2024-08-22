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
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - CG

  This benchmark is an OpenMP C version of the NPB CG code.
  
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

  Authors: M. Yarrow
           C. Kuszmaul

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------
*/
/*

c---------------------------------------------------------------------
c  Note: please observe that in the routine conj_grad three 
c  implementations of the sparse matrix-vector multiply have
c  been supplied.  The default matrix-vector multiply is not
c  loop unrolled.  The alternate implementations are unrolled
c  to a depth of 2 and unrolled to a depth of 8.  Please
c  experiment with these to find the fastest for your particular
c  architecture.  If reporting timing results, any of these three may
c  be used without penalty.
c---------------------------------------------------------------------

*/
#include "npb-C.h"
#include "npbparams.h"
/* global variables */
/* commonpartit_size */
static int naa;
static int nzz;
static int firstrow;
static int lastrow;
static int firstcol;
static int lastcol;
/* commonmain_int_mem */
static int colidx[((((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)))+1)];
/* colidx[1:NZ] */
static int rowstr[((NA+1)+1)];
/* rowstr[1:NA+1] */
static int iv[(((2*NA)+1)+1)];
/* iv[1:2NA+1] */
static int arow[((((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)))+1)];
/* arow[1:NZ] */
static int acol[((((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)))+1)];
/* acol[1:NZ] */
/* commonmain_flt_mem */
static double v[((NA+1)+1)];
/* v[1:NA+1] */
static double aelt[((((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)))+1)];
/* aelt[1:NZ] */
static double a[((((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)))+1)];
/* a[1:NZ] */
static double x[((NA+2)+1)];
/* x[1:NA+2] */
static double z[((NA+2)+1)];
/* z[1:NA+2] */
static double p[((NA+2)+1)];
/* p[1:NA+2] */
static double q[((NA+2)+1)];
/* q[1:NA+2] */
static double r[((NA+2)+1)];
/* r[1:NA+2] */
static double w[((NA+2)+1)];
/* w[1:NA+2] */
/* commonurando */
static double amult;
static double tran;
/* function declarations */
static void conj_grad(int colidx[], int rowstr[], double x[], double z[], double a[], double p[], double q[], double r[], double w[], double * rnorm);
static void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer, int firstrow, int lastrow, int firstcol, int lastcol, double rcond, int arow[], int acol[], double aelt[], double v[], int iv[], double shift);
static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[], double aelt[], int firstrow, int lastrow, double x[], boolean mark[], int nzloc[], int nnza);
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[]);
static int icnvrt(double x, int ipwr2);
static void vecset(int n, double v[], int iv[], int * nzv, int i, double val);
/*
--------------------------------------------------------------------
      program cg
--------------------------------------------------------------------
*/
int main(int argc, char * * argv)
{
	int i, j, k, it;
	int nthreads = 1;
	double zeta;
	double rnorm;
	double norm_temp11;
	double norm_temp12;
	double t, mflops;
	char class;
	boolean verified;
	double zeta_verify_value, epsilon;
	int _ret_val_0;
	firstrow=1;
	lastrow=NA;
	firstcol=1;
	lastcol=NA;
	if ((((NA==1400)&&(NONZER==7))&&(NITER==15))&&(SHIFT==10.0))
	{
		class='S';
		zeta_verify_value=8.5971775078648;
	}
	else
	{
		if ((((NA==7000)&&(NONZER==8))&&(NITER==15))&&(SHIFT==12.0))
		{
			class='W';
			zeta_verify_value=10.362595087124;
		}
		else
		{
			if ((((NA==14000)&&(NONZER==11))&&(NITER==15))&&(SHIFT==20.0))
			{
				class='A';
				zeta_verify_value=17.130235054029;
			}
			else
			{
				if ((((NA==75000)&&(NONZER==13))&&(NITER==75))&&(SHIFT==60.0))
				{
					class='B';
					zeta_verify_value=22.712745482631;
				}
				else
				{
					if ((((NA==150000)&&(NONZER==15))&&(NITER==75))&&(SHIFT==110.0))
					{
						class='C';
						zeta_verify_value=28.973605592845;
					}
					else
					{
						class='U';
					}
				}
			}
		}
	}
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - CG Benchmark\n");
	printf(" Size: %10d\n", NA);
	printf(" Iterations: %5d\n", NITER);
	naa=NA;
	nzz=(((NA*(NONZER+1))*(NONZER+1))+(NA*(NONZER+2)));
	/*
	--------------------------------------------------------------------
	c  Initialize random number generator
	c-------------------------------------------------------------------
	*/
	tran=3.14159265E8;
	amult=1.220703125E9;
	zeta=randlc( & tran, amult);
	/*
	--------------------------------------------------------------------
	c  
	c-------------------------------------------------------------------
	*/
	makea(naa, nzz, a, colidx, rowstr, NONZER, firstrow, lastrow, firstcol, lastcol, RCOND, arow, acol, aelt, v, iv, SHIFT);
	/*
	---------------------------------------------------------------------
	c  Note: as a result of the above call to makea:
	c        values of j used in indexing rowstr go from 1 --> lastrow-firstrow+1
	c        values of colidx which are col indexes go from firstcol --> lastcol
	c        So:
	c        Shift the col index vals from actual (firstcol --> lastcol ) 
	c        to local, i.e., (1 --> lastcol-firstcol+1)
	c---------------------------------------------------------------------
	*/
	{
		#pragma loop name main#0 
		/* #pragma cetus reduction(+: colidx[k])  */
		for (j=1; j<=((lastrow-firstrow)+1); j ++ )
		{
			#pragma loop name main#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (k=rowstr[j]; k<rowstr[j+1]; k ++ )
			{
				colidx[k]=((colidx[k]-firstcol)+1);
			}
		}
		/*
		--------------------------------------------------------------------
		c  set starting vector to (1, 1, .... 1)
		c-------------------------------------------------------------------
		*/
		#pragma loop name main#1 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (i=1; i<=(NA+1); i ++ )
		{
			x[i]=1.0;
		}
		zeta=0.0;
		/*
		-------------------------------------------------------------------
		c---->
		c  Do one iteration untimed to init all code and data page tables
		c---->                    (then reinit, start timing, to niter its)
		c-------------------------------------------------------------------
		*/
		#pragma loop name main#2 
		for (it=1; it<=1; it ++ )
		{
			/*
			--------------------------------------------------------------------
			c  The call to the conjugate gradient routine:
			c-------------------------------------------------------------------
			*/
			conj_grad(colidx, rowstr, x, z, a, p, q, r, w,  & rnorm);
			/*
			--------------------------------------------------------------------
			c  zeta = shift + 1(x.z)
			c  So, first: (x.z)
			c  Also, find norm of z
			c  So, first: (z.z)
			c-------------------------------------------------------------------
			*/
			{
				norm_temp11=0.0;
				norm_temp12=0.0;
			}
			/* end single */
			#pragma loop name main#2#0 
			#pragma cetus reduction(+: norm_temp11, norm_temp12) 
			#pragma cetus parallel 
			#pragma omp parallel for reduction(+: norm_temp11, norm_temp12)
			for (j=1; j<=((lastcol-firstcol)+1); j ++ )
			{
				norm_temp11=(norm_temp11+(x[j]*z[j]));
				norm_temp12=(norm_temp12+(z[j]*z[j]));
			}
			norm_temp12=(1.0/sqrt(norm_temp12));
			/*
			--------------------------------------------------------------------
			c  Normalize z to obtain x
			c-------------------------------------------------------------------
			*/
			#pragma loop name main#2#1 
			for (j=1; j<=((lastcol-firstcol)+1); j ++ )
			{
				x[j]=(norm_temp12*z[j]);
			}
		}
		/* end of do one iteration untimed */
		/*
		--------------------------------------------------------------------
		c  set starting vector to (1, 1, .... 1)
		c-------------------------------------------------------------------
		*/
		#pragma loop name main#3 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (i=1; i<=(NA+1); i ++ )
		{
			x[i]=1.0;
		}
		zeta=0.0;
	}
	/* end parallel */
	timer_clear(1);
	timer_start(1);
	/*
	--------------------------------------------------------------------
	c---->
	c  Main Iteration for inverse power method
	c---->
	c-------------------------------------------------------------------
	*/
	{
		#pragma loop name main#4 
		for (it=1; it<=NITER; it ++ )
		{
			/*
			--------------------------------------------------------------------
			c  The call to the conjugate gradient routine:
			c-------------------------------------------------------------------
			*/
			conj_grad(colidx, rowstr, x, z, a, p, q, r, w,  & rnorm);
			/*
			--------------------------------------------------------------------
			c  zeta = shift + 1(x.z)
			c  So, first: (x.z)
			c  Also, find norm of z
			c  So, first: (z.z)
			c-------------------------------------------------------------------
			*/
			{
				norm_temp11=0.0;
				norm_temp12=0.0;
			}
			/* end single */
			#pragma loop name main#4#0 
			#pragma cetus reduction(+: norm_temp11, norm_temp12) 
			#pragma cetus parallel 
			#pragma omp parallel for reduction(+: norm_temp11, norm_temp12)
			for (j=1; j<=((lastcol-firstcol)+1); j ++ )
			{
				norm_temp11=(norm_temp11+(x[j]*z[j]));
				norm_temp12=(norm_temp12+(z[j]*z[j]));
			}
			{
				norm_temp12=(1.0/sqrt(norm_temp12));
				zeta=(SHIFT+(1.0/norm_temp11));
			}
			/* end single */
			{
				if (it==1)
				{
					printf("   iteration           ||r||                 zeta\n");
				}
				printf("    %5d       %20.14e%20.13e\n", it, rnorm, zeta);
			}
			/* end master */
			/*
			--------------------------------------------------------------------
			c  Normalize z to obtain x
			c-------------------------------------------------------------------
			*/
			#pragma loop name main#4#1 
			for (j=1; j<=((lastcol-firstcol)+1); j ++ )
			{
				x[j]=(norm_temp12*z[j]);
			}
		}
		/* end of main iter inv pow meth */
	}
	/* end parallel */
	timer_stop(1);
	/*
	--------------------------------------------------------------------
	c  End of timed section
	c-------------------------------------------------------------------
	*/
	t=timer_read(1);
	printf(" Benchmark completed\n");
	epsilon=1.0E-10;
	if (class!='U')
	{
		if (fabs(zeta-zeta_verify_value)<=epsilon)
		{
			verified=1;
			printf(" VERIFICATION SUCCESSFUL\n");
			printf(" Zeta is    %20.12e\n", zeta);
			printf(" Error is   %20.12e\n", zeta-zeta_verify_value);
		}
		else
		{
			verified=0;
			printf(" VERIFICATION FAILED\n");
			printf(" Zeta                %20.12e\n", zeta);
			printf(" The correct zeta is %20.12e\n", zeta_verify_value);
		}
	}
	else
	{
		verified=0;
		printf(" Problem size unknown\n");
		printf(" NO VERIFICATION PERFORMED\n");
	}
	if (t!=0.0)
	{
		mflops=(((((2.0*NITER)*NA)*(((3.0+(NONZER*(NONZER+1)))+(25.0*(5.0+(NONZER*(NONZER+1)))))+3.0))/t)/1000000.0);
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("CG", class, NA, 0, 0, NITER, nthreads, t, mflops, "          floating point", verified, "2.3", "24 Jun 2024", "gcc", "gcc", "(none)", "-I../common", "-O3 ", "(none)", "randdp");
	return _ret_val_0;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
/* colidx[1:nzz] */
/* rowstr[1:naa+1] */
/* x[] */
/* z[] */
/* a[1:nzz] */
/* p[] */
/* q[] */
/* r[] */
/* w[] */
/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
/*
---------------------------------------------------------------------
c  Floaging point arrays here are named as in NPB1 spec discussion of 
c  CG algorithm
c---------------------------------------------------------------------
*/
static void conj_grad(int colidx[], int rowstr[], double x[], double z[], double a[], double p[], double q[], double r[], double w[], double * rnorm)
{
	static double d, sum, rho, rho0, alpha, beta;
	int i, j, k;
	int cgit, cgitmax = 25;
	rho=0.0;
	/*
	--------------------------------------------------------------------
	c  Initialize the CG algorithm:
	c-------------------------------------------------------------------
	*/
	#pragma loop name conj_grad#0 
	for (j=1; j<=(naa+1); j ++ )
	{
		q[j]=0.0;
		z[j]=0.0;
		r[j]=x[j];
		p[j]=r[j];
		w[j]=0.0;
	}
	/*
	--------------------------------------------------------------------
	c  rho = r.r
	c  Now, obtain the norm of r: First, sum squares of r elements locally...
	c-------------------------------------------------------------------
	*/
	#pragma loop name conj_grad#1 
	#pragma cetus reduction(+: rho) 
	#pragma cetus parallel 
	#pragma omp parallel for reduction(+: rho)
	for (j=1; j<=((lastcol-firstcol)+1); j ++ )
	{
		rho=(rho+(x[j]*x[j]));
	}
	/*
	--------------------------------------------------------------------
	c---->
	c  The conj grad iteration loop
	c---->
	c-------------------------------------------------------------------
	*/
	#pragma loop name conj_grad#2 
	for (cgit=1; cgit<=cgitmax; cgit ++ )
	{
		{
			rho0=rho;
			d=0.0;
			rho=0.0;
		}
		/* end single */
		/*
		--------------------------------------------------------------------
		c  q = A.p
		c  The partition submatrix-vector multiply: use workspace w
		c---------------------------------------------------------------------
		C
		C  NOTE: this version of the multiply is actually (slightly: maybe %5) 
		C        faster on the sp2 on 16 nodes than is the unrolled-by-2 version 
		C        below.   On the Cray t3d, the reverse is true, i.e., the 
		C        unrolled-by-two version is some 10% faster.  
		C        The unrolled-by-8 version below is significantly faster
		C        on the Cray t3d - overall speed of code is 1.5 times faster.
		
		*/
		/* rolled version */
		#pragma loop name conj_grad#2#0 
		for (j=1; j<=((lastrow-firstrow)+1); j ++ )
		{
			sum=0.0;
			#pragma loop name conj_grad#2#0#0 
			#pragma cetus reduction(+: sum) 
			#pragma cetus parallel 
			#pragma omp parallel for reduction(+: sum)
			for (k=rowstr[j]; k<rowstr[j+1]; k ++ )
			{
				sum=(sum+(a[k]*p[colidx[k]]));
			}
			w[j]=sum;
		}
		/*
		unrolled-by-two version
		#pragma omp for private(i,k)
		        for (j = 1; j <= lastrow-firstrow+1; j++) {
				    int iresidue;
				    double sum1, sum2;
				    i = rowstr[j]; 
			            iresidue = (rowstr[j+1]-i) % 2;
			            sum1 = 0.0;
			            sum2 = 0.0;
			            if (iresidue == 1) sum1 = sum1 + a[i]p[colidx[i]];
				    for (k = i+iresidue; k <= rowstr[j+1]-2; k += 2) {
						sum1 = sum1 + a[k]   * p[colidx[k]];
						sum2 = sum2 + a[k+1] * p[colidx[k+1]];
				    }
			            w[j] = sum1 + sum2;
		        }
		
		*/
		/*
		unrolled-by-8 version
		#pragma omp for private(i,k,sum)
		        for (j = 1; j <= lastrow-firstrow+1; j++) {
				    int iresidue;
			            i = rowstr[j]; 
			            iresidue = (rowstr[j+1]-i) % 8;
			            sum = 0.0;
			            for (k = i; k <= i+iresidue-1; k++) {
				                sum = sum +  a[k] p[colidx[k]];
			            }
			            for (k = i+iresidue; k <= rowstr[j+1]-8; k += 8) {
				                sum = sum + a[k  ] * p[colidx[k  ]]
				                          + a[k+1] * p[colidx[k+1]]
				                          + a[k+2] * p[colidx[k+2]]
				                          + a[k+3] * p[colidx[k+3]]
				                          + a[k+4] * p[colidx[k+4]]
				                          + a[k+5] * p[colidx[k+5]]
				                          + a[k+6] * p[colidx[k+6]]
				                          + a[k+7] * p[colidx[k+7]];
			            }
			            w[j] = sum;
		        }
		
		*/
		#pragma loop name conj_grad#2#1 
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			q[j]=w[j];
		}
		/*
		--------------------------------------------------------------------
		c  Clear w for reuse...
		c-------------------------------------------------------------------
		*/
		#pragma loop name conj_grad#2#2 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			w[j]=0.0;
		}
		/*
		--------------------------------------------------------------------
		c  Obtain p.q
		c-------------------------------------------------------------------
		*/
		#pragma loop name conj_grad#2#3 
		#pragma cetus reduction(+: d) 
		#pragma cetus parallel 
		#pragma omp parallel for reduction(+: d)
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			d=(d+(p[j]*q[j]));
		}
		/*
		--------------------------------------------------------------------
		c  Obtain alpha = rho (p.q)
		c-------------------------------------------------------------------
		*/
		alpha=(rho0/d);
		/*
		--------------------------------------------------------------------
		c  Save a temporary of rho
		c-------------------------------------------------------------------
		*/
		/* 	rho0 = rho; */
		/*
		---------------------------------------------------------------------
		c  Obtain z = z + alphap
		c  and    r = r - alpha*q
		c---------------------------------------------------------------------
		*/
		#pragma loop name conj_grad#2#4 
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			z[j]=(z[j]+(alpha*p[j]));
			r[j]=(r[j]-(alpha*q[j]));
		}
		/*
		---------------------------------------------------------------------
		c  rho = r.r
		c  Now, obtain the norm of r: First, sum squares of r elements locally...
		c---------------------------------------------------------------------
		*/
		#pragma loop name conj_grad#2#5 
		#pragma cetus reduction(+: rho) 
		#pragma cetus parallel 
		#pragma omp parallel for reduction(+: rho)
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			rho=(rho+(r[j]*r[j]));
		}
		/*
		--------------------------------------------------------------------
		c  Obtain beta:
		c-------------------------------------------------------------------
		*/
		beta=(rho/rho0);
		/*
		--------------------------------------------------------------------
		c  p = r + betap
		c-------------------------------------------------------------------
		*/
		#pragma loop name conj_grad#2#6 
		for (j=1; j<=((lastcol-firstcol)+1); j ++ )
		{
			p[j]=(r[j]+(beta*p[j]));
		}
	}
	/* end of do cgit=1,cgitmax */
	/*
	---------------------------------------------------------------------
	c  Compute residual norm explicitly:  ||r|| = ||x - A.z||
	c  First, form A.z
	c  The partition submatrix-vector multiply
	c---------------------------------------------------------------------
	*/
	sum=0.0;
	#pragma loop name conj_grad#3 
	for (j=1; j<=((lastrow-firstrow)+1); j ++ )
	{
		d=0.0;
		#pragma loop name conj_grad#3#0 
		#pragma cetus reduction(+: d) 
		#pragma cetus parallel 
		#pragma omp parallel for reduction(+: d)
		for (k=rowstr[j]; k<=(rowstr[j+1]-1); k ++ )
		{
			d=(d+(a[k]*z[colidx[k]]));
		}
		w[j]=d;
	}
	#pragma loop name conj_grad#4 
	for (j=1; j<=((lastcol-firstcol)+1); j ++ )
	{
		r[j]=w[j];
	}
	/*
	--------------------------------------------------------------------
	c  At this point, r contains A.z
	c-------------------------------------------------------------------
	*/
	#pragma loop name conj_grad#5 
	/* #pragma cetus reduction(+: sum)  */
	for (j=1; j<=((lastcol-firstcol)+1); j ++ )
	{
		d=(x[j]-r[j]);
		sum=(sum+(d*d));
	}
	{
		( * rnorm)=sqrt(sum);
	}
	/* end single */
	return ;
}

/*
---------------------------------------------------------------------
c       generate the test problem for benchmark 6
c       makea generates a sparse matrix with a
c       prescribed sparsity distribution
c
c       parameter    type        usage
c
c       input
c
c       n            i           number of colsrows of matrix
c       nz           i           nonzeros as declared array size
c       rcond        r*8         condition number
c       shift        r*8         main diagonal shift
c
c       output
c
c       a            r*8         array for nonzeros
c       colidx       i           col indices
c       rowstr       i           row pointers
c
c       workspace
c
c       iv, arow, acol i
c       v, aelt        r*8
c---------------------------------------------------------------------
*/
/* a[1:nz] */
/* colidx[1:nz] */
/* rowstr[1:n+1] */
/* arow[1:nz] */
/* acol[1:nz] */
/* aelt[1:nz] */
/* v[1:n+1] */
/* iv[1:2n+1] */
static void makea(int n, int nz, double a[], int colidx[], int rowstr[], int nonzer, int firstrow, int lastrow, int firstcol, int lastcol, double rcond, int arow[], int acol[], double aelt[], double v[], int iv[], double shift)
{
	int i, nnza, iouter, ivelt, ivelt1, irow, nzv;
	/*
	--------------------------------------------------------------------
	c      nonzer is approximately  (int(sqrt(nnzan)));
	c-------------------------------------------------------------------
	*/
	double size, ratio, scale;
	int jcol;
	size=1.0;
	ratio=pow(rcond, 1.0/((double)n));
	nnza=0;
	/*
	---------------------------------------------------------------------
	c  Initialize colidx(n+1 .. 2n) to zero.
	c  Used by sprnvc to mark nonzero positions
	c---------------------------------------------------------------------
	*/
	#pragma loop name makea#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (i=1; i<=n; i ++ )
	{
		colidx[n+i]=0;
	}
	#pragma loop name makea#1 
	for (iouter=1; iouter<=n; iouter ++ )
	{
		nzv=nonzer;
		sprnvc(n, nzv, v, iv,  & colidx[0],  & colidx[n]);
		vecset(n, v, iv,  & nzv, iouter, 0.5);
		#pragma loop name makea#1#0 
		for (ivelt=1; ivelt<=nzv; ivelt ++ )
		{
			jcol=iv[ivelt];
			if ((jcol>=firstcol)&&(jcol<=lastcol))
			{
				scale=(size*v[ivelt]);
				#pragma loop name makea#1#0#0 
				for (ivelt1=1; ivelt1<=nzv; ivelt1 ++ )
				{
					irow=iv[ivelt1];
					if ((irow>=firstrow)&&(irow<=lastrow))
					{
						nnza=(nnza+1);
						if (nnza>nz)
						{
							printf("Space for matrix elements exceeded in"" makea\n");
							printf("nnza, nzmax = %d, %d\n", nnza, nz);
							printf("iouter = %d\n", iouter);
							exit(1);
						}
						acol[nnza]=jcol;
						arow[nnza]=irow;
						aelt[nnza]=(v[ivelt1]*scale);
					}
				}
			}
		}
		size=(size*ratio);
	}
	/*
	---------------------------------------------------------------------
	c       ... add the identity rcond to the generated matrix to bound
	c           the smallest eigenvalue from below by rcond
	c---------------------------------------------------------------------
	*/
	#pragma loop name makea#2 
	for (i=firstrow; i<=lastrow; i ++ )
	{
		if ((i>=firstcol)&&(i<=lastcol))
		{
			iouter=(n+i);
			nnza=(nnza+1);
			if (nnza>nz)
			{
				printf("Space for matrix elements exceeded in makea\n");
				printf("nnza, nzmax = %d, %d\n", nnza, nz);
				printf("iouter = %d\n", iouter);
				exit(1);
			}
			acol[nnza]=i;
			arow[nnza]=i;
			aelt[nnza]=(rcond-shift);
		}
	}
	/*
	---------------------------------------------------------------------
	c       ... make the sparse matrix from list of elements with duplicates
	c           (v and iv are used as  workspace)
	c---------------------------------------------------------------------
	*/
	sparse(a, colidx, rowstr, n, arow, acol, aelt, firstrow, lastrow, v,  & iv[0],  & iv[n], nnza);
	return ;
}

/*
---------------------------------------------------
c       generate a sparse matrix from a list of
c       [col, row, element] tri
c---------------------------------------------------
*/
/* a[1:] */
/* colidx[1:] */
/* rowstr[1:] */
/* arow[1:] */
/* acol[1:] */
/* aelt[1:] */
/* x[1:n] */
/* mark[1:n] */
/* nzloc[1:n] */
/*
---------------------------------------------------------------------
c       rows range from firstrow to lastrow
c       the rowstr pointers are defined for nrows = lastrow-firstrow+1 values
c---------------------------------------------------------------------
*/
static void sparse(double a[], int colidx[], int rowstr[], int n, int arow[], int acol[], double aelt[], int firstrow, int lastrow, double x[], boolean mark[], int nzloc[], int nnza)
{
	int nrows;
	int i, j, jajp1, nza, k, nzrow;
	double xi;
	/*
	--------------------------------------------------------------------
	c    how many rows of result
	c-------------------------------------------------------------------
	*/
	nrows=((lastrow-firstrow)+1);
	/*
	--------------------------------------------------------------------
	c     ...count the number of triples in each row
	c-------------------------------------------------------------------
	*/
	#pragma loop name sparse#0 
	for (j=1; j<=n; j ++ )
	{
		rowstr[j]=0;
		mark[j]=0;
	}
	rowstr[n+1]=0;
	#pragma loop name sparse#1 
	for (nza=1; nza<=nnza; nza ++ )
	{
		j=(((arow[nza]-firstrow)+1)+1);
		rowstr[j]=(rowstr[j]+1);
	}
	rowstr[1]=1;
	#pragma loop name sparse#2 
	for (j=2; j<=(nrows+1); j ++ )
	{
		rowstr[j]=(rowstr[j]+rowstr[j-1]);
	}
	/*
	---------------------------------------------------------------------
	c     ... rowstr(j) now is the location of the first nonzero
	c           of row j of a
	c---------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     ... do a bucket sort of the triples on the row index
	c-------------------------------------------------------------------
	*/
	#pragma loop name sparse#3 
	for (nza=1; nza<=nnza; nza ++ )
	{
		j=((arow[nza]-firstrow)+1);
		k=rowstr[j];
		a[k]=aelt[nza];
		colidx[k]=acol[nza];
		rowstr[j]=(rowstr[j]+1);
	}
	/*
	--------------------------------------------------------------------
	c       ... rowstr(j) now points to the first element of row j+1
	c-------------------------------------------------------------------
	*/
	#pragma loop name sparse#4 
	for (j=nrows; j>=1; j -- )
	{
		rowstr[j+1]=rowstr[j];
	}
	rowstr[1]=1;
	/*
	--------------------------------------------------------------------
	c       ... generate the actual output rows by adding elements
	c-------------------------------------------------------------------
	*/
	nza=0;
	#pragma loop name sparse#5 
	for (i=1; i<=n; i ++ )
	{
		x[i]=0.0;
		mark[i]=0;
	}
	jajp1=rowstr[1];
	#pragma loop name sparse#6 
	for (j=1; j<=nrows; j ++ )
	{
		nzrow=0;
		/*
		--------------------------------------------------------------------
		c          ...loop over the jth row of a
		c-------------------------------------------------------------------
		*/
		#pragma loop name sparse#6#0 
		for (k=jajp1; k<rowstr[j+1]; k ++ )
		{
			i=colidx[k];
			x[i]=(x[i]+a[k]);
			if ((mark[i]==0)&&(x[i]!=0.0))
			{
				mark[i]=1;
				nzrow=(nzrow+1);
				nzloc[nzrow]=i;
			}
		}
		/*
		--------------------------------------------------------------------
		c          ... extract the nonzeros of this row
		c-------------------------------------------------------------------
		*/
		#pragma loop name sparse#6#1 
		for (k=1; k<=nzrow; k ++ )
		{
			i=nzloc[k];
			mark[i]=0;
			xi=x[i];
			x[i]=0.0;
			if (xi!=0.0)
			{
				nza=(nza+1);
				a[nza]=xi;
				colidx[nza]=i;
			}
		}
		jajp1=rowstr[j+1];
		rowstr[j+1]=(nza+rowstr[1]);
	}
	return ;
}

/*
---------------------------------------------------------------------
c       generate a sparse n-vector (v, iv)
c       having nzv nonzeros
c
c       mark(i) is set to 1 if position i is nonzero.
c       mark is all zero on entry and is reset to all zero before exit
c       this corrects a performance bug found by John G. Lewis, caused by
c       reinitialization of mark on every one of the n calls to sprnvc
---------------------------------------------------------------------
*/
/* v[1:] */
/* iv[1:] */
/* nzloc[1:n] */
/* mark[1:n] */
static void sprnvc(int n, int nz, double v[], int iv[], int nzloc[], int mark[])
{
	int nn1;
	int nzrow, nzv, ii, i;
	double vecelt, vecloc;
	nzv=0;
	nzrow=0;
	nn1=1;
	do
	{
		nn1=(2*nn1);
	}while(nn1<n);
	
	/*
	--------------------------------------------------------------------
	c    nn1 is the smallest power of two not less than n
	c-------------------------------------------------------------------
	*/
	while (nzv<nz)
	{
		vecelt=randlc( & tran, amult);
		/*
		--------------------------------------------------------------------
		c   generate an integer between 1 and n in a portable manner
		c-------------------------------------------------------------------
		*/
		vecloc=randlc( & tran, amult);
		i=(icnvrt(vecloc, nn1)+1);
		if (i>n)
		{
			continue;
		}
		/*
		--------------------------------------------------------------------
		c  was this integer generated already?
		c-------------------------------------------------------------------
		*/
		if (mark[i]==0)
		{
			mark[i]=1;
			nzrow=(nzrow+1);
			nzloc[nzrow]=i;
			nzv=(nzv+1);
			v[nzv]=vecelt;
			iv[nzv]=i;
		}
	}
	#pragma loop name sprnvc#0 
	for (ii=1; ii<=nzrow; ii ++ )
	{
		i=nzloc[ii];
		mark[i]=0;
	}
	return ;
}

/*
---------------------------------------------------------------------
 scale a double precision number x in (0,1) by a power of 2 and chop it
*---------------------------------------------------------------------
*/
static int icnvrt(double x, int ipwr2)
{
	int _ret_val_0;
	_ret_val_0=((int)(ipwr2*x));
	return _ret_val_0;
}

/*
--------------------------------------------------------------------
c       set ith element of sparse vector (v, iv) with
c       nzv nonzeros to val
c-------------------------------------------------------------------
*/
/* v[1:] */
/* iv[1:] */
static void vecset(int n, double v[], int iv[], int * nzv, int i, double val)
{
	int k;
	boolean set;
	set=0;
	#pragma loop name vecset#0 
	for (k=1; k<=( * nzv); k ++ )
	{
		if (iv[k]==i)
		{
			v[k]=val;
			set=1;
		}
	}
	if (set==0)
	{
		( * nzv)=(( * nzv)+1);
		v[ * nzv]=val;
		iv[ * nzv]=i;
	}
	return ;
}
