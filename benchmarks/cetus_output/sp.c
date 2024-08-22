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
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - SP

  This benchmark is an OpenMP C version of the NPB SP code.
  
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

  Author: R. Van der Wijngaart
          W. Saphir

  OpenMP C version: S. Satoh
  
--------------------------------------------------------------------
*/
#include "npb-C.h"
/* global variables */
#include "header.h"
/* function declarations */
static void add(void );
static void adi(void );
static void error_norm(double rms[5]);
static void rhs_norm(double rms[5]);
static void exact_rhs(void );
static void exact_solution(double xi, double eta, double zeta, double dtemp[5]);
static void initialize(void );
static void lhsinit(void );
static void lhsx(void );
static void lhsy(void );
static void lhsz(void );
static void ninvr(void );
static void pinvr(void );
static void compute_rhs(void );
static void set_constants(void );
static void txinvr(void );
static void tzetar(void );
static void verify(int no_time_steps, char * class, boolean * verified);
static void x_solve(void );
static void y_solve(void );
static void z_solve(void );
/*
--------------------------------------------------------------------
       program SP
c-------------------------------------------------------------------
*/
int main(int argc, char * * argv)
{
	int niter, step;
	double mflops, tmax;
	int nthreads = 1;
	boolean verified;
	char class;
	FILE * fp;
	/*
	--------------------------------------------------------------------
	c      Read input file (if it exists), else take
	c      defaults from parameters
	c-------------------------------------------------------------------
	*/
	int _ret_val_0;
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - SP Benchmark\n\n");
	fp=fopen("inputsp.data", "r");
	if (fp!=((void * )0))
	{
		printf(" Reading from input file inputsp.data\n");
		fscanf(fp, "%d",  & niter);
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%lf",  & dt);
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%d%d%d",  & grid_points[0],  & grid_points[1],  & grid_points[2]);
		fclose(fp);
	}
	else
	{
		printf(" No input file inputsp.data. Using compiled defaults");
		niter=200;
		dt=8.0E-4;
		grid_points[0]=64;
		grid_points[1]=64;
		grid_points[2]=64;
	}
	printf(" Size: %3dx%3dx%3d\n", grid_points[0], grid_points[1], grid_points[2]);
	printf(" Iterations: %3d   dt: %10.6f\n", niter, dt);
	if (((grid_points[0]>64)||(grid_points[1]>64))||(grid_points[2]>64))
	{
		printf("%d, %d, %d\n", grid_points[0], grid_points[1], grid_points[2]);
		printf(" Problem size too big for compiled array sizes\n");
		exit(1);
	}
	set_constants();
	initialize();
	lhsinit();
	exact_rhs();
	/*
	--------------------------------------------------------------------
	c      do one time step to touch all code, and reinitialize
	c-------------------------------------------------------------------
	*/
	{
		adi();
	}
	initialize();
	timer_clear(1);
	timer_start(1);
	{
		#pragma loop name main#0 
		for (step=1; step<=niter; step ++ )
		{
			if (((step%20)==0)||(step==1))
			{
				printf(" Time step %4d\n", step);
			}
			adi();
		}
	}
	/* end parallel */
	timer_stop(1);
	tmax=timer_read(1);
	verify(niter,  & class,  & verified);
	if (tmax!=0)
	{
		mflops=((((((881.174*pow((double)64, 3.0))-(4683.91*(((double)64)*((double)64))))+(11484.5*((double)64)))-19272.4)*((double)niter))/(tmax*1000000.0));
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("SP", class, grid_points[0], grid_points[1], grid_points[2], niter, nthreads, tmax, mflops, "          floating point", verified, "2.3", "24 Jun 2024", "gcc", "gcc", "(none)", "-I../common", "-O3 ", "(none)", "(none)");
	return _ret_val_0;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void add(void )
{
	int i, j, k, m;
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c addition of update to the vector u
	c-------------------------------------------------------------------
	*/
	#pragma loop name add#0 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name add#0#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name add#0#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name add#0#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					u[m][i][j][k]=(u[m][i][j][k]+rhs[m][i][j][k]);
				}
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void adi(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	compute_rhs();
	txinvr();
	x_solve();
	y_solve();
	z_solve();
	add();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void error_norm(double rms[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c this function computes the norm of the difference between the
	c computed solution and the exact solution
	c-------------------------------------------------------------------
	*/
	int i, j, k, m, d;
	double xi, eta, zeta, u_exact[5], add;
	#pragma loop name error_norm#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (m=0; m<5; m ++ )
	{
		rms[m]=0.0;
	}
	#pragma loop name error_norm#1 
	for (i=0; i<=(grid_points[0]-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name error_norm#1#0 
		for (j=0; j<=(grid_points[1]-1); j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name error_norm#1#0#0 
			for (k=0; k<=(grid_points[2]-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, u_exact);
				#pragma loop name error_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=(u[m][i][j][k]-u_exact[m]);
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
	#pragma loop name error_norm#2 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name error_norm#2#0 
		for (d=0; d<3; d ++ )
		{
			rms[m]=(rms[m]/((double)(grid_points[d]-2)));
		}
		rms[m]=sqrt(rms[m]);
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void rhs_norm(double rms[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	int i, j, k, d, m;
	double add;
	#pragma loop name rhs_norm#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (m=0; m<5; m ++ )
	{
		rms[m]=0.0;
	}
	#pragma loop name rhs_norm#1 
	for (i=0; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name rhs_norm#1#0 
		for (j=0; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name rhs_norm#1#0#0 
			for (k=0; k<=(grid_points[2]-2); k ++ )
			{
				#pragma loop name rhs_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=rhs[m][i][j][k];
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
	#pragma loop name rhs_norm#2 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name rhs_norm#2#0 
		for (d=0; d<3; d ++ )
		{
			rms[m]=(rms[m]/((double)(grid_points[d]-2)));
		}
		rms[m]=sqrt(rms[m]);
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void exact_rhs(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c compute the right hand side based on exact solution
	c-------------------------------------------------------------------
	*/
	double dtemp[5], xi, eta, zeta, dtpp;
	int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;
	/*
	--------------------------------------------------------------------
	c      initialize                                  
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#0 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name exact_rhs#0#0 
		for (i=0; i<=(grid_points[0]-1); i ++ )
		{
			#pragma loop name exact_rhs#0#0#0 
			for (j=0; j<=(grid_points[1]-1); j ++ )
			{
				#pragma loop name exact_rhs#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (k=0; k<=(grid_points[2]-1); k ++ )
				{
					forcing[m][i][j][k]=0.0;
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      xi-direction flux differences                      
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#1 
	for (k=1; k<=(grid_points[2]-2); k ++ )
	{
		zeta=(((double)k)*dnzm1);
		#pragma loop name exact_rhs#1#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name exact_rhs#1#0#0 
			for (i=0; i<=(grid_points[0]-1); i ++ )
			{
				xi=(((double)i)*dnxm1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][i]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#1#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][i]=(dtpp*dtemp[m]);
				}
				cuf[i]=(buf[1][i]*buf[1][i]);
				buf[0][i]=((cuf[i]+(buf[2][i]*buf[2][i]))+(buf[3][i]*buf[3][i]));
				q[i]=(0.5*(((buf[1][i]*ue[1][i])+(buf[2][i]*ue[2][i]))+(buf[3][i]*ue[3][i])));
			}
			#pragma loop name exact_rhs#1#0#1 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				im1=(i-1);
				ip1=(i+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(tx2*(ue[1][ip1]-ue[1][im1])))+(dx1tx1*((ue[0][ip1]-(2.0*ue[0][i]))+ue[0][im1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(tx2*(((ue[1][ip1]*buf[1][ip1])+(c2*(ue[4][ip1]-q[ip1])))-((ue[1][im1]*buf[1][im1])+(c2*(ue[4][im1]-q[im1]))))))+(xxcon1*((buf[1][ip1]-(2.0*buf[1][i]))+buf[1][im1])))+(dx2tx1*((ue[1][ip1]-(2.0*ue[1][i]))+ue[1][im1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(tx2*((ue[2][ip1]*buf[1][ip1])-(ue[2][im1]*buf[1][im1]))))+(xxcon2*((buf[2][ip1]-(2.0*buf[2][i]))+buf[2][im1])))+(dx3tx1*((ue[2][ip1]-(2.0*ue[2][i]))+ue[2][im1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(tx2*((ue[3][ip1]*buf[1][ip1])-(ue[3][im1]*buf[1][im1]))))+(xxcon2*((buf[3][ip1]-(2.0*buf[3][i]))+buf[3][im1])))+(dx4tx1*((ue[3][ip1]-(2.0*ue[3][i]))+ue[3][im1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(tx2*((buf[1][ip1]*((c1*ue[4][ip1])-(c2*q[ip1])))-(buf[1][im1]*((c1*ue[4][im1])-(c2*q[im1]))))))+((0.5*xxcon3)*((buf[0][ip1]-(2.0*buf[0][i]))+buf[0][im1])))+(xxcon4*((cuf[ip1]-(2.0*cuf[i]))+cuf[im1])))+(xxcon5*((buf[4][ip1]-(2.0*buf[4][i]))+buf[4][im1])))+(dx5tx1*((ue[4][ip1]-(2.0*ue[4][i]))+ue[4][im1])));
			}
			/*
			--------------------------------------------------------------------
			c            Fourth-order dissipation                         
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#1#0#2 
			for (m=0; m<5; m ++ )
			{
				i=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][i])-(4.0*ue[m][i+1]))+ue[m][i+2])));
				i=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][i-1])+(6.0*ue[m][i]))-(4.0*ue[m][i+1]))+ue[m][i+2])));
			}
			#pragma loop name exact_rhs#1#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#1#0#3#0 
				for (i=3; i<=(grid_points[0]-4); i ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][i-2]-(4.0*ue[m][i-1]))+(6.0*ue[m][i]))-(4.0*ue[m][i+1]))+ue[m][i+2])));
				}
			}
			#pragma loop name exact_rhs#1#0#4 
			for (m=0; m<5; m ++ )
			{
				i=(grid_points[0]-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][i-2]-(4.0*ue[m][i-1]))+(6.0*ue[m][i]))-(4.0*ue[m][i+1]))));
				i=(grid_points[0]-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][i-2]-(4.0*ue[m][i-1]))+(5.0*ue[m][i]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c  eta-direction flux differences             
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#2 
	for (k=1; k<=(grid_points[2]-2); k ++ )
	{
		zeta=(((double)k)*dnzm1);
		#pragma loop name exact_rhs#2#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			xi=(((double)i)*dnxm1);
			#pragma loop name exact_rhs#2#0#0 
			for (j=0; j<=(grid_points[1]-1); j ++ )
			{
				eta=(((double)j)*dnym1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#2#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][j]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#2#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][j]=(dtpp*dtemp[m]);
				}
				cuf[j]=(buf[2][j]*buf[2][j]);
				buf[0][j]=((cuf[j]+(buf[1][j]*buf[1][j]))+(buf[3][j]*buf[3][j]));
				q[j]=(0.5*(((buf[1][j]*ue[1][j])+(buf[2][j]*ue[2][j]))+(buf[3][j]*ue[3][j])));
			}
			#pragma loop name exact_rhs#2#0#1 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				jm1=(j-1);
				jp1=(j+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(ty2*(ue[2][jp1]-ue[2][jm1])))+(dy1ty1*((ue[0][jp1]-(2.0*ue[0][j]))+ue[0][jm1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(ty2*((ue[1][jp1]*buf[2][jp1])-(ue[1][jm1]*buf[2][jm1]))))+(yycon2*((buf[1][jp1]-(2.0*buf[1][j]))+buf[1][jm1])))+(dy2ty1*((ue[1][jp1]-(2.0*ue[1][j]))+ue[1][jm1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(ty2*(((ue[2][jp1]*buf[2][jp1])+(c2*(ue[4][jp1]-q[jp1])))-((ue[2][jm1]*buf[2][jm1])+(c2*(ue[4][jm1]-q[jm1]))))))+(yycon1*((buf[2][jp1]-(2.0*buf[2][j]))+buf[2][jm1])))+(dy3ty1*((ue[2][jp1]-(2.0*ue[2][j]))+ue[2][jm1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(ty2*((ue[3][jp1]*buf[2][jp1])-(ue[3][jm1]*buf[2][jm1]))))+(yycon2*((buf[3][jp1]-(2.0*buf[3][j]))+buf[3][jm1])))+(dy4ty1*((ue[3][jp1]-(2.0*ue[3][j]))+ue[3][jm1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(ty2*((buf[2][jp1]*((c1*ue[4][jp1])-(c2*q[jp1])))-(buf[2][jm1]*((c1*ue[4][jm1])-(c2*q[jm1]))))))+((0.5*yycon3)*((buf[0][jp1]-(2.0*buf[0][j]))+buf[0][jm1])))+(yycon4*((cuf[jp1]-(2.0*cuf[j]))+cuf[jm1])))+(yycon5*((buf[4][jp1]-(2.0*buf[4][j]))+buf[4][jm1])))+(dy5ty1*((ue[4][jp1]-(2.0*ue[4][j]))+ue[4][jm1])));
			}
			/*
			--------------------------------------------------------------------
			c            Fourth-order dissipation                      
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#2#0#2 
			for (m=0; m<5; m ++ )
			{
				j=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][j])-(4.0*ue[m][j+1]))+ue[m][j+2])));
				j=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][j-1])+(6.0*ue[m][j]))-(4.0*ue[m][j+1]))+ue[m][j+2])));
			}
			#pragma loop name exact_rhs#2#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#2#0#3#0 
				for (j=3; j<=(grid_points[1]-4); j ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][j-2]-(4.0*ue[m][j-1]))+(6.0*ue[m][j]))-(4.0*ue[m][j+1]))+ue[m][j+2])));
				}
			}
			#pragma loop name exact_rhs#2#0#4 
			for (m=0; m<5; m ++ )
			{
				j=(grid_points[1]-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][j-2]-(4.0*ue[m][j-1]))+(6.0*ue[m][j]))-(4.0*ue[m][j+1]))));
				j=(grid_points[1]-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][j-2]-(4.0*ue[m][j-1]))+(5.0*ue[m][j]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      zeta-direction flux differences                      
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#3 
	for (j=1; j<=(grid_points[1]-2); j ++ )
	{
		eta=(((double)j)*dnym1);
		#pragma loop name exact_rhs#3#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			xi=(((double)i)*dnxm1);
			#pragma loop name exact_rhs#3#0#0 
			for (k=0; k<=(grid_points[2]-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#3#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[m][k]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#3#0#0#1 
				for (m=1; m<5; m ++ )
				{
					buf[m][k]=(dtpp*dtemp[m]);
				}
				cuf[k]=(buf[3][k]*buf[3][k]);
				buf[0][k]=((cuf[k]+(buf[1][k]*buf[1][k]))+(buf[2][k]*buf[2][k]));
				q[k]=(0.5*(((buf[1][k]*ue[1][k])+(buf[2][k]*ue[2][k]))+(buf[3][k]*ue[3][k])));
			}
			#pragma loop name exact_rhs#3#0#1 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				km1=(k-1);
				kp1=(k+1);
				forcing[0][i][j][k]=((forcing[0][i][j][k]-(tz2*(ue[3][kp1]-ue[3][km1])))+(dz1tz1*((ue[0][kp1]-(2.0*ue[0][k]))+ue[0][km1])));
				forcing[1][i][j][k]=(((forcing[1][i][j][k]-(tz2*((ue[1][kp1]*buf[3][kp1])-(ue[1][km1]*buf[3][km1]))))+(zzcon2*((buf[1][kp1]-(2.0*buf[1][k]))+buf[1][km1])))+(dz2tz1*((ue[1][kp1]-(2.0*ue[1][k]))+ue[1][km1])));
				forcing[2][i][j][k]=(((forcing[2][i][j][k]-(tz2*((ue[2][kp1]*buf[3][kp1])-(ue[2][km1]*buf[3][km1]))))+(zzcon2*((buf[2][kp1]-(2.0*buf[2][k]))+buf[2][km1])))+(dz3tz1*((ue[2][kp1]-(2.0*ue[2][k]))+ue[2][km1])));
				forcing[3][i][j][k]=(((forcing[3][i][j][k]-(tz2*(((ue[3][kp1]*buf[3][kp1])+(c2*(ue[4][kp1]-q[kp1])))-((ue[3][km1]*buf[3][km1])+(c2*(ue[4][km1]-q[km1]))))))+(zzcon1*((buf[3][kp1]-(2.0*buf[3][k]))+buf[3][km1])))+(dz4tz1*((ue[3][kp1]-(2.0*ue[3][k]))+ue[3][km1])));
				forcing[4][i][j][k]=(((((forcing[4][i][j][k]-(tz2*((buf[3][kp1]*((c1*ue[4][kp1])-(c2*q[kp1])))-(buf[3][km1]*((c1*ue[4][km1])-(c2*q[km1]))))))+((0.5*zzcon3)*((buf[0][kp1]-(2.0*buf[0][k]))+buf[0][km1])))+(zzcon4*((cuf[kp1]-(2.0*cuf[k]))+cuf[km1])))+(zzcon5*((buf[4][kp1]-(2.0*buf[4][k]))+buf[4][km1])))+(dz5tz1*((ue[4][kp1]-(2.0*ue[4][k]))+ue[4][km1])));
			}
			/*
			--------------------------------------------------------------------
			c            Fourth-order dissipation                        
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#3#0#2 
			for (m=0; m<5; m ++ )
			{
				k=1;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((5.0*ue[m][k])-(4.0*ue[m][k+1]))+ue[m][k+2])));
				k=2;
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((( - 4.0)*ue[m][k-1])+(6.0*ue[m][k]))-(4.0*ue[m][k+1]))+ue[m][k+2])));
			}
			#pragma loop name exact_rhs#3#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#3#0#3#0 
				for (k=3; k<=(grid_points[2]-4); k ++ )
				{
					forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((((ue[m][k-2]-(4.0*ue[m][k-1]))+(6.0*ue[m][k]))-(4.0*ue[m][k+1]))+ue[m][k+2])));
				}
			}
			#pragma loop name exact_rhs#3#0#4 
			for (m=0; m<5; m ++ )
			{
				k=(grid_points[2]-3);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*(((ue[m][k-2]-(4.0*ue[m][k-1]))+(6.0*ue[m][k]))-(4.0*ue[m][k+1]))));
				k=(grid_points[2]-2);
				forcing[m][i][j][k]=(forcing[m][i][j][k]-(dssp*((ue[m][k-2]-(4.0*ue[m][k-1]))+(5.0*ue[m][k]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c now change the sign of the forcing function, 
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#4 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name exact_rhs#4#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name exact_rhs#4#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name exact_rhs#4#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					forcing[m][i][j][k]=(( - 1.0)*forcing[m][i][j][k]);
				}
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void exact_solution(double xi, double eta, double zeta, double dtemp[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c this function returns the exact solution at point xi, eta, zeta  
	c-------------------------------------------------------------------
	*/
	int m;
	#pragma loop name exact_solution#0 
	for (m=0; m<5; m ++ )
	{
		dtemp[m]=(((ce[0][m]+(xi*(ce[1][m]+(xi*(ce[4][m]+(xi*(ce[7][m]+(xi*ce[10][m]))))))))+(eta*(ce[2][m]+(eta*(ce[5][m]+(eta*(ce[8][m]+(eta*ce[11][m]))))))))+(zeta*(ce[3][m]+(zeta*(ce[6][m]+(zeta*(ce[9][m]+(zeta*ce[12][m]))))))));
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void initialize(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c This subroutine initializes the field variable u using 
	c tri-linear transfinite interpolation of the boundary values     
	c-------------------------------------------------------------------
	*/
	int i, j, k, m, ix, iy, iz;
	double xi, eta, zeta, Pface[2][3][5], Pxi, Peta, Pzeta, temp[5];
	/*
	--------------------------------------------------------------------
	c  Later (in compute_rhs) we compute 1u for every element. A few of 
	c  the corner elements are not used, but it convenient (and faster) 
	c  to compute the whole thing with a simple loop. Make sure those 
	c  values are nonzero by initializing the whole thing here. 
	c-------------------------------------------------------------------
	*/
	#pragma loop name initialize#0 
	for (i=0; i<=(64-1); i ++ )
	{
		#pragma loop name initialize#0#0 
		for (j=0; j<=(64-1); j ++ )
		{
			#pragma loop name initialize#0#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (k=0; k<=(64-1); k ++ )
			{
				u[0][i][j][k]=1.0;
				u[1][i][j][k]=0.0;
				u[2][i][j][k]=0.0;
				u[3][i][j][k]=0.0;
				u[4][i][j][k]=1.0;
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c first store the "interpolated" values everywhere on the grid    
	c-------------------------------------------------------------------
	*/
	#pragma loop name initialize#1 
	for (i=0; i<=(grid_points[0]-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#1#0 
		for (j=0; j<=(grid_points[1]-1); j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name initialize#1#0#0 
			for (k=0; k<=(grid_points[2]-1); k ++ )
			{
				zeta=(((double)k)*dnzm1);
				#pragma loop name initialize#1#0#0#0 
				for (ix=0; ix<2; ix ++ )
				{
					exact_solution((double)ix, eta, zeta,  & Pface[ix][0][0]);
				}
				#pragma loop name initialize#1#0#0#1 
				for (iy=0; iy<2; iy ++ )
				{
					exact_solution(xi, (double)iy, zeta,  & Pface[iy][1][0]);
				}
				#pragma loop name initialize#1#0#0#2 
				for (iz=0; iz<2; iz ++ )
				{
					exact_solution(xi, eta, (double)iz,  & Pface[iz][2][0]);
				}
				#pragma loop name initialize#1#0#0#3 
				for (m=0; m<5; m ++ )
				{
					Pxi=((xi*Pface[1][0][m])+((1.0-xi)*Pface[0][0][m]));
					Peta=((eta*Pface[1][1][m])+((1.0-eta)*Pface[0][1][m]));
					Pzeta=((zeta*Pface[1][2][m])+((1.0-zeta)*Pface[0][2][m]));
					u[m][i][j][k]=((((((Pxi+Peta)+Pzeta)-(Pxi*Peta))-(Pxi*Pzeta))-(Peta*Pzeta))+((Pxi*Peta)*Pzeta));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c now store the exact values on the boundaries        
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c west face                                                  
	c-------------------------------------------------------------------
	*/
	xi=0.0;
	i=0;
	#pragma loop name initialize#2 
	for (j=0; j<grid_points[1]; j ++ )
	{
		eta=(((double)j)*dnym1);
		#pragma loop name initialize#2#0 
		for (k=0; k<grid_points[2]; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#2#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c east face                                                      
	c-------------------------------------------------------------------
	*/
	xi=1.0;
	i=(grid_points[0]-1);
	#pragma loop name initialize#3 
	for (j=0; j<grid_points[1]; j ++ )
	{
		eta=(((double)j)*dnym1);
		#pragma loop name initialize#3#0 
		for (k=0; k<grid_points[2]; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#3#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c south face                                                 
	c-------------------------------------------------------------------
	*/
	eta=0.0;
	j=0;
	#pragma loop name initialize#4 
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#4#0 
		for (k=0; k<grid_points[2]; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#4#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c north face                                    
	c-------------------------------------------------------------------
	*/
	eta=1.0;
	j=(grid_points[1]-1);
	#pragma loop name initialize#5 
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#5#0 
		for (k=0; k<grid_points[2]; k ++ )
		{
			zeta=(((double)k)*dnzm1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#5#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c bottom face                                       
	c-------------------------------------------------------------------
	*/
	zeta=0.0;
	k=0;
	#pragma loop name initialize#6 
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#6#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			eta=(((double)j)*dnym1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#6#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c top face     
	c-------------------------------------------------------------------
	*/
	zeta=1.0;
	k=(grid_points[2]-1);
	#pragma loop name initialize#7 
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#7#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			eta=(((double)j)*dnym1);
			exact_solution(xi, eta, zeta, temp);
			#pragma loop name initialize#7#0#0 
			for (m=0; m<5; m ++ )
			{
				u[m][i][j][k]=temp[m];
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void lhsinit(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	int i, j, k, n;
	/*
	--------------------------------------------------------------------
	c     zap the whole left hand side for starters
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsinit#0 
	for (n=0; n<15; n ++ )
	{
		#pragma loop name lhsinit#0#0 
		for (i=0; i<grid_points[0]; i ++ )
		{
			#pragma loop name lhsinit#0#0#0 
			for (j=0; j<grid_points[1]; j ++ )
			{
				#pragma loop name lhsinit#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (k=0; k<grid_points[2]; k ++ )
				{
					lhs[n][i][j][k]=0.0;
				}
			}
		}
	}
	
	/*
	--------------------------------------------------------------------
	c      next, set all diagonal values to 1. This is overkill, but 
	c      convenient
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsinit#1 
	for (n=0; n<3; n ++ )
	{
		#pragma loop name lhsinit#1#0 
		for (i=0; i<grid_points[0]; i ++ )
		{
			#pragma loop name lhsinit#1#0#0 
			for (j=0; j<grid_points[1]; j ++ )
			{
				#pragma loop name lhsinit#1#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (k=0; k<grid_points[2]; k ++ )
				{
					lhs[(5*n)+2][i][j][k]=1.0;
				}
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void lhsx(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c This function computes the left hand side for the three x-factors  
	c-------------------------------------------------------------------
	*/
	double ru1;
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c      first fill the lhs for the u-eigenvalue                   
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsx#0 
	for (j=1; j<=(grid_points[1]-2); j ++ )
	{
		#pragma loop name lhsx#0#0 
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			#pragma loop name lhsx#0#0#0 
			for (i=0; i<=(grid_points[0]-1); i ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[i]=us[i][j][k];
				rhon[i]=(((dx2+(con43*ru1))>(((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1))) ? (dx2+(con43*ru1)) : (((dx5+(c1c5*ru1))>(((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)) ? (dx5+(c1c5*ru1)) : (((dxmax+ru1)>dx1) ? (dxmax+ru1) : dx1)));
			}
			#pragma loop name lhsx#0#0#1 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttx2)*cv[i-1])-(dttx1*rhon[i-1]));
				lhs[2][i][j][k]=(1.0+(c2dttx1*rhon[i]));
				lhs[3][i][j][k]=((dttx2*cv[i+1])-(dttx1*rhon[i+1]));
				lhs[4][i][j][k]=0.0;
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order dissipation                             
	c-------------------------------------------------------------------
	*/
	i=1;
	#pragma loop name lhsx#1 
	for (j=1; j<=(grid_points[1]-2); j ++ )
	{
		#pragma loop name lhsx#1#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz5);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			lhs[1][i+1][j][k]=(lhs[1][i+1][j][k]-comz4);
			lhs[2][i+1][j][k]=(lhs[2][i+1][j][k]+comz6);
			lhs[3][i+1][j][k]=(lhs[3][i+1][j][k]-comz4);
			lhs[4][i+1][j][k]=(lhs[4][i+1][j][k]+comz1);
		}
	}
	#pragma loop name lhsx#2 
	for (i=3; i<=(grid_points[0]-4); i ++ )
	{
		#pragma loop name lhsx#2#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsx#2#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			}
		}
	}
	i=(grid_points[0]-3);
	#pragma loop name lhsx#3 
	for (j=1; j<=(grid_points[1]-2); j ++ )
	{
		#pragma loop name lhsx#3#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
			lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[0][i+1][j][k]=(lhs[0][i+1][j][k]+comz1);
			lhs[1][i+1][j][k]=(lhs[1][i+1][j][k]-comz4);
			lhs[2][i+1][j][k]=(lhs[2][i+1][j][k]+comz5);
		}
	}
	/*
	--------------------------------------------------------------------
	c      subsequently, fill the other factors (u+c), (u-c) by adding to 
	c      the first  
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsx#4 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsx#4#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsx#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0+5][i][j][k]=lhs[0][i][j][k];
				lhs[1+5][i][j][k]=(lhs[1][i][j][k]-(dttx2*speed[i-1][j][k]));
				lhs[2+5][i][j][k]=lhs[2][i][j][k];
				lhs[3+5][i][j][k]=(lhs[3][i][j][k]+(dttx2*speed[i+1][j][k]));
				lhs[4+5][i][j][k]=lhs[4][i][j][k];
				lhs[0+10][i][j][k]=lhs[0][i][j][k];
				lhs[1+10][i][j][k]=(lhs[1][i][j][k]+(dttx2*speed[i-1][j][k]));
				lhs[2+10][i][j][k]=lhs[2][i][j][k];
				lhs[3+10][i][j][k]=(lhs[3][i][j][k]-(dttx2*speed[i+1][j][k]));
				lhs[4+10][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void lhsy(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c This function computes the left hand side for the three y-factors   
	c-------------------------------------------------------------------
	*/
	double ru1;
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c      first fill the lhs for the u-eigenvalue         
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsy#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsy#0#0 
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			#pragma loop name lhsy#0#0#0 
			for (j=0; j<=(grid_points[1]-1); j ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[j]=vs[i][j][k];
				rhoq[j]=(((dy3+(con43*ru1))>(((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1))) ? (dy3+(con43*ru1)) : (((dy5+(c1c5*ru1))>(((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)) ? (dy5+(c1c5*ru1)) : (((dymax+ru1)>dy1) ? (dymax+ru1) : dy1)));
			}
			#pragma loop name lhsy#0#0#1 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dtty2)*cv[j-1])-(dtty1*rhoq[j-1]));
				lhs[2][i][j][k]=(1.0+(c2dtty1*rhoq[j]));
				lhs[3][i][j][k]=((dtty2*cv[j+1])-(dtty1*rhoq[j+1]));
				lhs[4][i][j][k]=0.0;
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order dissipation                             
	c-------------------------------------------------------------------
	*/
	j=1;
	#pragma loop name lhsy#1 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsy#1#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz5);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			lhs[1][i][j+1][k]=(lhs[1][i][j+1][k]-comz4);
			lhs[2][i][j+1][k]=(lhs[2][i][j+1][k]+comz6);
			lhs[3][i][j+1][k]=(lhs[3][i][j+1][k]-comz4);
			lhs[4][i][j+1][k]=(lhs[4][i][j+1][k]+comz1);
		}
	}
	#pragma loop name lhsy#2 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsy#2#0 
		for (j=3; j<=(grid_points[1]-4); j ++ )
		{
			#pragma loop name lhsy#2#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			}
		}
	}
	j=(grid_points[1]-3);
	#pragma loop name lhsy#3 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsy#3#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
			lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[0][i][j+1][k]=(lhs[0][i][j+1][k]+comz1);
			lhs[1][i][j+1][k]=(lhs[1][i][j+1][k]-comz4);
			lhs[2][i][j+1][k]=(lhs[2][i][j+1][k]+comz5);
		}
	}
	/*
	--------------------------------------------------------------------
	c      subsequently, do the other two factors                    
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsy#4 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsy#4#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsy#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0+5][i][j][k]=lhs[0][i][j][k];
				lhs[1+5][i][j][k]=(lhs[1][i][j][k]-(dtty2*speed[i][j-1][k]));
				lhs[2+5][i][j][k]=lhs[2][i][j][k];
				lhs[3+5][i][j][k]=(lhs[3][i][j][k]+(dtty2*speed[i][j+1][k]));
				lhs[4+5][i][j][k]=lhs[4][i][j][k];
				lhs[0+10][i][j][k]=lhs[0][i][j][k];
				lhs[1+10][i][j][k]=(lhs[1][i][j][k]+(dtty2*speed[i][j-1][k]));
				lhs[2+10][i][j][k]=lhs[2][i][j][k];
				lhs[3+10][i][j][k]=(lhs[3][i][j][k]-(dtty2*speed[i][j+1][k]));
				lhs[4+10][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void lhsz(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c This function computes the left hand side for the three z-factors   
	c-------------------------------------------------------------------
	*/
	double ru1;
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c first fill the lhs for the u-eigenvalue                          
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsz#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsz#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsz#0#0#0 
			for (k=0; k<=(grid_points[2]-1); k ++ )
			{
				ru1=(c3c4*rho_i[i][j][k]);
				cv[k]=ws[i][j][k];
				rhos[k]=(((dz4+(con43*ru1))>(((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1))) ? (dz4+(con43*ru1)) : (((dz5+(c1c5*ru1))>(((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)) ? (dz5+(c1c5*ru1)) : (((dzmax+ru1)>dz1) ? (dzmax+ru1) : dz1)));
			}
			#pragma loop name lhsz#0#0#1 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0][i][j][k]=0.0;
				lhs[1][i][j][k]=((( - dttz2)*cv[k-1])-(dttz1*rhos[k-1]));
				lhs[2][i][j][k]=(1.0+(c2dttz1*rhos[k]));
				lhs[3][i][j][k]=((dttz2*cv[k+1])-(dttz1*rhos[k+1]));
				lhs[4][i][j][k]=0.0;
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order dissipation                                  
	c-------------------------------------------------------------------
	*/
	k=1;
	#pragma loop name lhsz#1 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsz#1#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz5);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			lhs[1][i][j][k+1]=(lhs[1][i][j][k+1]-comz4);
			lhs[2][i][j][k+1]=(lhs[2][i][j][k+1]+comz6);
			lhs[3][i][j][k+1]=(lhs[3][i][j][k+1]-comz4);
			lhs[4][i][j][k+1]=(lhs[4][i][j][k+1]+comz1);
		}
	}
	#pragma loop name lhsz#2 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsz#2#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsz#2#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (k=3; k<=(grid_points[2]-4); k ++ )
			{
				lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
				lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
				lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
				lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
				lhs[4][i][j][k]=(lhs[4][i][j][k]+comz1);
			}
		}
	}
	k=(grid_points[2]-3);
	#pragma loop name lhsz#3 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsz#3#0 
		#pragma cetus parallel 
		#pragma omp parallel for
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			lhs[0][i][j][k]=(lhs[0][i][j][k]+comz1);
			lhs[1][i][j][k]=(lhs[1][i][j][k]-comz4);
			lhs[2][i][j][k]=(lhs[2][i][j][k]+comz6);
			lhs[3][i][j][k]=(lhs[3][i][j][k]-comz4);
			lhs[0][i][j][k+1]=(lhs[0][i][j][k+1]+comz1);
			lhs[1][i][j][k+1]=(lhs[1][i][j][k+1]-comz4);
			lhs[2][i][j][k+1]=(lhs[2][i][j][k+1]+comz5);
		}
	}
	/*
	--------------------------------------------------------------------
	c      subsequently, fill the other factors (u+c), (u-c) 
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsz#4 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name lhsz#4#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name lhsz#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				lhs[0+5][i][j][k]=lhs[0][i][j][k];
				lhs[1+5][i][j][k]=(lhs[1][i][j][k]-(dttz2*speed[i][j][k-1]));
				lhs[2+5][i][j][k]=lhs[2][i][j][k];
				lhs[3+5][i][j][k]=(lhs[3][i][j][k]+(dttz2*speed[i][j][k+1]));
				lhs[4+5][i][j][k]=lhs[4][i][j][k];
				lhs[0+10][i][j][k]=lhs[0][i][j][k];
				lhs[1+10][i][j][k]=(lhs[1][i][j][k]+(dttz2*speed[i][j][k-1]));
				lhs[2+10][i][j][k]=lhs[2][i][j][k];
				lhs[3+10][i][j][k]=(lhs[3][i][j][k]-(dttz2*speed[i][j][k+1]));
				lhs[4+10][i][j][k]=lhs[4][i][j][k];
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void ninvr(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   block-diagonal matrix-vector multiplication              
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	double r1, r2, r3, r4, r5, t1, t2;
	#pragma loop name ninvr#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name ninvr#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name ninvr#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt*r3);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=( - r2);
				rhs[1][i][j][k]=r1;
				rhs[2][i][j][k]=(bt*(r4-r5));
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void pinvr(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   block-diagonal matrix-vector multiplication                       
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	double r1, r2, r3, r4, r5, t1, t2;
	#pragma loop name pinvr#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name pinvr#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name pinvr#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=(bt*r1);
				t2=(0.5*(r4+r5));
				rhs[0][i][j][k]=(bt*(r4-r5));
				rhs[1][i][j][k]=( - r3);
				rhs[2][i][j][k]=r2;
				rhs[3][i][j][k]=(( - t1)+t2);
				rhs[4][i][j][k]=(t1+t2);
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void compute_rhs(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	int i, j, k, m;
	double aux, rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
	/*
	--------------------------------------------------------------------
	c      compute the reciprocal of density, and the kinetic energy, 
	c      and the speed of sound. 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#0 
	for (i=0; i<=(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#0#0 
		for (j=0; j<=(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#0#0#0 
			for (k=0; k<=(grid_points[2]-1); k ++ )
			{
				rho_inv=(1.0/u[0][i][j][k]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u[1][i][j][k]*rho_inv);
				vs[i][j][k]=(u[2][i][j][k]*rho_inv);
				ws[i][j][k]=(u[3][i][j][k]*rho_inv);
				square[i][j][k]=((0.5*(((u[1][i][j][k]*u[1][i][j][k])+(u[2][i][j][k]*u[2][i][j][k]))+(u[3][i][j][k]*u[3][i][j][k])))*rho_inv);
				qs[i][j][k]=(square[i][j][k]*rho_inv);
				/*
				--------------------------------------------------------------------
				c               (do not need speed and ainx until the lhs computation)
				c-------------------------------------------------------------------
				*/
				aux=((c1c2*rho_inv)*(u[4][i][j][k]-square[i][j][k]));
				aux=sqrt(aux);
				speed[i][j][k]=aux;
				ainv[i][j][k]=(1.0/aux);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c copy the exact forcing term to the right hand side;  because 
	c this forcing term is known, we can store it on the whole grid
	c including the boundary                   
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#1 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#1#0 
		for (i=0; i<=(grid_points[0]-1); i ++ )
		{
			#pragma loop name compute_rhs#1#0#0 
			for (j=0; j<=(grid_points[1]-1); j ++ )
			{
				#pragma loop name compute_rhs#1#0#0#0 
				for (k=0; k<=(grid_points[2]-1); k ++ )
				{
					rhs[m][i][j][k]=forcing[m][i][j][k];
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      compute xi-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#2 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name compute_rhs#2#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#2#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				uijk=us[i][j][k];
				up1=us[i+1][j][k];
				um1=us[i-1][j][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(dx1tx1*((u[0][i+1][j][k]-(2.0*u[0][i][j][k]))+u[0][i-1][j][k])))-(tx2*(u[1][i+1][j][k]-u[1][i-1][j][k])));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(dx2tx1*((u[1][i+1][j][k]-(2.0*u[1][i][j][k]))+u[1][i-1][j][k])))+((xxcon2*con43)*((up1-(2.0*uijk))+um1)))-(tx2*(((u[1][i+1][j][k]*up1)-(u[1][i-1][j][k]*um1))+((((u[4][i+1][j][k]-square[i+1][j][k])-u[4][i-1][j][k])+square[i-1][j][k])*c2))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(dx3tx1*((u[2][i+1][j][k]-(2.0*u[2][i][j][k]))+u[2][i-1][j][k])))+(xxcon2*((vs[i+1][j][k]-(2.0*vs[i][j][k]))+vs[i-1][j][k])))-(tx2*((u[2][i+1][j][k]*up1)-(u[2][i-1][j][k]*um1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(dx4tx1*((u[3][i+1][j][k]-(2.0*u[3][i][j][k]))+u[3][i-1][j][k])))+(xxcon2*((ws[i+1][j][k]-(2.0*ws[i][j][k]))+ws[i-1][j][k])))-(tx2*((u[3][i+1][j][k]*up1)-(u[3][i-1][j][k]*um1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(dx5tx1*((u[4][i+1][j][k]-(2.0*u[4][i][j][k]))+u[4][i-1][j][k])))+(xxcon3*((qs[i+1][j][k]-(2.0*qs[i][j][k]))+qs[i-1][j][k])))+(xxcon4*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(xxcon5*(((u[4][i+1][j][k]*rho_i[i+1][j][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u[4][i-1][j][k]*rho_i[i-1][j][k]))))-(tx2*((((c1*u[4][i+1][j][k])-(c2*square[i+1][j][k]))*up1)-(((c1*u[4][i-1][j][k])-(c2*square[i-1][j][k]))*um1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order xi-direction dissipation               
	c-------------------------------------------------------------------
	*/
	i=1;
	#pragma loop name compute_rhs#3 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#3#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#3#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((5.0*u[m][i][j][k])-(4.0*u[m][i+1][j][k]))+u[m][i+2][j][k])));
			}
		}
	}
	i=2;
	#pragma loop name compute_rhs#4 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#4#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((( - 4.0)*u[m][i-1][j][k])+(6.0*u[m][i][j][k]))-(4.0*u[m][i+1][j][k]))+u[m][i+2][j][k])));
			}
		}
	}
	#pragma loop name compute_rhs#5 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#5#0 
		for (i=(3*1); i<=((grid_points[0]-(3*1))-1); i ++ )
		{
			#pragma loop name compute_rhs#5#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name compute_rhs#5#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((u[m][i-2][j][k]-(4.0*u[m][i-1][j][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i+1][j][k]))+u[m][i+2][j][k])));
				}
			}
		}
	}
	i=(grid_points[0]-3);
	#pragma loop name compute_rhs#6 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#6#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#6#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((u[m][i-2][j][k]-(4.0*u[m][i-1][j][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i+1][j][k]))));
			}
		}
	}
	i=(grid_points[0]-2);
	#pragma loop name compute_rhs#7 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#7#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#7#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((u[m][i-2][j][k]-(4.0*u[m][i-1][j][k]))+(5.0*u[m][i][j][k]))));
			}
		}
	}
	
	/*
	--------------------------------------------------------------------
	c      compute eta-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#8 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name compute_rhs#8#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#8#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				vijk=vs[i][j][k];
				vp1=vs[i][j+1][k];
				vm1=vs[i][j-1][k];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(dy1ty1*((u[0][i][j+1][k]-(2.0*u[0][i][j][k]))+u[0][i][j-1][k])))-(ty2*(u[2][i][j+1][k]-u[2][i][j-1][k])));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(dy2ty1*((u[1][i][j+1][k]-(2.0*u[1][i][j][k]))+u[1][i][j-1][k])))+(yycon2*((us[i][j+1][k]-(2.0*us[i][j][k]))+us[i][j-1][k])))-(ty2*((u[1][i][j+1][k]*vp1)-(u[1][i][j-1][k]*vm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(dy3ty1*((u[2][i][j+1][k]-(2.0*u[2][i][j][k]))+u[2][i][j-1][k])))+((yycon2*con43)*((vp1-(2.0*vijk))+vm1)))-(ty2*(((u[2][i][j+1][k]*vp1)-(u[2][i][j-1][k]*vm1))+((((u[4][i][j+1][k]-square[i][j+1][k])-u[4][i][j-1][k])+square[i][j-1][k])*c2))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(dy4ty1*((u[3][i][j+1][k]-(2.0*u[3][i][j][k]))+u[3][i][j-1][k])))+(yycon2*((ws[i][j+1][k]-(2.0*ws[i][j][k]))+ws[i][j-1][k])))-(ty2*((u[3][i][j+1][k]*vp1)-(u[3][i][j-1][k]*vm1))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(dy5ty1*((u[4][i][j+1][k]-(2.0*u[4][i][j][k]))+u[4][i][j-1][k])))+(yycon3*((qs[i][j+1][k]-(2.0*qs[i][j][k]))+qs[i][j-1][k])))+(yycon4*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(yycon5*(((u[4][i][j+1][k]*rho_i[i][j+1][k])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u[4][i][j-1][k]*rho_i[i][j-1][k]))))-(ty2*((((c1*u[4][i][j+1][k])-(c2*square[i][j+1][k]))*vp1)-(((c1*u[4][i][j-1][k])-(c2*square[i][j-1][k]))*vm1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order eta-direction dissipation         
	c-------------------------------------------------------------------
	*/
	j=1;
	#pragma loop name compute_rhs#9 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#9#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#9#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((5.0*u[m][i][j][k])-(4.0*u[m][i][j+1][k]))+u[m][i][j+2][k])));
			}
		}
	}
	j=2;
	#pragma loop name compute_rhs#10 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#10#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#10#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((( - 4.0)*u[m][i][j-1][k])+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j+1][k]))+u[m][i][j+2][k])));
			}
		}
	}
	#pragma loop name compute_rhs#11 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#11#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#11#0#0 
			for (j=(3*1); j<=((grid_points[1]-(3*1))-1); j ++ )
			{
				#pragma loop name compute_rhs#11#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((u[m][i][j-2][k]-(4.0*u[m][i][j-1][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j+1][k]))+u[m][i][j+2][k])));
				}
			}
		}
	}
	j=(grid_points[1]-3);
	#pragma loop name compute_rhs#12 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#12#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#12#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((u[m][i][j-2][k]-(4.0*u[m][i][j-1][k]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j+1][k]))));
			}
		}
	}
	j=(grid_points[1]-2);
	#pragma loop name compute_rhs#13 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#13#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#13#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((u[m][i][j-2][k]-(4.0*u[m][i][j-1][k]))+(5.0*u[m][i][j][k]))));
			}
		}
	}
	
	/*
	--------------------------------------------------------------------
	c      compute zeta-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#14 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name compute_rhs#14#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name compute_rhs#14#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				wijk=ws[i][j][k];
				wp1=ws[i][j][k+1];
				wm1=ws[i][j][k-1];
				rhs[0][i][j][k]=((rhs[0][i][j][k]+(dz1tz1*((u[0][i][j][k+1]-(2.0*u[0][i][j][k]))+u[0][i][j][k-1])))-(tz2*(u[3][i][j][k+1]-u[3][i][j][k-1])));
				rhs[1][i][j][k]=(((rhs[1][i][j][k]+(dz2tz1*((u[1][i][j][k+1]-(2.0*u[1][i][j][k]))+u[1][i][j][k-1])))+(zzcon2*((us[i][j][k+1]-(2.0*us[i][j][k]))+us[i][j][k-1])))-(tz2*((u[1][i][j][k+1]*wp1)-(u[1][i][j][k-1]*wm1))));
				rhs[2][i][j][k]=(((rhs[2][i][j][k]+(dz3tz1*((u[2][i][j][k+1]-(2.0*u[2][i][j][k]))+u[2][i][j][k-1])))+(zzcon2*((vs[i][j][k+1]-(2.0*vs[i][j][k]))+vs[i][j][k-1])))-(tz2*((u[2][i][j][k+1]*wp1)-(u[2][i][j][k-1]*wm1))));
				rhs[3][i][j][k]=(((rhs[3][i][j][k]+(dz4tz1*((u[3][i][j][k+1]-(2.0*u[3][i][j][k]))+u[3][i][j][k-1])))+((zzcon2*con43)*((wp1-(2.0*wijk))+wm1)))-(tz2*(((u[3][i][j][k+1]*wp1)-(u[3][i][j][k-1]*wm1))+((((u[4][i][j][k+1]-square[i][j][k+1])-u[4][i][j][k-1])+square[i][j][k-1])*c2))));
				rhs[4][i][j][k]=(((((rhs[4][i][j][k]+(dz5tz1*((u[4][i][j][k+1]-(2.0*u[4][i][j][k]))+u[4][i][j][k-1])))+(zzcon3*((qs[i][j][k+1]-(2.0*qs[i][j][k]))+qs[i][j][k-1])))+(zzcon4*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(zzcon5*(((u[4][i][j][k+1]*rho_i[i][j][k+1])-((2.0*u[4][i][j][k])*rho_i[i][j][k]))+(u[4][i][j][k-1]*rho_i[i][j][k-1]))))-(tz2*((((c1*u[4][i][j][k+1])-(c2*square[i][j][k+1]))*wp1)-(((c1*u[4][i][j][k-1])-(c2*square[i][j][k-1]))*wm1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      add fourth order zeta-direction dissipation                
	c-------------------------------------------------------------------
	*/
	k=1;
	#pragma loop name compute_rhs#15 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#15#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#15#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((5.0*u[m][i][j][k])-(4.0*u[m][i][j][k+1]))+u[m][i][j][k+2])));
			}
		}
	}
	k=2;
	#pragma loop name compute_rhs#16 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#16#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#16#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((( - 4.0)*u[m][i][j][k-1])+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][k+1]))+u[m][i][j][k+2])));
			}
		}
	}
	#pragma loop name compute_rhs#17 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#17#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#17#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name compute_rhs#17#0#0#0 
				for (k=(3*1); k<=((grid_points[2]-(3*1))-1); k ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((((u[m][i][j][k-2]-(4.0*u[m][i][j][k-1]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][k+1]))+u[m][i][j][k+2])));
				}
			}
		}
	}
	k=(grid_points[2]-3);
	#pragma loop name compute_rhs#18 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#18#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#18#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*(((u[m][i][j][k-2]-(4.0*u[m][i][j][k-1]))+(6.0*u[m][i][j][k]))-(4.0*u[m][i][j][k+1]))));
			}
		}
	}
	k=(grid_points[2]-2);
	#pragma loop name compute_rhs#19 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#19#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#19#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(dssp*((u[m][i][j][k-2]-(4.0*u[m][i][j][k-1]))+(5.0*u[m][i][j][k]))));
			}
		}
	}
	#pragma loop name compute_rhs#20 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name compute_rhs#20#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name compute_rhs#20#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name compute_rhs#20#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=(rhs[m][i][j][k]*dt);
				}
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void set_constants(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	ce[0][0]=2.0;
	ce[1][0]=0.0;
	ce[2][0]=0.0;
	ce[3][0]=4.0;
	ce[4][0]=5.0;
	ce[5][0]=3.0;
	ce[6][0]=0.5;
	ce[7][0]=0.02;
	ce[8][0]=0.01;
	ce[9][0]=0.03;
	ce[10][0]=0.5;
	ce[11][0]=0.4;
	ce[12][0]=0.3;
	ce[0][1]=1.0;
	ce[1][1]=0.0;
	ce[2][1]=0.0;
	ce[3][1]=0.0;
	ce[4][1]=1.0;
	ce[5][1]=2.0;
	ce[6][1]=3.0;
	ce[7][1]=0.01;
	ce[8][1]=0.03;
	ce[9][1]=0.02;
	ce[10][1]=0.4;
	ce[11][1]=0.3;
	ce[12][1]=0.5;
	ce[0][2]=2.0;
	ce[1][2]=2.0;
	ce[2][2]=0.0;
	ce[3][2]=0.0;
	ce[4][2]=0.0;
	ce[5][2]=2.0;
	ce[6][2]=3.0;
	ce[7][2]=0.04;
	ce[8][2]=0.03;
	ce[9][2]=0.05;
	ce[10][2]=0.3;
	ce[11][2]=0.5;
	ce[12][2]=0.4;
	ce[0][3]=2.0;
	ce[1][3]=2.0;
	ce[2][3]=0.0;
	ce[3][3]=0.0;
	ce[4][3]=0.0;
	ce[5][3]=2.0;
	ce[6][3]=3.0;
	ce[7][3]=0.03;
	ce[8][3]=0.05;
	ce[9][3]=0.04;
	ce[10][3]=0.2;
	ce[11][3]=0.1;
	ce[12][3]=0.3;
	ce[0][4]=5.0;
	ce[1][4]=4.0;
	ce[2][4]=3.0;
	ce[3][4]=2.0;
	ce[4][4]=0.1;
	ce[5][4]=0.4;
	ce[6][4]=0.3;
	ce[7][4]=0.05;
	ce[8][4]=0.04;
	ce[9][4]=0.03;
	ce[10][4]=0.1;
	ce[11][4]=0.3;
	ce[12][4]=0.2;
	c1=1.4;
	c2=0.4;
	c3=0.1;
	c4=1.0;
	c5=1.4;
	bt=sqrt(0.5);
	dnxm1=(1.0/((double)(grid_points[0]-1)));
	dnym1=(1.0/((double)(grid_points[1]-1)));
	dnzm1=(1.0/((double)(grid_points[2]-1)));
	c1c2=(c1*c2);
	c1c5=(c1*c5);
	c3c4=(c3*c4);
	c1345=(c1c5*c3c4);
	conz1=(1.0-c1c5);
	tx1=(1.0/(dnxm1*dnxm1));
	tx2=(1.0/(2.0*dnxm1));
	tx3=(1.0/dnxm1);
	ty1=(1.0/(dnym1*dnym1));
	ty2=(1.0/(2.0*dnym1));
	ty3=(1.0/dnym1);
	tz1=(1.0/(dnzm1*dnzm1));
	tz2=(1.0/(2.0*dnzm1));
	tz3=(1.0/dnzm1);
	dx1=0.75;
	dx2=0.75;
	dx3=0.75;
	dx4=0.75;
	dx5=0.75;
	dy1=0.75;
	dy2=0.75;
	dy3=0.75;
	dy4=0.75;
	dy5=0.75;
	dz1=1.0;
	dz2=1.0;
	dz3=1.0;
	dz4=1.0;
	dz5=1.0;
	dxmax=((dx3>dx4) ? dx3 : dx4);
	dymax=((dy2>dy4) ? dy2 : dy4);
	dzmax=((dz2>dz3) ? dz2 : dz3);
	dssp=(0.25*((dx1>((dy1>dz1) ? dy1 : dz1)) ? dx1 : ((dy1>dz1) ? dy1 : dz1)));
	c4dssp=(4.0*dssp);
	c5dssp=(5.0*dssp);
	dttx1=(dt*tx1);
	dttx2=(dt*tx2);
	dtty1=(dt*ty1);
	dtty2=(dt*ty2);
	dttz1=(dt*tz1);
	dttz2=(dt*tz2);
	c2dttx1=(2.0*dttx1);
	c2dtty1=(2.0*dtty1);
	c2dttz1=(2.0*dttz1);
	dtdssp=(dt*dssp);
	comz1=dtdssp;
	comz4=(4.0*dtdssp);
	comz5=(5.0*dtdssp);
	comz6=(6.0*dtdssp);
	c3c4tx3=(c3c4*tx3);
	c3c4ty3=(c3c4*ty3);
	c3c4tz3=(c3c4*tz3);
	dx1tx1=(dx1*tx1);
	dx2tx1=(dx2*tx1);
	dx3tx1=(dx3*tx1);
	dx4tx1=(dx4*tx1);
	dx5tx1=(dx5*tx1);
	dy1ty1=(dy1*ty1);
	dy2ty1=(dy2*ty1);
	dy3ty1=(dy3*ty1);
	dy4ty1=(dy4*ty1);
	dy5ty1=(dy5*ty1);
	dz1tz1=(dz1*tz1);
	dz2tz1=(dz2*tz1);
	dz3tz1=(dz3*tz1);
	dz4tz1=(dz4*tz1);
	dz5tz1=(dz5*tz1);
	c2iv=2.5;
	con43=(4.0/3.0);
	con16=(1.0/6.0);
	xxcon1=((c3c4tx3*con43)*tx3);
	xxcon2=(c3c4tx3*tx3);
	xxcon3=((c3c4tx3*conz1)*tx3);
	xxcon4=((c3c4tx3*con16)*tx3);
	xxcon5=((c3c4tx3*c1c5)*tx3);
	yycon1=((c3c4ty3*con43)*ty3);
	yycon2=(c3c4ty3*ty3);
	yycon3=((c3c4ty3*conz1)*ty3);
	yycon4=((c3c4ty3*con16)*ty3);
	yycon5=((c3c4ty3*c1c5)*ty3);
	zzcon1=((c3c4tz3*con43)*tz3);
	zzcon2=(c3c4tz3*tz3);
	zzcon3=((c3c4tz3*conz1)*tz3);
	zzcon4=((c3c4tz3*con16)*tz3);
	zzcon5=((c3c4tz3*c1c5)*tz3);
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void txinvr(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c block-diagonal matrix-vector multiplication                  
	--------------------------------------------------------------------
	*/
	int i, j, k;
	double t1, t2, t3, ac, ru1, uu, vv, ww, r1, r2, r3, r4, r5, ac2inv;
	#pragma loop name txinvr#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name txinvr#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name txinvr#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				ru1=rho_i[i][j][k];
				uu=us[i][j][k];
				vv=vs[i][j][k];
				ww=ws[i][j][k];
				ac=speed[i][j][k];
				ac2inv=(ainv[i][j][k]*ainv[i][j][k]);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				t1=((c2*ac2inv)*(((((qs[i][j][k]*r1)-(uu*r2))-(vv*r3))-(ww*r4))+r5));
				t2=((bt*ru1)*((uu*r1)-r2));
				t3=(((bt*ru1)*ac)*t1);
				rhs[0][i][j][k]=(r1-t1);
				rhs[1][i][j][k]=(( - ru1)*((ww*r1)-r4));
				rhs[2][i][j][k]=(ru1*((vv*r1)-r3));
				rhs[3][i][j][k]=(( - t2)+t3);
				rhs[4][i][j][k]=(t2+t3);
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void tzetar(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c   block-diagonal matrix-vector multiplication                       
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	double t1, t2, t3, ac, xvel, yvel, zvel, r1, r2, r3, r4, r5, btuz, acinv, ac2u, uzik1;
	#pragma loop name tzetar#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name tzetar#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name tzetar#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				xvel=us[i][j][k];
				yvel=vs[i][j][k];
				zvel=ws[i][j][k];
				ac=speed[i][j][k];
				acinv=ainv[i][j][k];
				ac2u=(ac*ac);
				r1=rhs[0][i][j][k];
				r2=rhs[1][i][j][k];
				r3=rhs[2][i][j][k];
				r4=rhs[3][i][j][k];
				r5=rhs[4][i][j][k];
				uzik1=u[0][i][j][k];
				btuz=(bt*uzik1);
				t1=((btuz*acinv)*(r4+r5));
				t2=(r3+t1);
				t3=(btuz*(r4-r5));
				rhs[0][i][j][k]=t2;
				rhs[1][i][j][k]=((( - uzik1)*r2)+(xvel*t2));
				rhs[2][i][j][k]=((uzik1*r1)+(yvel*t2));
				rhs[3][i][j][k]=((zvel*t2)+t3);
				rhs[4][i][j][k]=((((uzik1*((( - xvel)*r2)+(yvel*r1)))+(qs[i][j][k]*t2))+((c2iv*ac2u)*t1))+(zvel*t3));
			}
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void verify(int no_time_steps, char * class, boolean * verified)
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c  verification routine                         
	--------------------------------------------------------------------
	*/
	double xcrref[5], xceref[5], xcrdif[5], xcedif[5], epsilon, xce[5], xcr[5], dtref;
	int m;
	/*
	--------------------------------------------------------------------
	c   tolerance level
	--------------------------------------------------------------------
	*/
	epsilon=1.0E-8;
	/*
	--------------------------------------------------------------------
	c   compute the error norm and the residual norm, and exit if not printing
	--------------------------------------------------------------------
	*/
	error_norm(xce);
	compute_rhs();
	rhs_norm(xcr);
	#pragma loop name verify#0 
	#pragma cetus parallel 
	#pragma omp parallel for
	for (m=0; m<5; m ++ )
	{
		xcr[m]=(xcr[m]/dt);
	}
	( * class)='U';
	( * verified)=1;
	#pragma loop name verify#1 
	for (m=0; m<5; m ++ )
	{
		xcrref[m]=1.0;
		xceref[m]=1.0;
	}
	/*
	--------------------------------------------------------------------
	c    reference data for 12X12X12 grids after 100 time steps, with DT = 1.50d-02
	--------------------------------------------------------------------
	*/
	if ((((grid_points[0]==12)&&(grid_points[1]==12))&&(grid_points[2]==12))&&(no_time_steps==100))
	{
		( * class)='S';
		dtref=0.015;
		/*
		--------------------------------------------------------------------
		c    Reference values of RMS-norms of residual.
		--------------------------------------------------------------------
		*/
		xcrref[0]=0.02747031545133948;
		xcrref[1]=0.010360746705285417;
		xcrref[2]=0.016235745065095532;
		xcrref[3]=0.015840557224455615;
		xcrref[4]=0.03484904060936246;
		/*
		--------------------------------------------------------------------
		c    Reference values of RMS-norms of solution error.
		--------------------------------------------------------------------
		*/
		xceref[0]=2.7289258557377225E-5;
		xceref[1]=1.0364446640837285E-5;
		xceref[2]=1.615479828716647E-5;
		xceref[3]=1.57507049944801E-5;
		xceref[4]=3.417766618339053E-5;
		/*
		--------------------------------------------------------------------
		c    reference data for 36X36X36 grids after 400 time steps, with DT = 1.5d-03
		--------------------------------------------------------------------
		*/
	}
	else
	{
		if ((((grid_points[0]==36)&&(grid_points[1]==36))&&(grid_points[2]==36))&&(no_time_steps==400))
		{
			( * class)='W';
			dtref=0.0015;
			/*
			--------------------------------------------------------------------
			c    Reference values of RMS-norms of residual.
			--------------------------------------------------------------------
			*/
			xcrref[0]=0.001893253733584;
			xcrref[1]=1.717075447775E-4;
			xcrref[2]=2.778153350936E-4;
			xcrref[3]=2.887475409984E-4;
			xcrref[4]=0.003143611161242;
			/*
			--------------------------------------------------------------------
			c    Reference values of RMS-norms of solution error.
			--------------------------------------------------------------------
			*/
			xceref[0]=7.542088599534E-5;
			xceref[1]=6.512852253086E-6;
			xceref[2]=1.049092285688E-5;
			xceref[3]=1.128838671535E-5;
			xceref[4]=1.212845639773E-4;
			/*
			--------------------------------------------------------------------
			c    reference data for 64X64X64 grids after 400 time steps, with DT = 1.5d-03
			--------------------------------------------------------------------
			*/
		}
		else
		{
			if ((((grid_points[0]==64)&&(grid_points[1]==64))&&(grid_points[2]==64))&&(no_time_steps==400))
			{
				( * class)='A';
				dtref=0.0015;
				/*
				--------------------------------------------------------------------
				c    Reference values of RMS-norms of residual.
				--------------------------------------------------------------------
				*/
				xcrref[0]=2.4799822399300195;
				xcrref[1]=1.1276337964368832;
				xcrref[2]=1.5028977888770492;
				xcrref[3]=1.421781621169518;
				xcrref[4]=2.129211303513828;
				/*
				--------------------------------------------------------------------
				c    Reference values of RMS-norms of solution error.
				--------------------------------------------------------------------
				*/
				xceref[0]=1.090014029782055E-4;
				xceref[1]=3.734395176928209E-5;
				xceref[2]=5.009278540654163E-5;
				xceref[3]=4.767109393952825E-5;
				xceref[4]=1.3621613399213E-4;
				/*
				--------------------------------------------------------------------
				c    reference data for 102X102X102 grids after 400 time steps,
				c    with DT = 1.0d-03
				--------------------------------------------------------------------
				*/
			}
			else
			{
				if ((((grid_points[0]==102)&&(grid_points[1]==102))&&(grid_points[2]==102))&&(no_time_steps==400))
				{
					( * class)='B';
					dtref=0.001;
					/*
					--------------------------------------------------------------------
					c    Reference values of RMS-norms of residual.
					--------------------------------------------------------------------
					*/
					xcrref[0]=69.03293579998;
					xcrref[1]=30.95134488084;
					xcrref[2]=41.03336647017;
					xcrref[3]=38.64769009604;
					xcrref[4]=56.43482272596;
					/*
					--------------------------------------------------------------------
					c    Reference values of RMS-norms of solution error.
					--------------------------------------------------------------------
					*/
					xceref[0]=0.009810006190188;
					xceref[1]=0.00102282790567;
					xceref[2]=0.001720597911692;
					xceref[3]=0.001694479428231;
					xceref[4]=0.01847456263981;
					/*
					--------------------------------------------------------------------
					c    reference data for 162X162X162 grids after 400 time steps,
					c    with DT = 0.67d-03
					--------------------------------------------------------------------
					*/
				}
				else
				{
					if ((((grid_points[0]==162)&&(grid_points[1]==162))&&(grid_points[2]==162))&&(no_time_steps==400))
					{
						( * class)='C';
						dtref=6.7E-4;
						/*
						--------------------------------------------------------------------
						c    Reference values of RMS-norms of residual.
						--------------------------------------------------------------------
						*/
						xcrref[0]=588.1691581829;
						xcrref[1]=245.4417603569;
						xcrref[2]=329.3829191851;
						xcrref[3]=308.1924971891;
						xcrref[4]=459.7223799176;
						/*
						--------------------------------------------------------------------
						c    Reference values of RMS-norms of solution error.
						--------------------------------------------------------------------
						*/
						xceref[0]=0.2598120500183;
						xceref[1]=0.02590888922315;
						xceref[2]=0.0513288641632;
						xceref[3]=0.04806073419454;
						xceref[4]=0.5483377491301;
					}
					else
					{
						( * verified)=0;
					}
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c    verification test for residuals if gridsize is either 12X12X12 or 
	c    64X64X64 or 102X102X102 or 162X162X162
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c    Compute the difference of solution values and the known reference values.
	--------------------------------------------------------------------
	*/
	#pragma loop name verify#2 
	for (m=0; m<5; m ++ )
	{
		xcrdif[m]=fabs((xcr[m]-xcrref[m])/xcrref[m]);
		xcedif[m]=fabs((xce[m]-xceref[m])/xceref[m]);
	}
	/*
	--------------------------------------------------------------------
	c    Output the comparison of computed results to known cases.
	--------------------------------------------------------------------
	*/
	if (( * class)!='U')
	{
		printf(" Verification being performed for class %1c\n",  * class);
		printf(" accuracy setting for epsilon = %20.13e\n", epsilon);
		if (fabs(dt-dtref)>epsilon)
		{
			( * verified)=0;
			( * class)='U';
			printf(" DT does not match the reference value of %15.8e\n", dtref);
		}
	}
	else
	{
		printf(" Unknown class\n");
	}
	if (( * class)!='U')
	{
		printf(" Comparison of RMS-norms of residual\n");
	}
	else
	{
		printf(" RMS-norms of residual\n");
	}
	#pragma loop name verify#3 
	for (m=0; m<5; m ++ )
	{
		if (( * class)=='U')
		{
			printf("          %2d%20.13e\n", m, xcr[m]);
		}
		else
		{
			if (xcrdif[m]>epsilon)
			{
				( * verified)=0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n", m, xcr[m], xcrref[m], xcrdif[m]);
			}
			else
			{
				printf("          %2d%20.13e%20.13e%20.13e\n", m, xcr[m], xcrref[m], xcrdif[m]);
			}
		}
	}
	if (( * class)!='U')
	{
		printf(" Comparison of RMS-norms of solution error\n");
	}
	else
	{
		printf(" RMS-norms of solution error\n");
	}
	#pragma loop name verify#4 
	for (m=0; m<5; m ++ )
	{
		if (( * class)=='U')
		{
			printf("          %2d%20.13e\n", m, xce[m]);
		}
		else
		{
			if (xcedif[m]>epsilon)
			{
				( * verified)=0;
				printf(" FAILURE: %2d%20.13e%20.13e%20.13e\n", m, xce[m], xceref[m], xcedif[m]);
			}
			else
			{
				printf("          %2d%20.13e%20.13e%20.13e\n", m, xce[m], xceref[m], xcedif[m]);
			}
		}
	}
	if (( * class)=='U')
	{
		printf(" No reference values provided\n");
		printf(" No verification performed\n");
	}
	else
	{
		if ( * verified)
		{
			printf(" Verification Successful\n");
		}
		else
		{
			printf(" Verification failed\n");
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void x_solve(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c this function performs the solution of the approximate factorization
	c step in the x-direction for all five matrix components
	c simultaneously. The Thomas algorithm is employed to solve the
	c systems for the x-lines. Boundary conditions are non-periodic
	--------------------------------------------------------------------
	*/
	int i, j, k, n, i1, i2, m;
	double fac1, fac2;
	/*
	--------------------------------------------------------------------
	c                          FORWARD ELIMINATION  
	--------------------------------------------------------------------
	*/
	lhsx();
	/*
	--------------------------------------------------------------------
	c      perform the Thomas algorithm; first, FORWARD ELIMINATION     
	--------------------------------------------------------------------
	*/
	n=0;
	#pragma loop name x_solve#0 
	for (i=0; i<=(grid_points[0]-3); i ++ )
	{
		i1=(i+1);
		i2=(i+2);
		#pragma loop name x_solve#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name x_solve#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				#pragma loop name x_solve#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				}
				lhs[n+2][i1][j][k]=(lhs[n+2][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+3][i][j][k]));
				lhs[n+3][i1][j][k]=(lhs[n+3][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+4][i][j][k]));
				#pragma loop name x_solve#0#0#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i1][j][k]=(rhs[m][i1][j][k]-(lhs[n+1][i1][j][k]*rhs[m][i][j][k]));
				}
				lhs[n+1][i2][j][k]=(lhs[n+1][i2][j][k]-(lhs[n+0][i2][j][k]*lhs[n+3][i][j][k]));
				lhs[n+2][i2][j][k]=(lhs[n+2][i2][j][k]-(lhs[n+0][i2][j][k]*lhs[n+4][i][j][k]));
				#pragma loop name x_solve#0#0#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i2][j][k]=(rhs[m][i2][j][k]-(lhs[n+0][i2][j][k]*rhs[m][i][j][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      The last two rows in this grid block are a bit different, 
	c      since they do not have two more rows available for the
	c      elimination of off-diagonal entries
	--------------------------------------------------------------------
	*/
	i=(grid_points[0]-2);
	i1=(grid_points[0]-1);
	#pragma loop name x_solve#1 
	for (j=1; j<=(grid_points[1]-2); j ++ )
	{
		#pragma loop name x_solve#1#0 
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			fac1=(1.0/lhs[n+2][i][j][k]);
			lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
			lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
			#pragma loop name x_solve#1#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
			}
			lhs[n+2][i1][j][k]=(lhs[n+2][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+3][i][j][k]));
			lhs[n+3][i1][j][k]=(lhs[n+3][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+4][i][j][k]));
			#pragma loop name x_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1][j][k]=(rhs[m][i1][j][k]-(lhs[n+1][i1][j][k]*rhs[m][i][j][k]));
			}
			/*
			--------------------------------------------------------------------
			c            scale the last row immediately 
			--------------------------------------------------------------------
			*/
			fac2=(1.0/lhs[n+2][i1][j][k]);
			#pragma loop name x_solve#1#0#2 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i1][j][k]=(fac2*rhs[m][i1][j][k]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      do the u+c and the u-c factors                 
	--------------------------------------------------------------------
	*/
	#pragma loop name x_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name x_solve#2#0 
		for (i=0; i<=(grid_points[0]-3); i ++ )
		{
			i1=(i+1);
			i2=(i+2);
			#pragma loop name x_solve#2#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name x_solve#2#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					fac1=(1.0/lhs[n+2][i][j][k]);
					lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
					lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
					lhs[n+2][i1][j][k]=(lhs[n+2][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+3][i][j][k]));
					lhs[n+3][i1][j][k]=(lhs[n+3][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+4][i][j][k]));
					rhs[m][i1][j][k]=(rhs[m][i1][j][k]-(lhs[n+1][i1][j][k]*rhs[m][i][j][k]));
					lhs[n+1][i2][j][k]=(lhs[n+1][i2][j][k]-(lhs[n+0][i2][j][k]*lhs[n+3][i][j][k]));
					lhs[n+2][i2][j][k]=(lhs[n+2][i2][j][k]-(lhs[n+0][i2][j][k]*lhs[n+4][i][j][k]));
					rhs[m][i2][j][k]=(rhs[m][i2][j][k]-(lhs[n+0][i2][j][k]*rhs[m][i][j][k]));
				}
			}
		}
		/*
		--------------------------------------------------------------------
		c         And again the last two rows separately
		--------------------------------------------------------------------
		*/
		i=(grid_points[0]-2);
		i1=(grid_points[0]-1);
		#pragma loop name x_solve#2#1 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name x_solve#2#1#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				lhs[n+2][i1][j][k]=(lhs[n+2][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+3][i][j][k]));
				lhs[n+3][i1][j][k]=(lhs[n+3][i1][j][k]-(lhs[n+1][i1][j][k]*lhs[n+4][i][j][k]));
				rhs[m][i1][j][k]=(rhs[m][i1][j][k]-(lhs[n+1][i1][j][k]*rhs[m][i][j][k]));
				/*
				--------------------------------------------------------------------
				c               Scale the last row immediately
				--------------------------------------------------------------------
				*/
				fac2=(1.0/lhs[n+2][i1][j][k]);
				rhs[m][i1][j][k]=(fac2*rhs[m][i1][j][k]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c                         BACKSUBSTITUTION 
	--------------------------------------------------------------------
	*/
	i=(grid_points[0]-2);
	i1=(grid_points[0]-1);
	n=0;
	#pragma loop name x_solve#3 
	for (m=0; m<3; m ++ )
	{
		#pragma loop name x_solve#3#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name x_solve#3#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i1][j][k]));
			}
		}
	}
	#pragma loop name x_solve#4 
	for (m=3; m<5; m ++ )
	{
		#pragma loop name x_solve#4#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name x_solve#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				n=(((m-3)+1)*5);
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i1][j][k]));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      The first three factors
	--------------------------------------------------------------------
	*/
	n=0;
	#pragma loop name x_solve#5 
	for (i=(grid_points[0]-3); i>=0; i -- )
	{
		i1=(i+1);
		i2=(i+2);
		#pragma loop name x_solve#5#0 
		for (m=0; m<3; m ++ )
		{
			#pragma loop name x_solve#5#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name x_solve#5#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i1][j][k]))-(lhs[n+4][i][j][k]*rhs[m][i2][j][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      And the remaining two
	--------------------------------------------------------------------
	*/
	#pragma loop name x_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name x_solve#6#0 
		for (i=(grid_points[0]-3); i>=0; i -- )
		{
			i1=(i+1);
			i2=(i+2);
			#pragma loop name x_solve#6#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name x_solve#6#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i1][j][k]))-(lhs[n+4][i][j][k]*rhs[m][i2][j][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      Do the block-diagonal inversion          
	--------------------------------------------------------------------
	*/
	ninvr();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void y_solve(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c this function performs the solution of the approximate factorization
	c step in the y-direction for all five matrix components
	c simultaneously. The Thomas algorithm is employed to solve the
	c systems for the y-lines. Boundary conditions are non-periodic
	--------------------------------------------------------------------
	*/
	int i, j, k, n, j1, j2, m;
	double fac1, fac2;
	/*
	--------------------------------------------------------------------
	c                          FORWARD ELIMINATION  
	--------------------------------------------------------------------
	*/
	lhsy();
	n=0;
	#pragma loop name y_solve#0 
	for (j=0; j<=(grid_points[1]-3); j ++ )
	{
		j1=(j+1);
		j2=(j+2);
		#pragma loop name y_solve#0#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name y_solve#0#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				#pragma loop name y_solve#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				}
				lhs[n+2][i][j1][k]=(lhs[n+2][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+3][i][j][k]));
				lhs[n+3][i][j1][k]=(lhs[n+3][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+4][i][j][k]));
				#pragma loop name y_solve#0#0#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j1][k]=(rhs[m][i][j1][k]-(lhs[n+1][i][j1][k]*rhs[m][i][j][k]));
				}
				lhs[n+1][i][j2][k]=(lhs[n+1][i][j2][k]-(lhs[n+0][i][j2][k]*lhs[n+3][i][j][k]));
				lhs[n+2][i][j2][k]=(lhs[n+2][i][j2][k]-(lhs[n+0][i][j2][k]*lhs[n+4][i][j][k]));
				#pragma loop name y_solve#0#0#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j2][k]=(rhs[m][i][j2][k]-(lhs[n+0][i][j2][k]*rhs[m][i][j][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      The last two rows in this grid block are a bit different, 
	c      since they do not have two more rows available for the
	c      elimination of off-diagonal entries
	--------------------------------------------------------------------
	*/
	j=(grid_points[1]-2);
	j1=(grid_points[1]-1);
	#pragma loop name y_solve#1 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name y_solve#1#0 
		for (k=1; k<=(grid_points[2]-2); k ++ )
		{
			fac1=(1.0/lhs[n+2][i][j][k]);
			lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
			lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
			#pragma loop name y_solve#1#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
			}
			lhs[n+2][i][j1][k]=(lhs[n+2][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+3][i][j][k]));
			lhs[n+3][i][j1][k]=(lhs[n+3][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+4][i][j][k]));
			#pragma loop name y_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1][k]=(rhs[m][i][j1][k]-(lhs[n+1][i][j1][k]*rhs[m][i][j][k]));
			}
			/*
			--------------------------------------------------------------------
			c            scale the last row immediately 
			--------------------------------------------------------------------
			*/
			fac2=(1.0/lhs[n+2][i][j1][k]);
			#pragma loop name y_solve#1#0#2 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j1][k]=(fac2*rhs[m][i][j1][k]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      do the u+c and the u-c factors                 
	--------------------------------------------------------------------
	*/
	#pragma loop name y_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name y_solve#2#0 
		for (j=0; j<=(grid_points[1]-3); j ++ )
		{
			j1=(j+1);
			j2=(j+2);
			#pragma loop name y_solve#2#0#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				#pragma loop name y_solve#2#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					fac1=(1.0/lhs[n+2][i][j][k]);
					lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
					lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
					lhs[n+2][i][j1][k]=(lhs[n+2][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+3][i][j][k]));
					lhs[n+3][i][j1][k]=(lhs[n+3][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+4][i][j][k]));
					rhs[m][i][j1][k]=(rhs[m][i][j1][k]-(lhs[n+1][i][j1][k]*rhs[m][i][j][k]));
					lhs[n+1][i][j2][k]=(lhs[n+1][i][j2][k]-(lhs[n+0][i][j2][k]*lhs[n+3][i][j][k]));
					lhs[n+2][i][j2][k]=(lhs[n+2][i][j2][k]-(lhs[n+0][i][j2][k]*lhs[n+4][i][j][k]));
					rhs[m][i][j2][k]=(rhs[m][i][j2][k]-(lhs[n+0][i][j2][k]*rhs[m][i][j][k]));
				}
			}
		}
		/*
		--------------------------------------------------------------------
		c         And again the last two rows separately
		--------------------------------------------------------------------
		*/
		j=(grid_points[1]-2);
		j1=(grid_points[1]-1);
		#pragma loop name y_solve#2#1 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name y_solve#2#1#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				lhs[n+2][i][j1][k]=(lhs[n+2][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+3][i][j][k]));
				lhs[n+3][i][j1][k]=(lhs[n+3][i][j1][k]-(lhs[n+1][i][j1][k]*lhs[n+4][i][j][k]));
				rhs[m][i][j1][k]=(rhs[m][i][j1][k]-(lhs[n+1][i][j1][k]*rhs[m][i][j][k]));
				/*
				--------------------------------------------------------------------
				c               Scale the last row immediately 
				--------------------------------------------------------------------
				*/
				fac2=(1.0/lhs[n+2][i][j1][k]);
				rhs[m][i][j1][k]=(fac2*rhs[m][i][j1][k]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c                         BACKSUBSTITUTION 
	--------------------------------------------------------------------
	*/
	j=(grid_points[1]-2);
	j1=(grid_points[1]-1);
	n=0;
	#pragma loop name y_solve#3 
	for (m=0; m<3; m ++ )
	{
		#pragma loop name y_solve#3#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name y_solve#3#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j1][k]));
			}
		}
	}
	#pragma loop name y_solve#4 
	for (m=3; m<5; m ++ )
	{
		#pragma loop name y_solve#4#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name y_solve#4#0#0 
			for (k=1; k<=(grid_points[2]-2); k ++ )
			{
				n=(((m-3)+1)*5);
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j1][k]));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      The first three factors
	--------------------------------------------------------------------
	*/
	n=0;
	#pragma loop name y_solve#5 
	for (m=0; m<3; m ++ )
	{
		#pragma loop name y_solve#5#0 
		for (j=(grid_points[1]-3); j>=0; j -- )
		{
			j1=(j+1);
			j2=(j+2);
			#pragma loop name y_solve#5#0#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				#pragma loop name y_solve#5#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j1][k]))-(lhs[n+4][i][j][k]*rhs[m][i][j2][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      And the remaining two
	--------------------------------------------------------------------
	*/
	#pragma loop name y_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name y_solve#6#0 
		for (j=(grid_points[1]-3); j>=0; j -- )
		{
			j1=(j+1);
			j2=(j1+1);
			#pragma loop name y_solve#6#0#0 
			for (i=1; i<=(grid_points[0]-2); i ++ )
			{
				#pragma loop name y_solve#6#0#0#0 
				for (k=1; k<=(grid_points[2]-2); k ++ )
				{
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j1][k]))-(lhs[n+4][i][j][k]*rhs[m][i][j2][k]));
				}
			}
		}
	}
	pinvr();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void z_solve(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c this function performs the solution of the approximate factorization
	c step in the z-direction for all five matrix components
	c simultaneously. The Thomas algorithm is employed to solve the
	c systems for the z-lines. Boundary conditions are non-periodic
	c-------------------------------------------------------------------
	*/
	int i, j, k, n, k1, k2, m;
	double fac1, fac2;
	/*
	--------------------------------------------------------------------
	c                          FORWARD ELIMINATION  
	c-------------------------------------------------------------------
	*/
	lhsz();
	n=0;
	#pragma loop name z_solve#0 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name z_solve#0#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			#pragma loop name z_solve#0#0#0 
			for (k=0; k<=(grid_points[2]-3); k ++ )
			{
				k1=(k+1);
				k2=(k+2);
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				#pragma loop name z_solve#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				}
				lhs[n+2][i][j][k1]=(lhs[n+2][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+3][i][j][k]));
				lhs[n+3][i][j][k1]=(lhs[n+3][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+4][i][j][k]));
				#pragma loop name z_solve#0#0#0#1 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[n+1][i][j][k1]*rhs[m][i][j][k]));
				}
				lhs[n+1][i][j][k2]=(lhs[n+1][i][j][k2]-(lhs[n+0][i][j][k2]*lhs[n+3][i][j][k]));
				lhs[n+2][i][j][k2]=(lhs[n+2][i][j][k2]-(lhs[n+0][i][j][k2]*lhs[n+4][i][j][k]));
				#pragma loop name z_solve#0#0#0#2 
				for (m=0; m<3; m ++ )
				{
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[n+0][i][j][k2]*rhs[m][i][j][k]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      The last two rows in this grid block are a bit different, 
	c      since they do not have two more rows available for the
	c      elimination of off-diagonal entries
	c-------------------------------------------------------------------
	*/
	k=(grid_points[2]-2);
	k1=(grid_points[2]-1);
	#pragma loop name z_solve#1 
	for (i=1; i<=(grid_points[0]-2); i ++ )
	{
		#pragma loop name z_solve#1#0 
		for (j=1; j<=(grid_points[1]-2); j ++ )
		{
			fac1=(1.0/lhs[n+2][i][j][k]);
			lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
			lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
			#pragma loop name z_solve#1#0#0 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
			}
			lhs[n+2][i][j][k1]=(lhs[n+2][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+3][i][j][k]));
			lhs[n+3][i][j][k1]=(lhs[n+3][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+4][i][j][k]));
			#pragma loop name z_solve#1#0#1 
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[n+1][i][j][k1]*rhs[m][i][j][k]));
			}
			/*
			--------------------------------------------------------------------
			c               scale the last row immediately
			c-------------------------------------------------------------------
			*/
			fac2=(1.0/lhs[n+2][i][j][k1]);
			#pragma loop name z_solve#1#0#2 
			#pragma cetus parallel 
			#pragma omp parallel for
			for (m=0; m<3; m ++ )
			{
				rhs[m][i][j][k1]=(fac2*rhs[m][i][j][k1]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      do the u+c and the u-c factors               
	c-------------------------------------------------------------------
	*/
	#pragma loop name z_solve#2 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name z_solve#2#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#2#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name z_solve#2#0#0#0 
				for (k=0; k<=(grid_points[2]-3); k ++ )
				{
					k1=(k+1);
					k2=(k+2);
					fac1=(1.0/lhs[n+2][i][j][k]);
					lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
					lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
					rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
					lhs[n+2][i][j][k1]=(lhs[n+2][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+3][i][j][k]));
					lhs[n+3][i][j][k1]=(lhs[n+3][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+4][i][j][k]));
					rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[n+1][i][j][k1]*rhs[m][i][j][k]));
					lhs[n+1][i][j][k2]=(lhs[n+1][i][j][k2]-(lhs[n+0][i][j][k2]*lhs[n+3][i][j][k]));
					lhs[n+2][i][j][k2]=(lhs[n+2][i][j][k2]-(lhs[n+0][i][j][k2]*lhs[n+4][i][j][k]));
					rhs[m][i][j][k2]=(rhs[m][i][j][k2]-(lhs[n+0][i][j][k2]*rhs[m][i][j][k]));
				}
			}
		}
		/*
		--------------------------------------------------------------------
		c         And again the last two rows separately
		c-------------------------------------------------------------------
		*/
		k=(grid_points[2]-2);
		k1=(grid_points[2]-1);
		#pragma loop name z_solve#2#1 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#2#1#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				fac1=(1.0/lhs[n+2][i][j][k]);
				lhs[n+3][i][j][k]=(fac1*lhs[n+3][i][j][k]);
				lhs[n+4][i][j][k]=(fac1*lhs[n+4][i][j][k]);
				rhs[m][i][j][k]=(fac1*rhs[m][i][j][k]);
				lhs[n+2][i][j][k1]=(lhs[n+2][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+3][i][j][k]));
				lhs[n+3][i][j][k1]=(lhs[n+3][i][j][k1]-(lhs[n+1][i][j][k1]*lhs[n+4][i][j][k]));
				rhs[m][i][j][k1]=(rhs[m][i][j][k1]-(lhs[n+1][i][j][k1]*rhs[m][i][j][k]));
				/*
				--------------------------------------------------------------------
				c               Scale the last row immediately (some of this is overkill
				c               if this is the last cell)
				c-------------------------------------------------------------------
				*/
				fac2=(1.0/lhs[n+2][i][j][k1]);
				rhs[m][i][j][k1]=(fac2*rhs[m][i][j][k1]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c                         BACKSUBSTITUTION 
	c-------------------------------------------------------------------
	*/
	k=(grid_points[2]-2);
	k1=(grid_points[2]-1);
	n=0;
	#pragma loop name z_solve#3 
	for (m=0; m<3; m ++ )
	{
		#pragma loop name z_solve#3#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#3#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j][k1]));
			}
		}
	}
	#pragma loop name z_solve#4 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name z_solve#4#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#4#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				rhs[m][i][j][k]=(rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j][k1]));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      Whether or not this is the last processor, we always have
	c      to complete the back-substitution 
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c      The first three factors
	c-------------------------------------------------------------------
	*/
	n=0;
	#pragma loop name z_solve#5 
	for (m=0; m<3; m ++ )
	{
		#pragma loop name z_solve#5#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#5#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name z_solve#5#0#0#0 
				for (k=(grid_points[2]-3); k>=0; k -- )
				{
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j][k1]))-(lhs[n+4][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c      And the remaining two
	c-------------------------------------------------------------------
	*/
	#pragma loop name z_solve#6 
	for (m=3; m<5; m ++ )
	{
		n=(((m-3)+1)*5);
		#pragma loop name z_solve#6#0 
		for (i=1; i<=(grid_points[0]-2); i ++ )
		{
			#pragma loop name z_solve#6#0#0 
			for (j=1; j<=(grid_points[1]-2); j ++ )
			{
				#pragma loop name z_solve#6#0#0#0 
				for (k=(grid_points[2]-3); k>=0; k -- )
				{
					k1=(k+1);
					k2=(k+2);
					rhs[m][i][j][k]=((rhs[m][i][j][k]-(lhs[n+3][i][j][k]*rhs[m][i][j][k1]))-(lhs[n+4][i][j][k]*rhs[m][i][j][k2]));
				}
			}
		}
	}
	tzetar();
	return ;
}
