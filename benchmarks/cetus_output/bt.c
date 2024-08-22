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
  
  NAS Parallel Benchmarks 2.3 OpenMP C versions - BT

  This benchmark is an OpenMP C version of the NPB BT code.
  
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

  Authors: R. Van der Wijngaart
           T. Harris
           M. Yarrow

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
static void compute_rhs(void );
static void set_constants(void );
static void verify(int no_time_steps, char * class, boolean * verified);
static void x_solve(void );
static void x_backsubstitute(void );
static void x_solve_cell(void );
static void matvec_sub(double ablock[5][5], double avec[5], double bvec[5]);
static void matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5]);
static void binvcrhs(double lhs[5][5], double c[5][5], double r[5]);
static void binvrhs(double lhs[5][5], double r[5]);
static void y_solve(void );
static void y_backsubstitute(void );
static void y_solve_cell(void );
static void z_solve(void );
static void z_backsubstitute(void );
static void z_solve_cell(void );
/*
--------------------------------------------------------------------
      program BT
c-------------------------------------------------------------------
*/
int main(int argc, char * * argv)
{
	int niter, step, n3;
	int nthreads = 1;
	double navg, mflops;
	double tmax;
	boolean verified;
	char class;
	FILE * fp;
	/*
	--------------------------------------------------------------------
	c      Root node reads input file (if it exists) else takes
	c      defaults from parameters
	c-------------------------------------------------------------------
	*/
	int _ret_val_0;
	printf("\n\n NAS Parallel Benchmarks 2.3 OpenMP C version"" - BT Benchmark\n\n");
	fp=fopen("inputbt.data", "r");
	if (fp!=((void * )0))
	{
		printf(" Reading from input file inputbt.data");
		fscanf(fp, "%d",  & niter);
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%lg",  & dt);
		while (fgetc(fp)!='\n')
		{
			;
		}
		fscanf(fp, "%d%d%d",  & grid_points[0],  & grid_points[1],  & grid_points[2]);
		fclose(fp);
	}
	else
	{
		printf(" No input file inputbt.data. Using compiled defaults\n");
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
		printf(" %dx%dx%d\n", grid_points[0], grid_points[1], grid_points[2]);
		printf(" Problem size too big for compiled array sizes\n");
		exit(1);
	}
	set_constants();
	{
		initialize();
		lhsinit();
		exact_rhs();
		/*
		--------------------------------------------------------------------
		c      do one time step to touch all code, and reinitialize
		c-------------------------------------------------------------------
		*/
		adi();
		initialize();
	}
	/* end parallel */
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
	n3=((grid_points[0]*grid_points[1])*grid_points[2]);
	navg=(((grid_points[0]+grid_points[1])+grid_points[2])/3.0);
	if (tmax!=0.0)
	{
		mflops=(((1.0E-6*((double)niter))*(((3478.8*((double)n3))-(17655.7*(navg*navg)))+(28023.7*navg)))/tmax);
	}
	else
	{
		mflops=0.0;
	}
	c_print_results("BT", class, grid_points[0], grid_points[1], grid_points[2], niter, nthreads, tmax, mflops, "          floating point", verified, "2.3", "24 Jun 2024", "gcc", "gcc", "(none)", "-I../common", "-O3 ", "(none)", "(none)");
	return _ret_val_0;
}

/*
--------------------------------------------------------------------
c-------------------------------------------------------------------
*/
static void add(void )
{
	/*
	--------------------------------------------------------------------
	c     addition of update to the vector u
	c-------------------------------------------------------------------
	*/
	int i, j, k, m;
	#pragma loop name add#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name add#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name add#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name add#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					u[i][j][k][m]=(u[i][j][k][m]+rhs[i][j][k][m]);
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
	compute_rhs();
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
	c     this function computes the norm of the difference between the
	c     computed solution and the exact solution
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
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name error_norm#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name error_norm#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, u_exact);
				#pragma loop name error_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=(u[i][j][k][m]-u_exact[m]);
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
	#pragma loop name error_norm#2 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name error_norm#2#0 
		for (d=0; d<=2; d ++ )
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
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name rhs_norm#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name rhs_norm#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name rhs_norm#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					add=rhs[i][j][k][m];
					rms[m]=(rms[m]+(add*add));
				}
			}
		}
	}
	#pragma loop name rhs_norm#2 
	for (m=0; m<5; m ++ )
	{
		#pragma loop name rhs_norm#2#0 
		for (d=0; d<=2; d ++ )
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
	c     compute the right hand side based on exact solution
	c-------------------------------------------------------------------
	*/
	double dtemp[5], xi, eta, zeta, dtpp;
	int m, i, j, k, ip1, im1, jp1, jm1, km1, kp1;
	/*
	--------------------------------------------------------------------
	c     initialize                                  
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#0 
	for (i=0; i<grid_points[0]; i ++ )
	{
		#pragma loop name exact_rhs#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name exact_rhs#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				#pragma loop name exact_rhs#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<5; m ++ )
				{
					forcing[i][j][k][m]=0.0;
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     xi-direction flux differences                      
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#1 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		eta=(((double)j)*dnym1);
		#pragma loop name exact_rhs#1#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			zeta=(((double)k)*dnzm1);
			#pragma loop name exact_rhs#1#0#0 
			for (i=0; i<grid_points[0]; i ++ )
			{
				xi=(((double)i)*dnxm1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[i][m]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#1#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[i][m]=(dtpp*dtemp[m]);
				}
				cuf[i]=(buf[i][1]*buf[i][1]);
				buf[i][0]=((cuf[i]+(buf[i][2]*buf[i][2]))+(buf[i][3]*buf[i][3]));
				q[i]=(0.5*(((buf[i][1]*ue[i][1])+(buf[i][2]*ue[i][2]))+(buf[i][3]*ue[i][3])));
			}
			#pragma loop name exact_rhs#1#0#1 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				im1=(i-1);
				ip1=(i+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(tx2*(ue[ip1][1]-ue[im1][1])))+(dx1tx1*((ue[ip1][0]-(2.0*ue[i][0]))+ue[im1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(tx2*(((ue[ip1][1]*buf[ip1][1])+(c2*(ue[ip1][4]-q[ip1])))-((ue[im1][1]*buf[im1][1])+(c2*(ue[im1][4]-q[im1]))))))+(xxcon1*((buf[ip1][1]-(2.0*buf[i][1]))+buf[im1][1])))+(dx2tx1*((ue[ip1][1]-(2.0*ue[i][1]))+ue[im1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(tx2*((ue[ip1][2]*buf[ip1][1])-(ue[im1][2]*buf[im1][1]))))+(xxcon2*((buf[ip1][2]-(2.0*buf[i][2]))+buf[im1][2])))+(dx3tx1*((ue[ip1][2]-(2.0*ue[i][2]))+ue[im1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(tx2*((ue[ip1][3]*buf[ip1][1])-(ue[im1][3]*buf[im1][1]))))+(xxcon2*((buf[ip1][3]-(2.0*buf[i][3]))+buf[im1][3])))+(dx4tx1*((ue[ip1][3]-(2.0*ue[i][3]))+ue[im1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(tx2*((buf[ip1][1]*((c1*ue[ip1][4])-(c2*q[ip1])))-(buf[im1][1]*((c1*ue[im1][4])-(c2*q[im1]))))))+((0.5*xxcon3)*((buf[ip1][0]-(2.0*buf[i][0]))+buf[im1][0])))+(xxcon4*((cuf[ip1]-(2.0*cuf[i]))+cuf[im1])))+(xxcon5*((buf[ip1][4]-(2.0*buf[i][4]))+buf[im1][4])))+(dx5tx1*((ue[ip1][4]-(2.0*ue[i][4]))+ue[im1][4])));
			}
			/*
			--------------------------------------------------------------------
			c     Fourth-order dissipation                         
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#1#0#2 
			for (m=0; m<5; m ++ )
			{
				i=1;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((5.0*ue[i][m])-(4.0*ue[i+1][m]))+ue[i+2][m])));
				i=2;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((( - 4.0)*ue[i-1][m])+(6.0*ue[i][m]))-(4.0*ue[i+1][m]))+ue[i+2][m])));
			}
			#pragma loop name exact_rhs#1#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#1#0#3#0 
				for (i=(1*3); i<=((grid_points[0]-(3*1))-1); i ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((ue[i-2][m]-(4.0*ue[i-1][m]))+(6.0*ue[i][m]))-(4.0*ue[i+1][m]))+ue[i+2][m])));
				}
			}
			#pragma loop name exact_rhs#1#0#4 
			for (m=0; m<5; m ++ )
			{
				i=(grid_points[0]-3);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((ue[i-2][m]-(4.0*ue[i-1][m]))+(6.0*ue[i][m]))-(4.0*ue[i+1][m]))));
				i=(grid_points[0]-2);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((ue[i-2][m]-(4.0*ue[i-1][m]))+(5.0*ue[i][m]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     eta-direction flux differences             
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#2 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name exact_rhs#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			zeta=(((double)k)*dnzm1);
			#pragma loop name exact_rhs#2#0#0 
			for (j=0; j<grid_points[1]; j ++ )
			{
				eta=(((double)j)*dnym1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#2#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[j][m]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#2#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[j][m]=(dtpp*dtemp[m]);
				}
				cuf[j]=(buf[j][2]*buf[j][2]);
				buf[j][0]=((cuf[j]+(buf[j][1]*buf[j][1]))+(buf[j][3]*buf[j][3]));
				q[j]=(0.5*(((buf[j][1]*ue[j][1])+(buf[j][2]*ue[j][2]))+(buf[j][3]*ue[j][3])));
			}
			#pragma loop name exact_rhs#2#0#1 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
				jm1=(j-1);
				jp1=(j+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(ty2*(ue[jp1][2]-ue[jm1][2])))+(dy1ty1*((ue[jp1][0]-(2.0*ue[j][0]))+ue[jm1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(ty2*((ue[jp1][1]*buf[jp1][2])-(ue[jm1][1]*buf[jm1][2]))))+(yycon2*((buf[jp1][1]-(2.0*buf[j][1]))+buf[jm1][1])))+(dy2ty1*((ue[jp1][1]-(2.0*ue[j][1]))+ue[jm1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(ty2*(((ue[jp1][2]*buf[jp1][2])+(c2*(ue[jp1][4]-q[jp1])))-((ue[jm1][2]*buf[jm1][2])+(c2*(ue[jm1][4]-q[jm1]))))))+(yycon1*((buf[jp1][2]-(2.0*buf[j][2]))+buf[jm1][2])))+(dy3ty1*((ue[jp1][2]-(2.0*ue[j][2]))+ue[jm1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(ty2*((ue[jp1][3]*buf[jp1][2])-(ue[jm1][3]*buf[jm1][2]))))+(yycon2*((buf[jp1][3]-(2.0*buf[j][3]))+buf[jm1][3])))+(dy4ty1*((ue[jp1][3]-(2.0*ue[j][3]))+ue[jm1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(ty2*((buf[jp1][2]*((c1*ue[jp1][4])-(c2*q[jp1])))-(buf[jm1][2]*((c1*ue[jm1][4])-(c2*q[jm1]))))))+((0.5*yycon3)*((buf[jp1][0]-(2.0*buf[j][0]))+buf[jm1][0])))+(yycon4*((cuf[jp1]-(2.0*cuf[j]))+cuf[jm1])))+(yycon5*((buf[jp1][4]-(2.0*buf[j][4]))+buf[jm1][4])))+(dy5ty1*((ue[jp1][4]-(2.0*ue[j][4]))+ue[jm1][4])));
			}
			/*
			--------------------------------------------------------------------
			c     Fourth-order dissipation                      
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#2#0#2 
			for (m=0; m<5; m ++ )
			{
				j=1;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((5.0*ue[j][m])-(4.0*ue[j+1][m]))+ue[j+2][m])));
				j=2;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((( - 4.0)*ue[j-1][m])+(6.0*ue[j][m]))-(4.0*ue[j+1][m]))+ue[j+2][m])));
			}
			#pragma loop name exact_rhs#2#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#2#0#3#0 
				for (j=(1*3); j<=((grid_points[1]-(3*1))-1); j ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((ue[j-2][m]-(4.0*ue[j-1][m]))+(6.0*ue[j][m]))-(4.0*ue[j+1][m]))+ue[j+2][m])));
				}
			}
			#pragma loop name exact_rhs#2#0#4 
			for (m=0; m<5; m ++ )
			{
				j=(grid_points[1]-3);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((ue[j-2][m]-(4.0*ue[j-1][m]))+(6.0*ue[j][m]))-(4.0*ue[j+1][m]))));
				j=(grid_points[1]-2);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((ue[j-2][m]-(4.0*ue[j-1][m]))+(5.0*ue[j][m]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     zeta-direction flux differences                      
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#3 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name exact_rhs#3#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name exact_rhs#3#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				zeta=(((double)k)*dnzm1);
				exact_solution(xi, eta, zeta, dtemp);
				#pragma loop name exact_rhs#3#0#0#0 
				for (m=0; m<5; m ++ )
				{
					ue[k][m]=dtemp[m];
				}
				dtpp=(1.0/dtemp[0]);
				#pragma loop name exact_rhs#3#0#0#1 
				for (m=1; m<=4; m ++ )
				{
					buf[k][m]=(dtpp*dtemp[m]);
				}
				cuf[k]=(buf[k][3]*buf[k][3]);
				buf[k][0]=((cuf[k]+(buf[k][1]*buf[k][1]))+(buf[k][2]*buf[k][2]));
				q[k]=(0.5*(((buf[k][1]*ue[k][1])+(buf[k][2]*ue[k][2]))+(buf[k][3]*ue[k][3])));
			}
			#pragma loop name exact_rhs#3#0#1 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				km1=(k-1);
				kp1=(k+1);
				forcing[i][j][k][0]=((forcing[i][j][k][0]-(tz2*(ue[kp1][3]-ue[km1][3])))+(dz1tz1*((ue[kp1][0]-(2.0*ue[k][0]))+ue[km1][0])));
				forcing[i][j][k][1]=(((forcing[i][j][k][1]-(tz2*((ue[kp1][1]*buf[kp1][3])-(ue[km1][1]*buf[km1][3]))))+(zzcon2*((buf[kp1][1]-(2.0*buf[k][1]))+buf[km1][1])))+(dz2tz1*((ue[kp1][1]-(2.0*ue[k][1]))+ue[km1][1])));
				forcing[i][j][k][2]=(((forcing[i][j][k][2]-(tz2*((ue[kp1][2]*buf[kp1][3])-(ue[km1][2]*buf[km1][3]))))+(zzcon2*((buf[kp1][2]-(2.0*buf[k][2]))+buf[km1][2])))+(dz3tz1*((ue[kp1][2]-(2.0*ue[k][2]))+ue[km1][2])));
				forcing[i][j][k][3]=(((forcing[i][j][k][3]-(tz2*(((ue[kp1][3]*buf[kp1][3])+(c2*(ue[kp1][4]-q[kp1])))-((ue[km1][3]*buf[km1][3])+(c2*(ue[km1][4]-q[km1]))))))+(zzcon1*((buf[kp1][3]-(2.0*buf[k][3]))+buf[km1][3])))+(dz4tz1*((ue[kp1][3]-(2.0*ue[k][3]))+ue[km1][3])));
				forcing[i][j][k][4]=(((((forcing[i][j][k][4]-(tz2*((buf[kp1][3]*((c1*ue[kp1][4])-(c2*q[kp1])))-(buf[km1][3]*((c1*ue[km1][4])-(c2*q[km1]))))))+((0.5*zzcon3)*((buf[kp1][0]-(2.0*buf[k][0]))+buf[km1][0])))+(zzcon4*((cuf[kp1]-(2.0*cuf[k]))+cuf[km1])))+(zzcon5*((buf[kp1][4]-(2.0*buf[k][4]))+buf[km1][4])))+(dz5tz1*((ue[kp1][4]-(2.0*ue[k][4]))+ue[km1][4])));
			}
			/*
			--------------------------------------------------------------------
			c     Fourth-order dissipation                        
			c-------------------------------------------------------------------
			*/
			#pragma loop name exact_rhs#3#0#2 
			for (m=0; m<5; m ++ )
			{
				k=1;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((5.0*ue[k][m])-(4.0*ue[k+1][m]))+ue[k+2][m])));
				k=2;
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((( - 4.0)*ue[k-1][m])+(6.0*ue[k][m]))-(4.0*ue[k+1][m]))+ue[k+2][m])));
			}
			#pragma loop name exact_rhs#3#0#3 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name exact_rhs#3#0#3#0 
				for (k=(1*3); k<=((grid_points[2]-(3*1))-1); k ++ )
				{
					forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((((ue[k-2][m]-(4.0*ue[k-1][m]))+(6.0*ue[k][m]))-(4.0*ue[k+1][m]))+ue[k+2][m])));
				}
			}
			#pragma loop name exact_rhs#3#0#4 
			for (m=0; m<5; m ++ )
			{
				k=(grid_points[2]-3);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*(((ue[k-2][m]-(4.0*ue[k-1][m]))+(6.0*ue[k][m]))-(4.0*ue[k+1][m]))));
				k=(grid_points[2]-2);
				forcing[i][j][k][m]=(forcing[i][j][k][m]-(dssp*((ue[k-2][m]-(4.0*ue[k-1][m]))+(5.0*ue[k][m]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     now change the sign of the forcing function, 
	c-------------------------------------------------------------------
	*/
	#pragma loop name exact_rhs#4 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name exact_rhs#4#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name exact_rhs#4#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name exact_rhs#4#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<5; m ++ )
				{
					forcing[i][j][k][m]=(( - 1.0)*forcing[i][j][k][m]);
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
	c     this function returns the exact solution at point xi, eta, zeta  
	c-------------------------------------------------------------------
	*/
	int m;
	#pragma loop name exact_solution#0 
	for (m=0; m<5; m ++ )
	{
		dtemp[m]=(((ce[m][0]+(xi*(ce[m][1]+(xi*(ce[m][4]+(xi*(ce[m][7]+(xi*ce[m][10]))))))))+(eta*(ce[m][2]+(eta*(ce[m][5]+(eta*(ce[m][8]+(eta*ce[m][11]))))))))+(zeta*(ce[m][3]+(zeta*(ce[m][6]+(zeta*(ce[m][9]+(zeta*ce[m][12]))))))));
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
	c     This subroutine initializes the field variable u using 
	c     tri-linear transfinite interpolation of the boundary values     
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
	for (i=0; i<64; i ++ )
	{
		#pragma loop name initialize#0#0 
		for (j=0; j<64; j ++ )
		{
			#pragma loop name initialize#0#0#0 
			for (k=0; k<64; k ++ )
			{
				#pragma loop name initialize#0#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<5; m ++ )
				{
					u[i][j][k][m]=1.0;
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     first store the "interpolated" values everywhere on the grid    
	c-------------------------------------------------------------------
	*/
	#pragma loop name initialize#1 
	for (i=0; i<grid_points[0]; i ++ )
	{
		xi=(((double)i)*dnxm1);
		#pragma loop name initialize#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			eta=(((double)j)*dnym1);
			#pragma loop name initialize#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
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
					u[i][j][k][m]=((((((Pxi+Peta)+Pzeta)-(Pxi*Peta))-(Pxi*Pzeta))-(Peta*Pzeta))+((Pxi*Peta)*Pzeta));
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     now store the exact values on the boundaries        
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     west face                                                  
	c-------------------------------------------------------------------
	*/
	i=0;
	xi=0.0;
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
				u[i][j][k][m]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     east face                                                      
	c-------------------------------------------------------------------
	*/
	i=(grid_points[0]-1);
	xi=1.0;
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
				u[i][j][k][m]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     south face                                                 
	c-------------------------------------------------------------------
	*/
	j=0;
	eta=0.0;
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
				u[i][j][k][m]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     north face                                    
	c-------------------------------------------------------------------
	*/
	j=(grid_points[1]-1);
	eta=1.0;
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
				u[i][j][k][m]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     bottom face                                       
	c-------------------------------------------------------------------
	*/
	k=0;
	zeta=0.0;
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
				u[i][j][k][m]=temp[m];
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     top face     
	c-------------------------------------------------------------------
	*/
	k=(grid_points[2]-1);
	zeta=1.0;
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
				u[i][j][k][m]=temp[m];
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
	int i, j, k, m, n;
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     zero the whole left hand side for starters
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsinit#0 
	for (i=0; i<grid_points[0]; i ++ )
	{
		#pragma loop name lhsinit#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name lhsinit#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				#pragma loop name lhsinit#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					#pragma loop name lhsinit#0#0#0#0#0 
					#pragma cetus parallel 
					#pragma omp parallel for
					for (n=0; n<5; n ++ )
					{
						lhs[i][j][k][0][m][n]=0.0;
						lhs[i][j][k][1][m][n]=0.0;
						lhs[i][j][k][2][m][n]=0.0;
					}
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     next, set all diagonal values to 1. This is overkill, but convenient
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsinit#1 
	for (i=0; i<grid_points[0]; i ++ )
	{
		#pragma loop name lhsinit#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name lhsinit#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				#pragma loop name lhsinit#1#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (m=0; m<5; m ++ )
				{
					lhs[i][j][k][1][m][m]=1.0;
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
	c     This function computes the left hand side in the xi-direction
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c     determine a (labeled f) and n jacobians
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsx#0 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name lhsx#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name lhsx#0#0#0 
			for (i=0; i<grid_points[0]; i ++ )
			{
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				/*
				--------------------------------------------------------------------
				c     
				c-------------------------------------------------------------------
				*/
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=1.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - ((u[i][j][k][1]*tmp2)*u[i][j][k][1]))+(((c2*0.5)*(((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*tmp2));
				fjac[i][j][k][1][1]=((2.0-c2)*(u[i][j][k][1]/u[i][j][k][0]));
				fjac[i][j][k][1][2]=(( - c2)*(u[i][j][k][2]*tmp1));
				fjac[i][j][k][1][3]=(( - c2)*(u[i][j][k][3]*tmp1));
				fjac[i][j][k][1][4]=c2;
				fjac[i][j][k][2][0]=(( - (u[i][j][k][1]*u[i][j][k][2]))*tmp2);
				fjac[i][j][k][2][1]=(u[i][j][k][2]*tmp1);
				fjac[i][j][k][2][2]=(u[i][j][k][1]*tmp1);
				fjac[i][j][k][2][3]=0.0;
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - (u[i][j][k][1]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][3][1]=(u[i][j][k][3]*tmp1);
				fjac[i][j][k][3][2]=0.0;
				fjac[i][j][k][3][3]=(u[i][j][k][1]*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=((((c2*(((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*tmp2)-(c1*(u[i][j][k][4]*tmp1)))*(u[i][j][k][1]*tmp1));
				fjac[i][j][k][4][1]=(((c1*u[i][j][k][4])*tmp1)-(((0.5*c2)*((((3.0*u[i][j][k][1])*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*tmp2));
				fjac[i][j][k][4][2]=((( - c2)*(u[i][j][k][2]*u[i][j][k][1]))*tmp2);
				fjac[i][j][k][4][3]=((( - c2)*(u[i][j][k][3]*u[i][j][k][1]))*tmp2);
				fjac[i][j][k][4][4]=(c1*(u[i][j][k][1]*tmp1));
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=(((( - con43)*c3c4)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][1][1]=((con43*c3c4)*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - c3c4)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - c3c4)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - ((con43*c3c4)-c1345))*tmp3)*(u[i][j][k][1]*u[i][j][k][1]))-(((c3c4-c1345)*tmp3)*(u[i][j][k][2]*u[i][j][k][2])))-(((c3c4-c1345)*tmp3)*(u[i][j][k][3]*u[i][j][k][3])))-((c1345*tmp2)*u[i][j][k][4]));
				njac[i][j][k][4][1]=((((con43*c3c4)-c1345)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][4][2]=(((c3c4-c1345)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][4][3]=(((c3c4-c1345)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][4][4]=(c1345*tmp1);
			}
			/*
			--------------------------------------------------------------------
			c     now jacobians set, so form left hand side in x direction
			c-------------------------------------------------------------------
			*/
			#pragma loop name lhsx#0#0#1 
			for (i=1; i<(grid_points[0]-1); i ++ )
			{
				tmp1=(dt*tx1);
				tmp2=(dt*tx2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i-1][j][k][0][0])-(tmp1*njac[i-1][j][k][0][0]))-(tmp1*dx1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i-1][j][k][0][1])-(tmp1*njac[i-1][j][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i-1][j][k][0][2])-(tmp1*njac[i-1][j][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i-1][j][k][0][3])-(tmp1*njac[i-1][j][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i-1][j][k][0][4])-(tmp1*njac[i-1][j][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i-1][j][k][1][0])-(tmp1*njac[i-1][j][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i-1][j][k][1][1])-(tmp1*njac[i-1][j][k][1][1]))-(tmp1*dx2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i-1][j][k][1][2])-(tmp1*njac[i-1][j][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i-1][j][k][1][3])-(tmp1*njac[i-1][j][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i-1][j][k][1][4])-(tmp1*njac[i-1][j][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i-1][j][k][2][0])-(tmp1*njac[i-1][j][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i-1][j][k][2][1])-(tmp1*njac[i-1][j][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i-1][j][k][2][2])-(tmp1*njac[i-1][j][k][2][2]))-(tmp1*dx3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i-1][j][k][2][3])-(tmp1*njac[i-1][j][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i-1][j][k][2][4])-(tmp1*njac[i-1][j][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i-1][j][k][3][0])-(tmp1*njac[i-1][j][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i-1][j][k][3][1])-(tmp1*njac[i-1][j][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i-1][j][k][3][2])-(tmp1*njac[i-1][j][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i-1][j][k][3][3])-(tmp1*njac[i-1][j][k][3][3]))-(tmp1*dx4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i-1][j][k][3][4])-(tmp1*njac[i-1][j][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i-1][j][k][4][0])-(tmp1*njac[i-1][j][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i-1][j][k][4][1])-(tmp1*njac[i-1][j][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i-1][j][k][4][2])-(tmp1*njac[i-1][j][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i-1][j][k][4][3])-(tmp1*njac[i-1][j][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i-1][j][k][4][4])-(tmp1*njac[i-1][j][k][4][4]))-(tmp1*dx5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*dx1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*dx2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*dx3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*dx4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*dx5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i+1][j][k][0][0])-(tmp1*njac[i+1][j][k][0][0]))-(tmp1*dx1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i+1][j][k][0][1])-(tmp1*njac[i+1][j][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i+1][j][k][0][2])-(tmp1*njac[i+1][j][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i+1][j][k][0][3])-(tmp1*njac[i+1][j][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i+1][j][k][0][4])-(tmp1*njac[i+1][j][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i+1][j][k][1][0])-(tmp1*njac[i+1][j][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i+1][j][k][1][1])-(tmp1*njac[i+1][j][k][1][1]))-(tmp1*dx2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i+1][j][k][1][2])-(tmp1*njac[i+1][j][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i+1][j][k][1][3])-(tmp1*njac[i+1][j][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i+1][j][k][1][4])-(tmp1*njac[i+1][j][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i+1][j][k][2][0])-(tmp1*njac[i+1][j][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i+1][j][k][2][1])-(tmp1*njac[i+1][j][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i+1][j][k][2][2])-(tmp1*njac[i+1][j][k][2][2]))-(tmp1*dx3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i+1][j][k][2][3])-(tmp1*njac[i+1][j][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i+1][j][k][2][4])-(tmp1*njac[i+1][j][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i+1][j][k][3][0])-(tmp1*njac[i+1][j][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i+1][j][k][3][1])-(tmp1*njac[i+1][j][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i+1][j][k][3][2])-(tmp1*njac[i+1][j][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i+1][j][k][3][3])-(tmp1*njac[i+1][j][k][3][3]))-(tmp1*dx4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i+1][j][k][3][4])-(tmp1*njac[i+1][j][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i+1][j][k][4][0])-(tmp1*njac[i+1][j][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i+1][j][k][4][1])-(tmp1*njac[i+1][j][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i+1][j][k][4][2])-(tmp1*njac[i+1][j][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i+1][j][k][4][3])-(tmp1*njac[i+1][j][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i+1][j][k][4][4])-(tmp1*njac[i+1][j][k][4][4]))-(tmp1*dx5));
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
	c     This function computes the left hand side for the three y-factors   
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c     Compute the indices for storing the tri-diagonal matrix;
	c     determine a (labeled f) and n jacobians for cell c
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsy#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name lhsy#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name lhsy#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=1.0;
				fjac[i][j][k][0][3]=0.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u[i][j][k][1]*u[i][j][k][2]))*tmp2);
				fjac[i][j][k][1][1]=(u[i][j][k][2]*tmp1);
				fjac[i][j][k][1][2]=(u[i][j][k][1]*tmp1);
				fjac[i][j][k][1][3]=0.0;
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - ((u[i][j][k][2]*u[i][j][k][2])*tmp2))+((0.5*c2)*((((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3]))*tmp2)));
				fjac[i][j][k][2][1]=((( - c2)*u[i][j][k][1])*tmp1);
				fjac[i][j][k][2][2]=(((2.0-c2)*u[i][j][k][2])*tmp1);
				fjac[i][j][k][2][3]=((( - c2)*u[i][j][k][3])*tmp1);
				fjac[i][j][k][2][4]=c2;
				fjac[i][j][k][3][0]=(( - (u[i][j][k][2]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][3][1]=0.0;
				fjac[i][j][k][3][2]=(u[i][j][k][3]*tmp1);
				fjac[i][j][k][3][3]=(u[i][j][k][2]*tmp1);
				fjac[i][j][k][3][4]=0.0;
				fjac[i][j][k][4][0]=(((((c2*(((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*tmp2)-((c1*u[i][j][k][4])*tmp1))*u[i][j][k][2])*tmp1);
				fjac[i][j][k][4][1]=(((( - c2)*u[i][j][k][1])*u[i][j][k][2])*tmp2);
				fjac[i][j][k][4][2]=(((c1*u[i][j][k][4])*tmp1)-((0.5*c2)*((((u[i][j][k][1]*u[i][j][k][1])+((3.0*u[i][j][k][2])*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3]))*tmp2)));
				fjac[i][j][k][4][3]=((( - c2)*(u[i][j][k][2]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][4][4]=((c1*u[i][j][k][2])*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - c3c4)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][1][1]=(c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=(((( - con43)*c3c4)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=((con43*c3c4)*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=((( - c3c4)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(c3c4*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (c3c4-c1345))*tmp3)*(u[i][j][k][1]*u[i][j][k][1]))-((((con43*c3c4)-c1345)*tmp3)*(u[i][j][k][2]*u[i][j][k][2])))-(((c3c4-c1345)*tmp3)*(u[i][j][k][3]*u[i][j][k][3])))-((c1345*tmp2)*u[i][j][k][4]));
				njac[i][j][k][4][1]=(((c3c4-c1345)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][4][2]=((((con43*c3c4)-c1345)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][4][3]=(((c3c4-c1345)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][4][4]=(c1345*tmp1);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     now joacobians set, so form left hand side in y direction
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsy#1 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name lhsy#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name lhsy#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(dt*ty1);
				tmp2=(dt*ty2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][j-1][k][0][0])-(tmp1*njac[i][j-1][k][0][0]))-(tmp1*dy1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][j-1][k][0][1])-(tmp1*njac[i][j-1][k][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][j-1][k][0][2])-(tmp1*njac[i][j-1][k][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][j-1][k][0][3])-(tmp1*njac[i][j-1][k][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][j-1][k][0][4])-(tmp1*njac[i][j-1][k][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][j-1][k][1][0])-(tmp1*njac[i][j-1][k][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][j-1][k][1][1])-(tmp1*njac[i][j-1][k][1][1]))-(tmp1*dy2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][j-1][k][1][2])-(tmp1*njac[i][j-1][k][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][j-1][k][1][3])-(tmp1*njac[i][j-1][k][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][j-1][k][1][4])-(tmp1*njac[i][j-1][k][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][j-1][k][2][0])-(tmp1*njac[i][j-1][k][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][j-1][k][2][1])-(tmp1*njac[i][j-1][k][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][j-1][k][2][2])-(tmp1*njac[i][j-1][k][2][2]))-(tmp1*dy3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][j-1][k][2][3])-(tmp1*njac[i][j-1][k][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][j-1][k][2][4])-(tmp1*njac[i][j-1][k][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][j-1][k][3][0])-(tmp1*njac[i][j-1][k][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][j-1][k][3][1])-(tmp1*njac[i][j-1][k][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][j-1][k][3][2])-(tmp1*njac[i][j-1][k][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][j-1][k][3][3])-(tmp1*njac[i][j-1][k][3][3]))-(tmp1*dy4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][j-1][k][3][4])-(tmp1*njac[i][j-1][k][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][j-1][k][4][0])-(tmp1*njac[i][j-1][k][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][j-1][k][4][1])-(tmp1*njac[i][j-1][k][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][j-1][k][4][2])-(tmp1*njac[i][j-1][k][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][j-1][k][4][3])-(tmp1*njac[i][j-1][k][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][j-1][k][4][4])-(tmp1*njac[i][j-1][k][4][4]))-(tmp1*dy5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*dy1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*dy2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*dy3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*dy4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*dy5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][j+1][k][0][0])-(tmp1*njac[i][j+1][k][0][0]))-(tmp1*dy1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][j+1][k][0][1])-(tmp1*njac[i][j+1][k][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][j+1][k][0][2])-(tmp1*njac[i][j+1][k][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][j+1][k][0][3])-(tmp1*njac[i][j+1][k][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][j+1][k][0][4])-(tmp1*njac[i][j+1][k][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][j+1][k][1][0])-(tmp1*njac[i][j+1][k][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][j+1][k][1][1])-(tmp1*njac[i][j+1][k][1][1]))-(tmp1*dy2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][j+1][k][1][2])-(tmp1*njac[i][j+1][k][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][j+1][k][1][3])-(tmp1*njac[i][j+1][k][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][j+1][k][1][4])-(tmp1*njac[i][j+1][k][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][j+1][k][2][0])-(tmp1*njac[i][j+1][k][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][j+1][k][2][1])-(tmp1*njac[i][j+1][k][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][j+1][k][2][2])-(tmp1*njac[i][j+1][k][2][2]))-(tmp1*dy3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][j+1][k][2][3])-(tmp1*njac[i][j+1][k][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][j+1][k][2][4])-(tmp1*njac[i][j+1][k][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][j+1][k][3][0])-(tmp1*njac[i][j+1][k][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][j+1][k][3][1])-(tmp1*njac[i][j+1][k][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][j+1][k][3][2])-(tmp1*njac[i][j+1][k][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][j+1][k][3][3])-(tmp1*njac[i][j+1][k][3][3]))-(tmp1*dy4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][j+1][k][3][4])-(tmp1*njac[i][j+1][k][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][j+1][k][4][0])-(tmp1*njac[i][j+1][k][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][j+1][k][4][1])-(tmp1*njac[i][j+1][k][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][j+1][k][4][2])-(tmp1*njac[i][j+1][k][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][j+1][k][4][3])-(tmp1*njac[i][j+1][k][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][j+1][k][4][4])-(tmp1*njac[i][j+1][k][4][4]))-(tmp1*dy5));
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
	c     This function computes the left hand side for the three z-factors   
	c-------------------------------------------------------------------
	*/
	int i, j, k;
	/*
	--------------------------------------------------------------------
	c     Compute the indices for storing the block-diagonal matrix;
	c     determine c (labeled f) and s jacobians
	c---------------------------------------------------------------------
	*/
	#pragma loop name lhsz#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name lhsz#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name lhsz#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				tmp1=(1.0/u[i][j][k][0]);
				tmp2=(tmp1*tmp1);
				tmp3=(tmp1*tmp2);
				fjac[i][j][k][0][0]=0.0;
				fjac[i][j][k][0][1]=0.0;
				fjac[i][j][k][0][2]=0.0;
				fjac[i][j][k][0][3]=1.0;
				fjac[i][j][k][0][4]=0.0;
				fjac[i][j][k][1][0]=(( - (u[i][j][k][1]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][1][1]=(u[i][j][k][3]*tmp1);
				fjac[i][j][k][1][2]=0.0;
				fjac[i][j][k][1][3]=(u[i][j][k][1]*tmp1);
				fjac[i][j][k][1][4]=0.0;
				fjac[i][j][k][2][0]=(( - (u[i][j][k][2]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][2][1]=0.0;
				fjac[i][j][k][2][2]=(u[i][j][k][3]*tmp1);
				fjac[i][j][k][2][3]=(u[i][j][k][2]*tmp1);
				fjac[i][j][k][2][4]=0.0;
				fjac[i][j][k][3][0]=(( - ((u[i][j][k][3]*u[i][j][k][3])*tmp2))+((0.5*c2)*((((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3]))*tmp2)));
				fjac[i][j][k][3][1]=((( - c2)*u[i][j][k][1])*tmp1);
				fjac[i][j][k][3][2]=((( - c2)*u[i][j][k][2])*tmp1);
				fjac[i][j][k][3][3]=(((2.0-c2)*u[i][j][k][3])*tmp1);
				fjac[i][j][k][3][4]=c2;
				fjac[i][j][k][4][0]=((((c2*(((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*tmp2)-(c1*(u[i][j][k][4]*tmp1)))*(u[i][j][k][3]*tmp1));
				fjac[i][j][k][4][1]=((( - c2)*(u[i][j][k][1]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][4][2]=((( - c2)*(u[i][j][k][2]*u[i][j][k][3]))*tmp2);
				fjac[i][j][k][4][3]=((c1*(u[i][j][k][4]*tmp1))-((0.5*c2)*((((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+((3.0*u[i][j][k][3])*u[i][j][k][3]))*tmp2)));
				fjac[i][j][k][4][4]=((c1*u[i][j][k][3])*tmp1);
				njac[i][j][k][0][0]=0.0;
				njac[i][j][k][0][1]=0.0;
				njac[i][j][k][0][2]=0.0;
				njac[i][j][k][0][3]=0.0;
				njac[i][j][k][0][4]=0.0;
				njac[i][j][k][1][0]=((( - c3c4)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][1][1]=(c3c4*tmp1);
				njac[i][j][k][1][2]=0.0;
				njac[i][j][k][1][3]=0.0;
				njac[i][j][k][1][4]=0.0;
				njac[i][j][k][2][0]=((( - c3c4)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][2][1]=0.0;
				njac[i][j][k][2][2]=(c3c4*tmp1);
				njac[i][j][k][2][3]=0.0;
				njac[i][j][k][2][4]=0.0;
				njac[i][j][k][3][0]=(((( - con43)*c3c4)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][3][1]=0.0;
				njac[i][j][k][3][2]=0.0;
				njac[i][j][k][3][3]=(((con43*c3)*c4)*tmp1);
				njac[i][j][k][3][4]=0.0;
				njac[i][j][k][4][0]=(((((( - (c3c4-c1345))*tmp3)*(u[i][j][k][1]*u[i][j][k][1]))-(((c3c4-c1345)*tmp3)*(u[i][j][k][2]*u[i][j][k][2])))-((((con43*c3c4)-c1345)*tmp3)*(u[i][j][k][3]*u[i][j][k][3])))-((c1345*tmp2)*u[i][j][k][4]));
				njac[i][j][k][4][1]=(((c3c4-c1345)*tmp2)*u[i][j][k][1]);
				njac[i][j][k][4][2]=(((c3c4-c1345)*tmp2)*u[i][j][k][2]);
				njac[i][j][k][4][3]=((((con43*c3c4)-c1345)*tmp2)*u[i][j][k][3]);
				njac[i][j][k][4][4]=(c1345*tmp1);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     now jacobians set, so form left hand side in z direction
	c-------------------------------------------------------------------
	*/
	#pragma loop name lhsz#1 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name lhsz#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name lhsz#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				tmp1=(dt*tz1);
				tmp2=(dt*tz2);
				lhs[i][j][k][0][0][0]=(((( - tmp2)*fjac[i][j][k-1][0][0])-(tmp1*njac[i][j][k-1][0][0]))-(tmp1*dz1));
				lhs[i][j][k][0][0][1]=((( - tmp2)*fjac[i][j][k-1][0][1])-(tmp1*njac[i][j][k-1][0][1]));
				lhs[i][j][k][0][0][2]=((( - tmp2)*fjac[i][j][k-1][0][2])-(tmp1*njac[i][j][k-1][0][2]));
				lhs[i][j][k][0][0][3]=((( - tmp2)*fjac[i][j][k-1][0][3])-(tmp1*njac[i][j][k-1][0][3]));
				lhs[i][j][k][0][0][4]=((( - tmp2)*fjac[i][j][k-1][0][4])-(tmp1*njac[i][j][k-1][0][4]));
				lhs[i][j][k][0][1][0]=((( - tmp2)*fjac[i][j][k-1][1][0])-(tmp1*njac[i][j][k-1][1][0]));
				lhs[i][j][k][0][1][1]=(((( - tmp2)*fjac[i][j][k-1][1][1])-(tmp1*njac[i][j][k-1][1][1]))-(tmp1*dz2));
				lhs[i][j][k][0][1][2]=((( - tmp2)*fjac[i][j][k-1][1][2])-(tmp1*njac[i][j][k-1][1][2]));
				lhs[i][j][k][0][1][3]=((( - tmp2)*fjac[i][j][k-1][1][3])-(tmp1*njac[i][j][k-1][1][3]));
				lhs[i][j][k][0][1][4]=((( - tmp2)*fjac[i][j][k-1][1][4])-(tmp1*njac[i][j][k-1][1][4]));
				lhs[i][j][k][0][2][0]=((( - tmp2)*fjac[i][j][k-1][2][0])-(tmp1*njac[i][j][k-1][2][0]));
				lhs[i][j][k][0][2][1]=((( - tmp2)*fjac[i][j][k-1][2][1])-(tmp1*njac[i][j][k-1][2][1]));
				lhs[i][j][k][0][2][2]=(((( - tmp2)*fjac[i][j][k-1][2][2])-(tmp1*njac[i][j][k-1][2][2]))-(tmp1*dz3));
				lhs[i][j][k][0][2][3]=((( - tmp2)*fjac[i][j][k-1][2][3])-(tmp1*njac[i][j][k-1][2][3]));
				lhs[i][j][k][0][2][4]=((( - tmp2)*fjac[i][j][k-1][2][4])-(tmp1*njac[i][j][k-1][2][4]));
				lhs[i][j][k][0][3][0]=((( - tmp2)*fjac[i][j][k-1][3][0])-(tmp1*njac[i][j][k-1][3][0]));
				lhs[i][j][k][0][3][1]=((( - tmp2)*fjac[i][j][k-1][3][1])-(tmp1*njac[i][j][k-1][3][1]));
				lhs[i][j][k][0][3][2]=((( - tmp2)*fjac[i][j][k-1][3][2])-(tmp1*njac[i][j][k-1][3][2]));
				lhs[i][j][k][0][3][3]=(((( - tmp2)*fjac[i][j][k-1][3][3])-(tmp1*njac[i][j][k-1][3][3]))-(tmp1*dz4));
				lhs[i][j][k][0][3][4]=((( - tmp2)*fjac[i][j][k-1][3][4])-(tmp1*njac[i][j][k-1][3][4]));
				lhs[i][j][k][0][4][0]=((( - tmp2)*fjac[i][j][k-1][4][0])-(tmp1*njac[i][j][k-1][4][0]));
				lhs[i][j][k][0][4][1]=((( - tmp2)*fjac[i][j][k-1][4][1])-(tmp1*njac[i][j][k-1][4][1]));
				lhs[i][j][k][0][4][2]=((( - tmp2)*fjac[i][j][k-1][4][2])-(tmp1*njac[i][j][k-1][4][2]));
				lhs[i][j][k][0][4][3]=((( - tmp2)*fjac[i][j][k-1][4][3])-(tmp1*njac[i][j][k-1][4][3]));
				lhs[i][j][k][0][4][4]=(((( - tmp2)*fjac[i][j][k-1][4][4])-(tmp1*njac[i][j][k-1][4][4]))-(tmp1*dz5));
				lhs[i][j][k][1][0][0]=((1.0+((tmp1*2.0)*njac[i][j][k][0][0]))+((tmp1*2.0)*dz1));
				lhs[i][j][k][1][0][1]=((tmp1*2.0)*njac[i][j][k][0][1]);
				lhs[i][j][k][1][0][2]=((tmp1*2.0)*njac[i][j][k][0][2]);
				lhs[i][j][k][1][0][3]=((tmp1*2.0)*njac[i][j][k][0][3]);
				lhs[i][j][k][1][0][4]=((tmp1*2.0)*njac[i][j][k][0][4]);
				lhs[i][j][k][1][1][0]=((tmp1*2.0)*njac[i][j][k][1][0]);
				lhs[i][j][k][1][1][1]=((1.0+((tmp1*2.0)*njac[i][j][k][1][1]))+((tmp1*2.0)*dz2));
				lhs[i][j][k][1][1][2]=((tmp1*2.0)*njac[i][j][k][1][2]);
				lhs[i][j][k][1][1][3]=((tmp1*2.0)*njac[i][j][k][1][3]);
				lhs[i][j][k][1][1][4]=((tmp1*2.0)*njac[i][j][k][1][4]);
				lhs[i][j][k][1][2][0]=((tmp1*2.0)*njac[i][j][k][2][0]);
				lhs[i][j][k][1][2][1]=((tmp1*2.0)*njac[i][j][k][2][1]);
				lhs[i][j][k][1][2][2]=((1.0+((tmp1*2.0)*njac[i][j][k][2][2]))+((tmp1*2.0)*dz3));
				lhs[i][j][k][1][2][3]=((tmp1*2.0)*njac[i][j][k][2][3]);
				lhs[i][j][k][1][2][4]=((tmp1*2.0)*njac[i][j][k][2][4]);
				lhs[i][j][k][1][3][0]=((tmp1*2.0)*njac[i][j][k][3][0]);
				lhs[i][j][k][1][3][1]=((tmp1*2.0)*njac[i][j][k][3][1]);
				lhs[i][j][k][1][3][2]=((tmp1*2.0)*njac[i][j][k][3][2]);
				lhs[i][j][k][1][3][3]=((1.0+((tmp1*2.0)*njac[i][j][k][3][3]))+((tmp1*2.0)*dz4));
				lhs[i][j][k][1][3][4]=((tmp1*2.0)*njac[i][j][k][3][4]);
				lhs[i][j][k][1][4][0]=((tmp1*2.0)*njac[i][j][k][4][0]);
				lhs[i][j][k][1][4][1]=((tmp1*2.0)*njac[i][j][k][4][1]);
				lhs[i][j][k][1][4][2]=((tmp1*2.0)*njac[i][j][k][4][2]);
				lhs[i][j][k][1][4][3]=((tmp1*2.0)*njac[i][j][k][4][3]);
				lhs[i][j][k][1][4][4]=((1.0+((tmp1*2.0)*njac[i][j][k][4][4]))+((tmp1*2.0)*dz5));
				lhs[i][j][k][2][0][0]=(((tmp2*fjac[i][j][k+1][0][0])-(tmp1*njac[i][j][k+1][0][0]))-(tmp1*dz1));
				lhs[i][j][k][2][0][1]=((tmp2*fjac[i][j][k+1][0][1])-(tmp1*njac[i][j][k+1][0][1]));
				lhs[i][j][k][2][0][2]=((tmp2*fjac[i][j][k+1][0][2])-(tmp1*njac[i][j][k+1][0][2]));
				lhs[i][j][k][2][0][3]=((tmp2*fjac[i][j][k+1][0][3])-(tmp1*njac[i][j][k+1][0][3]));
				lhs[i][j][k][2][0][4]=((tmp2*fjac[i][j][k+1][0][4])-(tmp1*njac[i][j][k+1][0][4]));
				lhs[i][j][k][2][1][0]=((tmp2*fjac[i][j][k+1][1][0])-(tmp1*njac[i][j][k+1][1][0]));
				lhs[i][j][k][2][1][1]=(((tmp2*fjac[i][j][k+1][1][1])-(tmp1*njac[i][j][k+1][1][1]))-(tmp1*dz2));
				lhs[i][j][k][2][1][2]=((tmp2*fjac[i][j][k+1][1][2])-(tmp1*njac[i][j][k+1][1][2]));
				lhs[i][j][k][2][1][3]=((tmp2*fjac[i][j][k+1][1][3])-(tmp1*njac[i][j][k+1][1][3]));
				lhs[i][j][k][2][1][4]=((tmp2*fjac[i][j][k+1][1][4])-(tmp1*njac[i][j][k+1][1][4]));
				lhs[i][j][k][2][2][0]=((tmp2*fjac[i][j][k+1][2][0])-(tmp1*njac[i][j][k+1][2][0]));
				lhs[i][j][k][2][2][1]=((tmp2*fjac[i][j][k+1][2][1])-(tmp1*njac[i][j][k+1][2][1]));
				lhs[i][j][k][2][2][2]=(((tmp2*fjac[i][j][k+1][2][2])-(tmp1*njac[i][j][k+1][2][2]))-(tmp1*dz3));
				lhs[i][j][k][2][2][3]=((tmp2*fjac[i][j][k+1][2][3])-(tmp1*njac[i][j][k+1][2][3]));
				lhs[i][j][k][2][2][4]=((tmp2*fjac[i][j][k+1][2][4])-(tmp1*njac[i][j][k+1][2][4]));
				lhs[i][j][k][2][3][0]=((tmp2*fjac[i][j][k+1][3][0])-(tmp1*njac[i][j][k+1][3][0]));
				lhs[i][j][k][2][3][1]=((tmp2*fjac[i][j][k+1][3][1])-(tmp1*njac[i][j][k+1][3][1]));
				lhs[i][j][k][2][3][2]=((tmp2*fjac[i][j][k+1][3][2])-(tmp1*njac[i][j][k+1][3][2]));
				lhs[i][j][k][2][3][3]=(((tmp2*fjac[i][j][k+1][3][3])-(tmp1*njac[i][j][k+1][3][3]))-(tmp1*dz4));
				lhs[i][j][k][2][3][4]=((tmp2*fjac[i][j][k+1][3][4])-(tmp1*njac[i][j][k+1][3][4]));
				lhs[i][j][k][2][4][0]=((tmp2*fjac[i][j][k+1][4][0])-(tmp1*njac[i][j][k+1][4][0]));
				lhs[i][j][k][2][4][1]=((tmp2*fjac[i][j][k+1][4][1])-(tmp1*njac[i][j][k+1][4][1]));
				lhs[i][j][k][2][4][2]=((tmp2*fjac[i][j][k+1][4][2])-(tmp1*njac[i][j][k+1][4][2]));
				lhs[i][j][k][2][4][3]=((tmp2*fjac[i][j][k+1][4][3])-(tmp1*njac[i][j][k+1][4][3]));
				lhs[i][j][k][2][4][4]=(((tmp2*fjac[i][j][k+1][4][4])-(tmp1*njac[i][j][k+1][4][4]))-(tmp1*dz5));
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
	int i, j, k, m;
	double rho_inv, uijk, up1, um1, vijk, vp1, vm1, wijk, wp1, wm1;
	/*
	--------------------------------------------------------------------
	c     compute the reciprocal of density, and the kinetic energy, 
	c     and the speed of sound.
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#0 
	for (i=0; i<grid_points[0]; i ++ )
	{
		#pragma loop name compute_rhs#0#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name compute_rhs#0#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				rho_inv=(1.0/u[i][j][k][0]);
				rho_i[i][j][k]=rho_inv;
				us[i][j][k]=(u[i][j][k][1]*rho_inv);
				vs[i][j][k]=(u[i][j][k][2]*rho_inv);
				ws[i][j][k]=(u[i][j][k][3]*rho_inv);
				square[i][j][k]=((0.5*(((u[i][j][k][1]*u[i][j][k][1])+(u[i][j][k][2]*u[i][j][k][2]))+(u[i][j][k][3]*u[i][j][k][3])))*rho_inv);
				qs[i][j][k]=(square[i][j][k]*rho_inv);
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
	for (i=0; i<grid_points[0]; i ++ )
	{
		#pragma loop name compute_rhs#1#0 
		for (j=0; j<grid_points[1]; j ++ )
		{
			#pragma loop name compute_rhs#1#0#0 
			for (k=0; k<grid_points[2]; k ++ )
			{
				#pragma loop name compute_rhs#1#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[i][j][k][m]=forcing[i][j][k][m];
				}
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     compute xi-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#2 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#2#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				uijk=us[i][j][k];
				up1=us[i+1][j][k];
				um1=us[i-1][j][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(dx1tx1*((u[i+1][j][k][0]-(2.0*u[i][j][k][0]))+u[i-1][j][k][0])))-(tx2*(u[i+1][j][k][1]-u[i-1][j][k][1])));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(dx2tx1*((u[i+1][j][k][1]-(2.0*u[i][j][k][1]))+u[i-1][j][k][1])))+((xxcon2*con43)*((up1-(2.0*uijk))+um1)))-(tx2*(((u[i+1][j][k][1]*up1)-(u[i-1][j][k][1]*um1))+((((u[i+1][j][k][4]-square[i+1][j][k])-u[i-1][j][k][4])+square[i-1][j][k])*c2))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(dx3tx1*((u[i+1][j][k][2]-(2.0*u[i][j][k][2]))+u[i-1][j][k][2])))+(xxcon2*((vs[i+1][j][k]-(2.0*vs[i][j][k]))+vs[i-1][j][k])))-(tx2*((u[i+1][j][k][2]*up1)-(u[i-1][j][k][2]*um1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(dx4tx1*((u[i+1][j][k][3]-(2.0*u[i][j][k][3]))+u[i-1][j][k][3])))+(xxcon2*((ws[i+1][j][k]-(2.0*ws[i][j][k]))+ws[i-1][j][k])))-(tx2*((u[i+1][j][k][3]*up1)-(u[i-1][j][k][3]*um1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(dx5tx1*((u[i+1][j][k][4]-(2.0*u[i][j][k][4]))+u[i-1][j][k][4])))+(xxcon3*((qs[i+1][j][k]-(2.0*qs[i][j][k]))+qs[i-1][j][k])))+(xxcon4*(((up1*up1)-((2.0*uijk)*uijk))+(um1*um1))))+(xxcon5*(((u[i+1][j][k][4]*rho_i[i+1][j][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u[i-1][j][k][4]*rho_i[i-1][j][k]))))-(tx2*((((c1*u[i+1][j][k][4])-(c2*square[i+1][j][k]))*up1)-(((c1*u[i-1][j][k][4])-(c2*square[i-1][j][k]))*um1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     add fourth order xi-direction dissipation               
	c-------------------------------------------------------------------
	*/
	i=1;
	#pragma loop name compute_rhs#3 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name compute_rhs#3#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#3#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((5.0*u[i][j][k][m])-(4.0*u[i+1][j][k][m]))+u[i+2][j][k][m])));
			}
		}
	}
	i=2;
	#pragma loop name compute_rhs#4 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name compute_rhs#4#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#4#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((( - 4.0)*u[i-1][j][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[i+1][j][k][m]))+u[i+2][j][k][m])));
			}
		}
	}
	#pragma loop name compute_rhs#5 
	for (i=3; i<(grid_points[0]-3); i ++ )
	{
		#pragma loop name compute_rhs#5#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#5#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name compute_rhs#5#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((u[i-2][j][k][m]-(4.0*u[i-1][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i+1][j][k][m]))+u[i+2][j][k][m])));
				}
			}
		}
	}
	i=(grid_points[0]-3);
	#pragma loop name compute_rhs#6 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name compute_rhs#6#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#6#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((u[i-2][j][k][m]-(4.0*u[i-1][j][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i+1][j][k][m]))));
			}
		}
	}
	i=(grid_points[0]-2);
	#pragma loop name compute_rhs#7 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name compute_rhs#7#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#7#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((u[i-2][j][k][m]-(4.0*u[i-1][j][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     compute eta-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#8 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#8#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#8#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				vijk=vs[i][j][k];
				vp1=vs[i][j+1][k];
				vm1=vs[i][j-1][k];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(dy1ty1*((u[i][j+1][k][0]-(2.0*u[i][j][k][0]))+u[i][j-1][k][0])))-(ty2*(u[i][j+1][k][2]-u[i][j-1][k][2])));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(dy2ty1*((u[i][j+1][k][1]-(2.0*u[i][j][k][1]))+u[i][j-1][k][1])))+(yycon2*((us[i][j+1][k]-(2.0*us[i][j][k]))+us[i][j-1][k])))-(ty2*((u[i][j+1][k][1]*vp1)-(u[i][j-1][k][1]*vm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(dy3ty1*((u[i][j+1][k][2]-(2.0*u[i][j][k][2]))+u[i][j-1][k][2])))+((yycon2*con43)*((vp1-(2.0*vijk))+vm1)))-(ty2*(((u[i][j+1][k][2]*vp1)-(u[i][j-1][k][2]*vm1))+((((u[i][j+1][k][4]-square[i][j+1][k])-u[i][j-1][k][4])+square[i][j-1][k])*c2))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(dy4ty1*((u[i][j+1][k][3]-(2.0*u[i][j][k][3]))+u[i][j-1][k][3])))+(yycon2*((ws[i][j+1][k]-(2.0*ws[i][j][k]))+ws[i][j-1][k])))-(ty2*((u[i][j+1][k][3]*vp1)-(u[i][j-1][k][3]*vm1))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(dy5ty1*((u[i][j+1][k][4]-(2.0*u[i][j][k][4]))+u[i][j-1][k][4])))+(yycon3*((qs[i][j+1][k]-(2.0*qs[i][j][k]))+qs[i][j-1][k])))+(yycon4*(((vp1*vp1)-((2.0*vijk)*vijk))+(vm1*vm1))))+(yycon5*(((u[i][j+1][k][4]*rho_i[i][j+1][k])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u[i][j-1][k][4]*rho_i[i][j-1][k]))))-(ty2*((((c1*u[i][j+1][k][4])-(c2*square[i][j+1][k]))*vp1)-(((c1*u[i][j-1][k][4])-(c2*square[i][j-1][k]))*vm1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     add fourth order eta-direction dissipation         
	c-------------------------------------------------------------------
	*/
	j=1;
	#pragma loop name compute_rhs#9 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#9#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#9#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][j+1][k][m]))+u[i][j+2][k][m])));
			}
		}
	}
	j=2;
	#pragma loop name compute_rhs#10 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#10#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#10#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((( - 4.0)*u[i][j-1][k][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][j+1][k][m]))+u[i][j+2][k][m])));
			}
		}
	}
	#pragma loop name compute_rhs#11 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#11#0 
		for (j=3; j<(grid_points[1]-3); j ++ )
		{
			#pragma loop name compute_rhs#11#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name compute_rhs#11#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((u[i][j-2][k][m]-(4.0*u[i][j-1][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j+1][k][m]))+u[i][j+2][k][m])));
				}
			}
		}
	}
	j=(grid_points[1]-3);
	#pragma loop name compute_rhs#12 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#12#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#12#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((u[i][j-2][k][m]-(4.0*u[i][j-1][k][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j+1][k][m]))));
			}
		}
	}
	j=(grid_points[1]-2);
	#pragma loop name compute_rhs#13 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#13#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#13#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((u[i][j-2][k][m]-(4.0*u[i][j-1][k][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     compute zeta-direction fluxes 
	c-------------------------------------------------------------------
	*/
	#pragma loop name compute_rhs#14 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#14#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#14#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				wijk=ws[i][j][k];
				wp1=ws[i][j][k+1];
				wm1=ws[i][j][k-1];
				rhs[i][j][k][0]=((rhs[i][j][k][0]+(dz1tz1*((u[i][j][k+1][0]-(2.0*u[i][j][k][0]))+u[i][j][k-1][0])))-(tz2*(u[i][j][k+1][3]-u[i][j][k-1][3])));
				rhs[i][j][k][1]=(((rhs[i][j][k][1]+(dz2tz1*((u[i][j][k+1][1]-(2.0*u[i][j][k][1]))+u[i][j][k-1][1])))+(zzcon2*((us[i][j][k+1]-(2.0*us[i][j][k]))+us[i][j][k-1])))-(tz2*((u[i][j][k+1][1]*wp1)-(u[i][j][k-1][1]*wm1))));
				rhs[i][j][k][2]=(((rhs[i][j][k][2]+(dz3tz1*((u[i][j][k+1][2]-(2.0*u[i][j][k][2]))+u[i][j][k-1][2])))+(zzcon2*((vs[i][j][k+1]-(2.0*vs[i][j][k]))+vs[i][j][k-1])))-(tz2*((u[i][j][k+1][2]*wp1)-(u[i][j][k-1][2]*wm1))));
				rhs[i][j][k][3]=(((rhs[i][j][k][3]+(dz4tz1*((u[i][j][k+1][3]-(2.0*u[i][j][k][3]))+u[i][j][k-1][3])))+((zzcon2*con43)*((wp1-(2.0*wijk))+wm1)))-(tz2*(((u[i][j][k+1][3]*wp1)-(u[i][j][k-1][3]*wm1))+((((u[i][j][k+1][4]-square[i][j][k+1])-u[i][j][k-1][4])+square[i][j][k-1])*c2))));
				rhs[i][j][k][4]=(((((rhs[i][j][k][4]+(dz5tz1*((u[i][j][k+1][4]-(2.0*u[i][j][k][4]))+u[i][j][k-1][4])))+(zzcon3*((qs[i][j][k+1]-(2.0*qs[i][j][k]))+qs[i][j][k-1])))+(zzcon4*(((wp1*wp1)-((2.0*wijk)*wijk))+(wm1*wm1))))+(zzcon5*(((u[i][j][k+1][4]*rho_i[i][j][k+1])-((2.0*u[i][j][k][4])*rho_i[i][j][k]))+(u[i][j][k-1][4]*rho_i[i][j][k-1]))))-(tz2*((((c1*u[i][j][k+1][4])-(c2*square[i][j][k+1]))*wp1)-(((c1*u[i][j][k-1][4])-(c2*square[i][j][k-1]))*wm1))));
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     add fourth order zeta-direction dissipation                
	c-------------------------------------------------------------------
	*/
	k=1;
	#pragma loop name compute_rhs#15 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#15#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#15#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((5.0*u[i][j][k][m])-(4.0*u[i][j][k+1][m]))+u[i][j][k+2][m])));
			}
		}
	}
	k=2;
	#pragma loop name compute_rhs#16 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#16#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#16#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((( - 4.0)*u[i][j][k-1][m])+(6.0*u[i][j][k][m]))-(4.0*u[i][j][k+1][m]))+u[i][j][k+2][m])));
			}
		}
	}
	#pragma loop name compute_rhs#17 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#17#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#17#0#0 
			for (k=3; k<(grid_points[2]-3); k ++ )
			{
				#pragma loop name compute_rhs#17#0#0#0 
				for (m=0; m<5; m ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((((u[i][j][k-2][m]-(4.0*u[i][j][k-1][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][k+1][m]))+u[i][j][k+2][m])));
				}
			}
		}
	}
	k=(grid_points[2]-3);
	#pragma loop name compute_rhs#18 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#18#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#18#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*(((u[i][j][k-2][m]-(4.0*u[i][j][k-1][m]))+(6.0*u[i][j][k][m]))-(4.0*u[i][j][k+1][m]))));
			}
		}
	}
	k=(grid_points[2]-2);
	#pragma loop name compute_rhs#19 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name compute_rhs#19#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name compute_rhs#19#0#0 
			for (m=0; m<5; m ++ )
			{
				rhs[i][j][k][m]=(rhs[i][j][k][m]-(dssp*((u[i][j][k-2][m]-(4.0*u[i][j][k-1][m]))+(5.0*u[i][j][k][m]))));
			}
		}
	}
	#pragma loop name compute_rhs#20 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name compute_rhs#20#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			#pragma loop name compute_rhs#20#0#0 
			for (m=0; m<5; m ++ )
			{
				#pragma loop name compute_rhs#20#0#0#0 
				#pragma cetus parallel 
				#pragma omp parallel for
				for (i=1; i<(grid_points[0]-1); i ++ )
				{
					rhs[i][j][k][m]=(rhs[i][j][k][m]*dt);
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
	ce[0][1]=0.0;
	ce[0][2]=0.0;
	ce[0][3]=4.0;
	ce[0][4]=5.0;
	ce[0][5]=3.0;
	ce[0][6]=0.5;
	ce[0][7]=0.02;
	ce[0][8]=0.01;
	ce[0][9]=0.03;
	ce[0][10]=0.5;
	ce[0][11]=0.4;
	ce[0][12]=0.3;
	ce[1][0]=1.0;
	ce[1][1]=0.0;
	ce[1][2]=0.0;
	ce[1][3]=0.0;
	ce[1][4]=1.0;
	ce[1][5]=2.0;
	ce[1][6]=3.0;
	ce[1][7]=0.01;
	ce[1][8]=0.03;
	ce[1][9]=0.02;
	ce[1][10]=0.4;
	ce[1][11]=0.3;
	ce[1][12]=0.5;
	ce[2][0]=2.0;
	ce[2][1]=2.0;
	ce[2][2]=0.0;
	ce[2][3]=0.0;
	ce[2][4]=0.0;
	ce[2][5]=2.0;
	ce[2][6]=3.0;
	ce[2][7]=0.04;
	ce[2][8]=0.03;
	ce[2][9]=0.05;
	ce[2][10]=0.3;
	ce[2][11]=0.5;
	ce[2][12]=0.4;
	ce[3][0]=2.0;
	ce[3][1]=2.0;
	ce[3][2]=0.0;
	ce[3][3]=0.0;
	ce[3][4]=0.0;
	ce[3][5]=2.0;
	ce[3][6]=3.0;
	ce[3][7]=0.03;
	ce[3][8]=0.05;
	ce[3][9]=0.04;
	ce[3][10]=0.2;
	ce[3][11]=0.1;
	ce[3][12]=0.3;
	ce[4][0]=5.0;
	ce[4][1]=4.0;
	ce[4][2]=3.0;
	ce[4][3]=2.0;
	ce[4][4]=0.1;
	ce[4][5]=0.4;
	ce[4][6]=0.3;
	ce[4][7]=0.05;
	ce[4][8]=0.04;
	ce[4][9]=0.03;
	ce[4][10]=0.1;
	ce[4][11]=0.3;
	ce[4][12]=0.2;
	c1=1.4;
	c2=0.4;
	c3=0.1;
	c4=1.0;
	c5=1.4;
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
static void verify(int no_time_steps, char * class, boolean * verified)
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c  verification routine                         
	c-------------------------------------------------------------------
	*/
	double xcrref[5], xceref[5], xcrdif[5], xcedif[5], epsilon, xce[5], xcr[5], dtref;
	int m;
	/*
	--------------------------------------------------------------------
	c   tolerance level
	c-------------------------------------------------------------------
	*/
	epsilon=1.0E-8;
	/*
	--------------------------------------------------------------------
	c   compute the error norm and the residual norm, and exit if not printing
	c-------------------------------------------------------------------
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
	c    reference data for 12X12X12 grids after 100 time steps, with DT = 1.0d-02
	c-------------------------------------------------------------------
	*/
	if ((((grid_points[0]==12)&&(grid_points[1]==12))&&(grid_points[2]==12))&&(no_time_steps==60))
	{
		( * class)='S';
		dtref=0.01;
		/*
		--------------------------------------------------------------------
		c  Reference values of RMS-norms of residual.
		c-------------------------------------------------------------------
		*/
		xcrref[0]=0.17034283709541312;
		xcrref[1]=0.012975252070034096;
		xcrref[2]=0.032527926989486054;
		xcrref[3]=0.0264364212751668;
		xcrref[4]=0.1921178413174443;
		/*
		--------------------------------------------------------------------
		c  Reference values of RMS-norms of solution error.
		c-------------------------------------------------------------------
		*/
		xceref[0]=4.997691334581158E-4;
		xceref[1]=4.519566678296193E-5;
		xceref[2]=7.397376517292135E-5;
		xceref[3]=7.382123863243973E-5;
		xceref[4]=8.926963098749145E-4;
		/*
		--------------------------------------------------------------------
		c    reference data for 24X24X24 grids after 200 time steps, with DT = 0.8d-3
		c-------------------------------------------------------------------
		*/
	}
	else
	{
		if ((((grid_points[0]==24)&&(grid_points[1]==24))&&(grid_points[2]==24))&&(no_time_steps==200))
		{
			( * class)='W';
			dtref=8.0E-4;
			/*
			--------------------------------------------------------------------
			c  Reference values of RMS-norms of residual.
			c-------------------------------------------------------------------
			*/
			xcrref[0]=112.5590409344;
			xcrref[1]=11.80007595731;
			xcrref[2]=27.10329767846;
			xcrref[3]=24.69174937669;
			xcrref[4]=263.8427874317;
			/*
			--------------------------------------------------------------------
			c  Reference values of RMS-norms of solution error.
			c-------------------------------------------------------------------
			*/
			xceref[0]=4.419655736008;
			xceref[1]=0.4638531260002;
			xceref[2]=1.011551749967;
			xceref[3]=0.9235878729944;
			xceref[4]=10.18045837718;
			/*
			--------------------------------------------------------------------
			c    reference data for 64X64X64 grids after 200 time steps, with DT = 0.8d-3
			c-------------------------------------------------------------------
			*/
		}
		else
		{
			if ((((grid_points[0]==64)&&(grid_points[1]==64))&&(grid_points[2]==64))&&(no_time_steps==200))
			{
				( * class)='A';
				dtref=8.0E-4;
				/*
				--------------------------------------------------------------------
				c  Reference values of RMS-norms of residual.
				c-------------------------------------------------------------------
				*/
				xcrref[0]=108.06346714637264;
				xcrref[1]=11.319730901220813;
				xcrref[2]=25.974354511582465;
				xcrref[3]=23.66562254467891;
				xcrref[4]=252.78963211748345;
				/*
				--------------------------------------------------------------------
				c  Reference values of RMS-norms of solution error.
				c-------------------------------------------------------------------
				*/
				xceref[0]=4.2348416040525025;
				xceref[1]=0.443902824969957;
				xceref[2]=0.9669248013634565;
				xceref[3]=0.8830206303976548;
				xceref[4]=9.737990177082928;
				/*
				--------------------------------------------------------------------
				c    reference data for 102X102X102 grids after 200 time steps,
				c    with DT = 3.0d-04
				c-------------------------------------------------------------------
				*/
			}
			else
			{
				if ((((grid_points[0]==102)&&(grid_points[1]==102))&&(grid_points[2]==102))&&(no_time_steps==200))
				{
					( * class)='B';
					dtref=3.0E-4;
					/*
					--------------------------------------------------------------------
					c  Reference values of RMS-norms of residual.
					c-------------------------------------------------------------------
					*/
					xcrref[0]=1423.3597229287254;
					xcrref[1]=99.33052259015024;
					xcrref[2]=356.46025644535285;
					xcrref[3]=324.8544795908409;
					xcrref[4]=3270.7541254659363;
					/*
					--------------------------------------------------------------------
					c  Reference values of RMS-norms of solution error.
					c-------------------------------------------------------------------
					*/
					xceref[0]=52.96984714093686;
					xceref[1]=4.463289611567067;
					xceref[2]=13.122573342210174;
					xceref[3]=12.006925323559145;
					xceref[4]=124.59576151035986;
					/*
					--------------------------------------------------------------------
					c    reference data for 162X162X162 grids after 200 time steps,
					c    with DT = 1.0d-04
					c-------------------------------------------------------------------
					*/
				}
				else
				{
					if ((((grid_points[0]==162)&&(grid_points[1]==162))&&(grid_points[2]==162))&&(no_time_steps==200))
					{
						( * class)='C';
						dtref=1.0E-4;
						/*
						--------------------------------------------------------------------
						c  Reference values of RMS-norms of residual.
						c-------------------------------------------------------------------
						*/
						xcrref[0]=6239.8116551764615;
						xcrref[1]=507.93239190423964;
						xcrref[2]=1542.3530093013596;
						xcrref[3]=1330.238792929119;
						xcrref[4]=11604.087428436455;
						/*
						--------------------------------------------------------------------
						c  Reference values of RMS-norms of solution error.
						c-------------------------------------------------------------------
						*/
						xceref[0]=164.62008369091265;
						xceref[1]=11.497107903824313;
						xceref[2]=41.20744620746151;
						xceref[3]=37.08765105969417;
						xceref[4]=362.11053051841265;
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
	c-------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c    Compute the difference of solution values and the known reference values.
	c-------------------------------------------------------------------
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
	c-------------------------------------------------------------------
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
		if (( * verified)==1)
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
	c     
	c     Performs line solves in X direction by first factoring
	c     the block-tridiagonal matrix into an upper triangular matrix, 
	c     and then performing back substitution to solve for the unknow
	c     vectors of each line.  
	c     
	c     Make sure we treat elements zero to cell_size in the direction
	c     of the sweep.
	c     
	c-------------------------------------------------------------------
	*/
	lhsx();
	x_solve_cell();
	x_backsubstitute();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void x_backsubstitute(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     back solve: if last cell, then generate U(isize)=rhs[isize)
	c     else assume U(isize) is loaded in un pack backsub_info
	c     so just use it
	c     after call u(istart) will be sent to next cell
	c-------------------------------------------------------------------
	*/
	int i, j, k, m, n;
	#pragma loop name x_backsubstitute#0 
	for (i=(grid_points[0]-2); i>=0; i -- )
	{
		#pragma loop name x_backsubstitute#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name x_backsubstitute#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name x_backsubstitute#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					#pragma loop name x_backsubstitute#0#0#0#0#0 
					for (n=0; n<5; n ++ )
					{
						rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i+1][j][k][n]));
					}
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
static void x_solve_cell(void )
{
	/*
	--------------------------------------------------------------------
	c     performs guaussian elimination on this cell.
	c     
	c     assumes that unpacking routines for non-first cells 
	c     preload C' and rhs' from previous cell.
	c     
	c     assumed send happens outside this routine, but that
	c     c'(IMAX) and rhs'(IMAX) will be sent to next cell
	c-------------------------------------------------------------------
	*/
	int i, j, k, isize;
	isize=(grid_points[0]-1);
	/*
	--------------------------------------------------------------------
	c     outer most do loops - sweeping in i direction
	c-------------------------------------------------------------------
	*/
	#pragma loop name x_solve_cell#0 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name x_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			--------------------------------------------------------------------
			c     multiply c(0,j,k) by b_inverse and copy back to c
			c     multiply rhs(0) by b_inverse(0) and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvcrhs(lhs[0][j][k][1], lhs[0][j][k][2], rhs[0][j][k]);
		}
	}
	/*
	--------------------------------------------------------------------
	c     begin inner most do loop
	c     do all the elements of the cell unless last 
	c-------------------------------------------------------------------
	*/
	#pragma loop name x_solve_cell#1 
	for (i=1; i<isize; i ++ )
	{
		#pragma loop name x_solve_cell#1#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name x_solve_cell#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				/*
				--------------------------------------------------------------------
				c     rhs(i) = rhs(i) - Arhs(i-1)
				c-------------------------------------------------------------------
				*/
				matvec_sub(lhs[i][j][k][0], rhs[i-1][j][k], rhs[i][j][k]);
				/*
				--------------------------------------------------------------------
				c     B(i) = B(i) - C(i-1)A(i)
				c-------------------------------------------------------------------
				*/
				matmul_sub(lhs[i][j][k][0], lhs[i-1][j][k][2], lhs[i][j][k][1]);
				/*
				--------------------------------------------------------------------
				c     multiply c(i,j,k) by b_inverse and copy back to c
				c     multiply rhs(1,j,k) by b_inverse(1,j,k) and copy to rhs
				c-------------------------------------------------------------------
				*/
				binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
			}
		}
	}
	#pragma loop name x_solve_cell#2 
	for (j=1; j<(grid_points[1]-1); j ++ )
	{
		#pragma loop name x_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			--------------------------------------------------------------------
			c     rhs(isize) = rhs(isize) - Arhs(isize-1)
			c-------------------------------------------------------------------
			*/
			matvec_sub(lhs[isize][j][k][0], rhs[isize-1][j][k], rhs[isize][j][k]);
			/*
			--------------------------------------------------------------------
			c     B(isize) = B(isize) - C(isize-1)A(isize)
			c-------------------------------------------------------------------
			*/
			matmul_sub(lhs[isize][j][k][0], lhs[isize-1][j][k][2], lhs[isize][j][k][1]);
			/*
			--------------------------------------------------------------------
			c     multiply rhs() by b_inverse() and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvrhs(lhs[i][j][k][1], rhs[i][j][k]);
		}
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void matvec_sub(double ablock[5][5], double avec[5], double bvec[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     subtracts bvec=bvec - ablockavec
	c-------------------------------------------------------------------
	*/
	int i;
	#pragma loop name matvec_sub#0 
	for (i=0; i<5; i ++ )
	{
		/*
		--------------------------------------------------------------------
		c            rhs(i,ic,jc,kc,ccell) = rhs(i,ic,jc,kc,ccell) 
		c     $           - lhs[i,1,ablock,ia,ja,ka,acell)
		c-------------------------------------------------------------------
		*/
		bvec[i]=(((((bvec[i]-(ablock[i][0]*avec[0]))-(ablock[i][1]*avec[1]))-(ablock[i][2]*avec[2]))-(ablock[i][3]*avec[3]))-(ablock[i][4]*avec[4]));
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void matmul_sub(double ablock[5][5], double bblock[5][5], double cblock[5][5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     subtracts a(i,j,k) X b(i,j,k) from c(i,j,k)
	c-------------------------------------------------------------------
	*/
	int j;
	#pragma loop name matmul_sub#0 
	for (j=0; j<5; j ++ )
	{
		cblock[0][j]=(((((cblock[0][j]-(ablock[0][0]*bblock[0][j]))-(ablock[0][1]*bblock[1][j]))-(ablock[0][2]*bblock[2][j]))-(ablock[0][3]*bblock[3][j]))-(ablock[0][4]*bblock[4][j]));
		cblock[1][j]=(((((cblock[1][j]-(ablock[1][0]*bblock[0][j]))-(ablock[1][1]*bblock[1][j]))-(ablock[1][2]*bblock[2][j]))-(ablock[1][3]*bblock[3][j]))-(ablock[1][4]*bblock[4][j]));
		cblock[2][j]=(((((cblock[2][j]-(ablock[2][0]*bblock[0][j]))-(ablock[2][1]*bblock[1][j]))-(ablock[2][2]*bblock[2][j]))-(ablock[2][3]*bblock[3][j]))-(ablock[2][4]*bblock[4][j]));
		cblock[3][j]=(((((cblock[3][j]-(ablock[3][0]*bblock[0][j]))-(ablock[3][1]*bblock[1][j]))-(ablock[3][2]*bblock[2][j]))-(ablock[3][3]*bblock[3][j]))-(ablock[3][4]*bblock[4][j]));
		cblock[4][j]=(((((cblock[4][j]-(ablock[4][0]*bblock[0][j]))-(ablock[4][1]*bblock[1][j]))-(ablock[4][2]*bblock[2][j]))-(ablock[4][3]*bblock[3][j]))-(ablock[4][4]*bblock[4][j]));
	}
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void binvcrhs(double lhs[5][5], double c[5][5], double r[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	double pivot, coeff;
	/*
	--------------------------------------------------------------------
	c     
	c-------------------------------------------------------------------
	*/
	pivot=(1.0/lhs[0][0]);
	lhs[0][1]=(lhs[0][1]*pivot);
	lhs[0][2]=(lhs[0][2]*pivot);
	lhs[0][3]=(lhs[0][3]*pivot);
	lhs[0][4]=(lhs[0][4]*pivot);
	c[0][0]=(c[0][0]*pivot);
	c[0][1]=(c[0][1]*pivot);
	c[0][2]=(c[0][2]*pivot);
	c[0][3]=(c[0][3]*pivot);
	c[0][4]=(c[0][4]*pivot);
	r[0]=(r[0]*pivot);
	coeff=lhs[1][0];
	lhs[1][1]=(lhs[1][1]-(coeff*lhs[0][1]));
	lhs[1][2]=(lhs[1][2]-(coeff*lhs[0][2]));
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[0][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[0][4]));
	c[1][0]=(c[1][0]-(coeff*c[0][0]));
	c[1][1]=(c[1][1]-(coeff*c[0][1]));
	c[1][2]=(c[1][2]-(coeff*c[0][2]));
	c[1][3]=(c[1][3]-(coeff*c[0][3]));
	c[1][4]=(c[1][4]-(coeff*c[0][4]));
	r[1]=(r[1]-(coeff*r[0]));
	coeff=lhs[2][0];
	lhs[2][1]=(lhs[2][1]-(coeff*lhs[0][1]));
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[0][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[0][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[0][4]));
	c[2][0]=(c[2][0]-(coeff*c[0][0]));
	c[2][1]=(c[2][1]-(coeff*c[0][1]));
	c[2][2]=(c[2][2]-(coeff*c[0][2]));
	c[2][3]=(c[2][3]-(coeff*c[0][3]));
	c[2][4]=(c[2][4]-(coeff*c[0][4]));
	r[2]=(r[2]-(coeff*r[0]));
	coeff=lhs[3][0];
	lhs[3][1]=(lhs[3][1]-(coeff*lhs[0][1]));
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[0][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[0][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[0][4]));
	c[3][0]=(c[3][0]-(coeff*c[0][0]));
	c[3][1]=(c[3][1]-(coeff*c[0][1]));
	c[3][2]=(c[3][2]-(coeff*c[0][2]));
	c[3][3]=(c[3][3]-(coeff*c[0][3]));
	c[3][4]=(c[3][4]-(coeff*c[0][4]));
	r[3]=(r[3]-(coeff*r[0]));
	coeff=lhs[4][0];
	lhs[4][1]=(lhs[4][1]-(coeff*lhs[0][1]));
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[0][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[0][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[0][4]));
	c[4][0]=(c[4][0]-(coeff*c[0][0]));
	c[4][1]=(c[4][1]-(coeff*c[0][1]));
	c[4][2]=(c[4][2]-(coeff*c[0][2]));
	c[4][3]=(c[4][3]-(coeff*c[0][3]));
	c[4][4]=(c[4][4]-(coeff*c[0][4]));
	r[4]=(r[4]-(coeff*r[0]));
	pivot=(1.0/lhs[1][1]);
	lhs[1][2]=(lhs[1][2]*pivot);
	lhs[1][3]=(lhs[1][3]*pivot);
	lhs[1][4]=(lhs[1][4]*pivot);
	c[1][0]=(c[1][0]*pivot);
	c[1][1]=(c[1][1]*pivot);
	c[1][2]=(c[1][2]*pivot);
	c[1][3]=(c[1][3]*pivot);
	c[1][4]=(c[1][4]*pivot);
	r[1]=(r[1]*pivot);
	coeff=lhs[0][1];
	lhs[0][2]=(lhs[0][2]-(coeff*lhs[1][2]));
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[1][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[1][4]));
	c[0][0]=(c[0][0]-(coeff*c[1][0]));
	c[0][1]=(c[0][1]-(coeff*c[1][1]));
	c[0][2]=(c[0][2]-(coeff*c[1][2]));
	c[0][3]=(c[0][3]-(coeff*c[1][3]));
	c[0][4]=(c[0][4]-(coeff*c[1][4]));
	r[0]=(r[0]-(coeff*r[1]));
	coeff=lhs[2][1];
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[1][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[1][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[1][4]));
	c[2][0]=(c[2][0]-(coeff*c[1][0]));
	c[2][1]=(c[2][1]-(coeff*c[1][1]));
	c[2][2]=(c[2][2]-(coeff*c[1][2]));
	c[2][3]=(c[2][3]-(coeff*c[1][3]));
	c[2][4]=(c[2][4]-(coeff*c[1][4]));
	r[2]=(r[2]-(coeff*r[1]));
	coeff=lhs[3][1];
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[1][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[1][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[1][4]));
	c[3][0]=(c[3][0]-(coeff*c[1][0]));
	c[3][1]=(c[3][1]-(coeff*c[1][1]));
	c[3][2]=(c[3][2]-(coeff*c[1][2]));
	c[3][3]=(c[3][3]-(coeff*c[1][3]));
	c[3][4]=(c[3][4]-(coeff*c[1][4]));
	r[3]=(r[3]-(coeff*r[1]));
	coeff=lhs[4][1];
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[1][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[1][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[1][4]));
	c[4][0]=(c[4][0]-(coeff*c[1][0]));
	c[4][1]=(c[4][1]-(coeff*c[1][1]));
	c[4][2]=(c[4][2]-(coeff*c[1][2]));
	c[4][3]=(c[4][3]-(coeff*c[1][3]));
	c[4][4]=(c[4][4]-(coeff*c[1][4]));
	r[4]=(r[4]-(coeff*r[1]));
	pivot=(1.0/lhs[2][2]);
	lhs[2][3]=(lhs[2][3]*pivot);
	lhs[2][4]=(lhs[2][4]*pivot);
	c[2][0]=(c[2][0]*pivot);
	c[2][1]=(c[2][1]*pivot);
	c[2][2]=(c[2][2]*pivot);
	c[2][3]=(c[2][3]*pivot);
	c[2][4]=(c[2][4]*pivot);
	r[2]=(r[2]*pivot);
	coeff=lhs[0][2];
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[2][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[2][4]));
	c[0][0]=(c[0][0]-(coeff*c[2][0]));
	c[0][1]=(c[0][1]-(coeff*c[2][1]));
	c[0][2]=(c[0][2]-(coeff*c[2][2]));
	c[0][3]=(c[0][3]-(coeff*c[2][3]));
	c[0][4]=(c[0][4]-(coeff*c[2][4]));
	r[0]=(r[0]-(coeff*r[2]));
	coeff=lhs[1][2];
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[2][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[2][4]));
	c[1][0]=(c[1][0]-(coeff*c[2][0]));
	c[1][1]=(c[1][1]-(coeff*c[2][1]));
	c[1][2]=(c[1][2]-(coeff*c[2][2]));
	c[1][3]=(c[1][3]-(coeff*c[2][3]));
	c[1][4]=(c[1][4]-(coeff*c[2][4]));
	r[1]=(r[1]-(coeff*r[2]));
	coeff=lhs[3][2];
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[2][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[2][4]));
	c[3][0]=(c[3][0]-(coeff*c[2][0]));
	c[3][1]=(c[3][1]-(coeff*c[2][1]));
	c[3][2]=(c[3][2]-(coeff*c[2][2]));
	c[3][3]=(c[3][3]-(coeff*c[2][3]));
	c[3][4]=(c[3][4]-(coeff*c[2][4]));
	r[3]=(r[3]-(coeff*r[2]));
	coeff=lhs[4][2];
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[2][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[2][4]));
	c[4][0]=(c[4][0]-(coeff*c[2][0]));
	c[4][1]=(c[4][1]-(coeff*c[2][1]));
	c[4][2]=(c[4][2]-(coeff*c[2][2]));
	c[4][3]=(c[4][3]-(coeff*c[2][3]));
	c[4][4]=(c[4][4]-(coeff*c[2][4]));
	r[4]=(r[4]-(coeff*r[2]));
	pivot=(1.0/lhs[3][3]);
	lhs[3][4]=(lhs[3][4]*pivot);
	c[3][0]=(c[3][0]*pivot);
	c[3][1]=(c[3][1]*pivot);
	c[3][2]=(c[3][2]*pivot);
	c[3][3]=(c[3][3]*pivot);
	c[3][4]=(c[3][4]*pivot);
	r[3]=(r[3]*pivot);
	coeff=lhs[0][3];
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[3][4]));
	c[0][0]=(c[0][0]-(coeff*c[3][0]));
	c[0][1]=(c[0][1]-(coeff*c[3][1]));
	c[0][2]=(c[0][2]-(coeff*c[3][2]));
	c[0][3]=(c[0][3]-(coeff*c[3][3]));
	c[0][4]=(c[0][4]-(coeff*c[3][4]));
	r[0]=(r[0]-(coeff*r[3]));
	coeff=lhs[1][3];
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[3][4]));
	c[1][0]=(c[1][0]-(coeff*c[3][0]));
	c[1][1]=(c[1][1]-(coeff*c[3][1]));
	c[1][2]=(c[1][2]-(coeff*c[3][2]));
	c[1][3]=(c[1][3]-(coeff*c[3][3]));
	c[1][4]=(c[1][4]-(coeff*c[3][4]));
	r[1]=(r[1]-(coeff*r[3]));
	coeff=lhs[2][3];
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[3][4]));
	c[2][0]=(c[2][0]-(coeff*c[3][0]));
	c[2][1]=(c[2][1]-(coeff*c[3][1]));
	c[2][2]=(c[2][2]-(coeff*c[3][2]));
	c[2][3]=(c[2][3]-(coeff*c[3][3]));
	c[2][4]=(c[2][4]-(coeff*c[3][4]));
	r[2]=(r[2]-(coeff*r[3]));
	coeff=lhs[4][3];
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[3][4]));
	c[4][0]=(c[4][0]-(coeff*c[3][0]));
	c[4][1]=(c[4][1]-(coeff*c[3][1]));
	c[4][2]=(c[4][2]-(coeff*c[3][2]));
	c[4][3]=(c[4][3]-(coeff*c[3][3]));
	c[4][4]=(c[4][4]-(coeff*c[3][4]));
	r[4]=(r[4]-(coeff*r[3]));
	pivot=(1.0/lhs[4][4]);
	c[4][0]=(c[4][0]*pivot);
	c[4][1]=(c[4][1]*pivot);
	c[4][2]=(c[4][2]*pivot);
	c[4][3]=(c[4][3]*pivot);
	c[4][4]=(c[4][4]*pivot);
	r[4]=(r[4]*pivot);
	coeff=lhs[0][4];
	c[0][0]=(c[0][0]-(coeff*c[4][0]));
	c[0][1]=(c[0][1]-(coeff*c[4][1]));
	c[0][2]=(c[0][2]-(coeff*c[4][2]));
	c[0][3]=(c[0][3]-(coeff*c[4][3]));
	c[0][4]=(c[0][4]-(coeff*c[4][4]));
	r[0]=(r[0]-(coeff*r[4]));
	coeff=lhs[1][4];
	c[1][0]=(c[1][0]-(coeff*c[4][0]));
	c[1][1]=(c[1][1]-(coeff*c[4][1]));
	c[1][2]=(c[1][2]-(coeff*c[4][2]));
	c[1][3]=(c[1][3]-(coeff*c[4][3]));
	c[1][4]=(c[1][4]-(coeff*c[4][4]));
	r[1]=(r[1]-(coeff*r[4]));
	coeff=lhs[2][4];
	c[2][0]=(c[2][0]-(coeff*c[4][0]));
	c[2][1]=(c[2][1]-(coeff*c[4][1]));
	c[2][2]=(c[2][2]-(coeff*c[4][2]));
	c[2][3]=(c[2][3]-(coeff*c[4][3]));
	c[2][4]=(c[2][4]-(coeff*c[4][4]));
	r[2]=(r[2]-(coeff*r[4]));
	coeff=lhs[3][4];
	c[3][0]=(c[3][0]-(coeff*c[4][0]));
	c[3][1]=(c[3][1]-(coeff*c[4][1]));
	c[3][2]=(c[3][2]-(coeff*c[4][2]));
	c[3][3]=(c[3][3]-(coeff*c[4][3]));
	c[3][4]=(c[3][4]-(coeff*c[4][4]));
	r[3]=(r[3]-(coeff*r[4]));
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void binvrhs(double lhs[5][5], double r[5])
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	double pivot, coeff;
	/*
	--------------------------------------------------------------------
	c     
	c-------------------------------------------------------------------
	*/
	pivot=(1.0/lhs[0][0]);
	lhs[0][1]=(lhs[0][1]*pivot);
	lhs[0][2]=(lhs[0][2]*pivot);
	lhs[0][3]=(lhs[0][3]*pivot);
	lhs[0][4]=(lhs[0][4]*pivot);
	r[0]=(r[0]*pivot);
	coeff=lhs[1][0];
	lhs[1][1]=(lhs[1][1]-(coeff*lhs[0][1]));
	lhs[1][2]=(lhs[1][2]-(coeff*lhs[0][2]));
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[0][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[0][4]));
	r[1]=(r[1]-(coeff*r[0]));
	coeff=lhs[2][0];
	lhs[2][1]=(lhs[2][1]-(coeff*lhs[0][1]));
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[0][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[0][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[0][4]));
	r[2]=(r[2]-(coeff*r[0]));
	coeff=lhs[3][0];
	lhs[3][1]=(lhs[3][1]-(coeff*lhs[0][1]));
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[0][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[0][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[0][4]));
	r[3]=(r[3]-(coeff*r[0]));
	coeff=lhs[4][0];
	lhs[4][1]=(lhs[4][1]-(coeff*lhs[0][1]));
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[0][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[0][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[0][4]));
	r[4]=(r[4]-(coeff*r[0]));
	pivot=(1.0/lhs[1][1]);
	lhs[1][2]=(lhs[1][2]*pivot);
	lhs[1][3]=(lhs[1][3]*pivot);
	lhs[1][4]=(lhs[1][4]*pivot);
	r[1]=(r[1]*pivot);
	coeff=lhs[0][1];
	lhs[0][2]=(lhs[0][2]-(coeff*lhs[1][2]));
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[1][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[1][4]));
	r[0]=(r[0]-(coeff*r[1]));
	coeff=lhs[2][1];
	lhs[2][2]=(lhs[2][2]-(coeff*lhs[1][2]));
	lhs[2][3]=(lhs[2][3]-(coeff*lhs[1][3]));
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[1][4]));
	r[2]=(r[2]-(coeff*r[1]));
	coeff=lhs[3][1];
	lhs[3][2]=(lhs[3][2]-(coeff*lhs[1][2]));
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[1][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[1][4]));
	r[3]=(r[3]-(coeff*r[1]));
	coeff=lhs[4][1];
	lhs[4][2]=(lhs[4][2]-(coeff*lhs[1][2]));
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[1][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[1][4]));
	r[4]=(r[4]-(coeff*r[1]));
	pivot=(1.0/lhs[2][2]);
	lhs[2][3]=(lhs[2][3]*pivot);
	lhs[2][4]=(lhs[2][4]*pivot);
	r[2]=(r[2]*pivot);
	coeff=lhs[0][2];
	lhs[0][3]=(lhs[0][3]-(coeff*lhs[2][3]));
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[2][4]));
	r[0]=(r[0]-(coeff*r[2]));
	coeff=lhs[1][2];
	lhs[1][3]=(lhs[1][3]-(coeff*lhs[2][3]));
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[2][4]));
	r[1]=(r[1]-(coeff*r[2]));
	coeff=lhs[3][2];
	lhs[3][3]=(lhs[3][3]-(coeff*lhs[2][3]));
	lhs[3][4]=(lhs[3][4]-(coeff*lhs[2][4]));
	r[3]=(r[3]-(coeff*r[2]));
	coeff=lhs[4][2];
	lhs[4][3]=(lhs[4][3]-(coeff*lhs[2][3]));
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[2][4]));
	r[4]=(r[4]-(coeff*r[2]));
	pivot=(1.0/lhs[3][3]);
	lhs[3][4]=(lhs[3][4]*pivot);
	r[3]=(r[3]*pivot);
	coeff=lhs[0][3];
	lhs[0][4]=(lhs[0][4]-(coeff*lhs[3][4]));
	r[0]=(r[0]-(coeff*r[3]));
	coeff=lhs[1][3];
	lhs[1][4]=(lhs[1][4]-(coeff*lhs[3][4]));
	r[1]=(r[1]-(coeff*r[3]));
	coeff=lhs[2][3];
	lhs[2][4]=(lhs[2][4]-(coeff*lhs[3][4]));
	r[2]=(r[2]-(coeff*r[3]));
	coeff=lhs[4][3];
	lhs[4][4]=(lhs[4][4]-(coeff*lhs[3][4]));
	r[4]=(r[4]-(coeff*r[3]));
	pivot=(1.0/lhs[4][4]);
	r[4]=(r[4]*pivot);
	coeff=lhs[0][4];
	r[0]=(r[0]-(coeff*r[4]));
	coeff=lhs[1][4];
	r[1]=(r[1]-(coeff*r[4]));
	coeff=lhs[2][4];
	r[2]=(r[2]-(coeff*r[4]));
	coeff=lhs[3][4];
	r[3]=(r[3]-(coeff*r[4]));
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
	c     Performs line solves in Y direction by first factoring
	c     the block-tridiagonal matrix into an upper triangular matrix][ 
	c     and then performing back substitution to solve for the unknow
	c     vectors of each line.  
	c     
	c     Make sure we treat elements zero to cell_size in the direction
	c     of the sweep.
	c-------------------------------------------------------------------
	*/
	lhsy();
	y_solve_cell();
	y_backsubstitute();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void y_backsubstitute(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     back solve: if last cell][ then generate U(jsize)=rhs(jsize)
	c     else assume U(jsize) is loaded in un pack backsub_info
	c     so just use it
	c     after call u(jstart) will be sent to next cell
	c-------------------------------------------------------------------
	*/
	int i, j, k, m, n;
	#pragma loop name y_backsubstitute#0 
	for (j=(grid_points[1]-2); j>=0; j -- )
	{
		#pragma loop name y_backsubstitute#0#0 
		for (i=1; i<(grid_points[0]-1); i ++ )
		{
			#pragma loop name y_backsubstitute#0#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				#pragma loop name y_backsubstitute#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					#pragma loop name y_backsubstitute#0#0#0#0#0 
					for (n=0; n<5; n ++ )
					{
						rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][j+1][k][n]));
					}
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
static void y_solve_cell(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     performs guaussian elimination on this cell.
	c     
	c     assumes that unpacking routines for non-first cells 
	c     preload C' and rhs' from previous cell.
	c     
	c     assumed send happens outside this routine, but that
	c     c'(JMAX) and rhs'(JMAX) will be sent to next cell
	c-------------------------------------------------------------------
	*/
	int i, j, k, jsize;
	jsize=(grid_points[1]-1);
	#pragma loop name y_solve_cell#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name y_solve_cell#0#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			--------------------------------------------------------------------
			c     multiply c(i,0,k) by b_inverse and copy back to c
			c     multiply rhs(0) by b_inverse(0) and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvcrhs(lhs[i][0][k][1], lhs[i][0][k][2], rhs[i][0][k]);
		}
	}
	/*
	--------------------------------------------------------------------
	c     begin inner most do loop
	c     do all the elements of the cell unless last 
	c-------------------------------------------------------------------
	*/
	#pragma loop name y_solve_cell#1 
	for (j=1; j<jsize; j ++ )
	{
		#pragma loop name y_solve_cell#1#0 
		for (i=1; i<(grid_points[0]-1); i ++ )
		{
			#pragma loop name y_solve_cell#1#0#0 
			for (k=1; k<(grid_points[2]-1); k ++ )
			{
				/*
				--------------------------------------------------------------------
				c     subtract Alhs_vector(j-1) from lhs_vector(j)
				c     
				c     rhs(j) = rhs(j) - A*rhs(j-1)
				c-------------------------------------------------------------------
				*/
				matvec_sub(lhs[i][j][k][0], rhs[i][j-1][k], rhs[i][j][k]);
				/*
				--------------------------------------------------------------------
				c     B(j) = B(j) - C(j-1)A(j)
				c-------------------------------------------------------------------
				*/
				matmul_sub(lhs[i][j][k][0], lhs[i][j-1][k][2], lhs[i][j][k][1]);
				/*
				--------------------------------------------------------------------
				c     multiply c(i,j,k) by b_inverse and copy back to c
				c     multiply rhs(i,1,k) by b_inverse(i,1,k) and copy to rhs
				c-------------------------------------------------------------------
				*/
				binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
			}
		}
	}
	#pragma loop name y_solve_cell#2 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name y_solve_cell#2#0 
		for (k=1; k<(grid_points[2]-1); k ++ )
		{
			/*
			--------------------------------------------------------------------
			c     rhs(jsize) = rhs(jsize) - Arhs(jsize-1)
			c-------------------------------------------------------------------
			*/
			matvec_sub(lhs[i][jsize][k][0], rhs[i][jsize-1][k], rhs[i][jsize][k]);
			/*
			--------------------------------------------------------------------
			c     B(jsize) = B(jsize) - C(jsize-1)A(jsize)
			c     call matmul_sub(aa,i,jsize,k,c,
			c     $              cc,i,jsize-1,k,c,BB,i,jsize,k)
			c-------------------------------------------------------------------
			*/
			matmul_sub(lhs[i][jsize][k][0], lhs[i][jsize-1][k][2], lhs[i][jsize][k][1]);
			/*
			--------------------------------------------------------------------
			c     multiply rhs(jsize) by b_inverse(jsize) and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvrhs(lhs[i][jsize][k][1], rhs[i][jsize][k]);
		}
	}
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
	c     Performs line solves in Z direction by first factoring
	c     the block-tridiagonal matrix into an upper triangular matrix, 
	c     and then performing back substitution to solve for the unknow
	c     vectors of each line.  
	c     
	c     Make sure we treat elements zero to cell_size in the direction
	c     of the sweep.
	c-------------------------------------------------------------------
	*/
	lhsz();
	z_solve_cell();
	z_backsubstitute();
	return ;
}

/*
--------------------------------------------------------------------
--------------------------------------------------------------------
*/
static void z_backsubstitute(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     back solve: if last cell, then generate U(ksize)=rhs(ksize)
	c     else assume U(ksize) is loaded in un pack backsub_info
	c     so just use it
	c     after call u(kstart) will be sent to next cell
	c-------------------------------------------------------------------
	*/
	int i, j, k, m, n;
	#pragma loop name z_backsubstitute#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name z_backsubstitute#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			#pragma loop name z_backsubstitute#0#0#0 
			for (k=(grid_points[2]-2); k>=0; k -- )
			{
				#pragma loop name z_backsubstitute#0#0#0#0 
				for (m=0; m<5; m ++ )
				{
					#pragma loop name z_backsubstitute#0#0#0#0#0 
					for (n=0; n<5; n ++ )
					{
						rhs[i][j][k][m]=(rhs[i][j][k][m]-(lhs[i][j][k][2][m][n]*rhs[i][j][k+1][n]));
					}
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
static void z_solve_cell(void )
{
	/*
	--------------------------------------------------------------------
	--------------------------------------------------------------------
	*/
	/*
	--------------------------------------------------------------------
	c     performs guaussian elimination on this cell.
	c     
	c     assumes that unpacking routines for non-first cells 
	c     preload C' and rhs' from previous cell.
	c     
	c     assumed send happens outside this routine, but that
	c     c'(KMAX) and rhs'(KMAX) will be sent to next cell.
	c-------------------------------------------------------------------
	*/
	int i, j, k, ksize;
	ksize=(grid_points[2]-1);
	/*
	--------------------------------------------------------------------
	c     outer most do loops - sweeping in i direction
	c-------------------------------------------------------------------
	*/
	#pragma loop name z_solve_cell#0 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name z_solve_cell#0#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			--------------------------------------------------------------------
			c     multiply c(i,j,0) by b_inverse and copy back to c
			c     multiply rhs(0) by b_inverse(0) and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvcrhs(lhs[i][j][0][1], lhs[i][j][0][2], rhs[i][j][0]);
		}
	}
	/*
	--------------------------------------------------------------------
	c     begin inner most do loop
	c     do all the elements of the cell unless last 
	c-------------------------------------------------------------------
	*/
	#pragma loop name z_solve_cell#1 
	for (k=1; k<ksize; k ++ )
	{
		#pragma loop name z_solve_cell#1#0 
		for (i=1; i<(grid_points[0]-1); i ++ )
		{
			#pragma loop name z_solve_cell#1#0#0 
			for (j=1; j<(grid_points[1]-1); j ++ )
			{
				/*
				--------------------------------------------------------------------
				c     subtract Alhs_vector(k-1) from lhs_vector(k)
				c     
				c     rhs(k) = rhs(k) - A*rhs(k-1)
				c-------------------------------------------------------------------
				*/
				matvec_sub(lhs[i][j][k][0], rhs[i][j][k-1], rhs[i][j][k]);
				/*
				--------------------------------------------------------------------
				c     B(k) = B(k) - C(k-1)A(k)
				c     call matmul_sub(aa,i,j,k,c,cc,i,j,k-1,c,BB,i,j,k)
				c-------------------------------------------------------------------
				*/
				matmul_sub(lhs[i][j][k][0], lhs[i][j][k-1][2], lhs[i][j][k][1]);
				/*
				--------------------------------------------------------------------
				c     multiply c(i,j,k) by b_inverse and copy back to c
				c     multiply rhs(i,j,1) by b_inverse(i,j,1) and copy to rhs
				c-------------------------------------------------------------------
				*/
				binvcrhs(lhs[i][j][k][1], lhs[i][j][k][2], rhs[i][j][k]);
			}
		}
	}
	/*
	--------------------------------------------------------------------
	c     Now finish up special cases for last cell
	c-------------------------------------------------------------------
	*/
	#pragma loop name z_solve_cell#2 
	for (i=1; i<(grid_points[0]-1); i ++ )
	{
		#pragma loop name z_solve_cell#2#0 
		for (j=1; j<(grid_points[1]-1); j ++ )
		{
			/*
			--------------------------------------------------------------------
			c     rhs(ksize) = rhs(ksize) - Arhs(ksize-1)
			c-------------------------------------------------------------------
			*/
			matvec_sub(lhs[i][j][ksize][0], rhs[i][j][ksize-1], rhs[i][j][ksize]);
			/*
			--------------------------------------------------------------------
			c     B(ksize) = B(ksize) - C(ksize-1)A(ksize)
			c     call matmul_sub(aa,i,j,ksize,c,
			c     $              cc,i,j,ksize-1,c,BB,i,j,ksize)
			c-------------------------------------------------------------------
			*/
			matmul_sub(lhs[i][j][ksize][0], lhs[i][j][ksize-1][2], lhs[i][j][ksize][1]);
			/*
			--------------------------------------------------------------------
			c     multiply rhs(ksize) by b_inverse(ksize) and copy to rhs
			c-------------------------------------------------------------------
			*/
			binvrhs(lhs[i][j][ksize][1], rhs[i][j][ksize]);
		}
	}
	return ;
}
