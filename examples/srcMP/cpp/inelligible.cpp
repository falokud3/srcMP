/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
    Based on cetus TooSmall.c
*/

#include <stdio.h>
#include <math.h>

int main(){

  float a[1000], b[1000];
  
  for (int i=1; i<1000; i++) {
    a[i + 1]= a[i];
    continue;
  }
	
   return 0;
}
