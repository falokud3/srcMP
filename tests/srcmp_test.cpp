/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int main(int argc){

  float a[1000], b[1000];
  int i = 0; int j = 0;
  int step = i++;
  step++;
  // step, i += 3;
  

  #pragma omp parallel for
for (i = 0; i < 2; i++) {
      a[i] = a[i] * 5;
  }
  
   return 0;
}

