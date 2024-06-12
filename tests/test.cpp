/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int main(int argc) {

  float a[1000], b[1000];
  
  for (int i = 0; i<1000 ; i++) {
    a[i] = a[i + 1];
  }
   return 0;
}
