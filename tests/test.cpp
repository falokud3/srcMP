/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int main(){

  int x[] = {0, 1, 2};

  for (int i = 0; i < 3; i++) {
    x[i] = x[2];
    x[1] = x[i];
  }
  
  return 5;
}
