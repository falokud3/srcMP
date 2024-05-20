/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int z = 4;

int main(){

  int x;

  if (true) {
    x = 5;
    x = x + 1;
    // 3 = 2
  } else {
    x = 4;
    x = 0;
  }
  
  return x;
}

