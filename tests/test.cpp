/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int main(int argc){

  float a[1000], b[1000];
  int i; int j;
  int step = 5;

  step, i += 3;

  switch (step)
  {
  case 5:
  case 4:
    step = 3;
    break;
  case 6:
    step = 5;
    break;
  default:
  }

  for (i = 0; i < 2 ; i++) {
      a[i] = a[i + 1];
  }
  
   return 0;
}

