/*  Non-profitable Parallel Loop:
     This loop could be parallelized but has only a small number of iterations.
     With 1000 iterations, the sequantial loop executes in less than 
      
*/

#include <stdio.h>
#include <math.h>

int main(){

  int i = 0;
  i = i + 1;
  int z = 1;

  float a[1000], b[1000];
  // int i = 0;
  // i = 3;
  // int x = 1;
  // int y = x;
  // int z = 5;
  
  for (i = 1; i<1000 ; i += z) {
    a[i]= b[i];
  }
	
   return 0;
}
