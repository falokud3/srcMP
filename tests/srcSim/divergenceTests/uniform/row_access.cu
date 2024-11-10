// https://medium.com/distributed-knowledge/cuda-memory-management-use-cases-f9d340f7c704


// not divergent/unifrom split into new folders
__global__ void coalesced_access_kernel(float* m, int dimx, int dimy, int sz, float mul) {
  // Calculate matrix cell coordinate based on thread indexes
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = row * dimx + col;
  
  // Check out of bound condition
  for (; idx < sz; idx += blockDim.x * gridDim.x) {
    m[idx] *= mul;
  }
}