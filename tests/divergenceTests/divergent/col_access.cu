// https://medium.com/distributed-knowledge/cuda-memory-management-use-cases-f9d340f7c704

__global__ void uncoalesced_access_kernel(float* m, int dimx, int sz, int mul) {
  // Calculate matrix cell coordinate based on thread indexes
  int col = blockDim.x * blockIdx.x + threadIdx.x;
  int row = blockDim.y * blockIdx.y + threadIdx.y;
  int idx = col * dimx + row;
  
  for (; idx < sz; idx += blockDim.x * gridDim.x) {
    m[idx] *= mul;
  }
}