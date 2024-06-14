#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1024

__global__ void coalescedMemoryAccessKernel(int *d_in, int *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Access contiguous memory locations for coalesced access
    if (idx < n) {
        d_out[idx] = d_in[idx];
    }
}

void coalescedMemoryAccess(int *h_in, int *h_out, int n) {
    int *d_in, *d_out;
    size_t bytes = n * sizeof(int);

    // Allocate device memory
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    // Copy data to device
    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    // Launch the kernel
    coalescedMemoryAccessKernel<<<gridSize, blockSize>>>(d_in, d_out, n);

    // Copy results back to host
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    const int N = ARRAY_SIZE;
    int h_in[N], h_out[N];

    // Initialize input array with some values
    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    // Perform memory access with coalesced patterns
    coalescedMemoryAccess(h_in, h_out, N);

    // Print the results
    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
