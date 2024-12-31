#include <iostream>
#include <cuda_runtime.h>

#define BLOCK_SIZE 256
#define ARRAY_SIZE 1024

__global__ void uncoalescedMemoryAccessKernel(int *d_in, int *d_out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        int offset = (idx % 4) * 256;
        d_out[idx] = d_in[offset];
    }
}

void uncoalescedMemoryAccess(int *h_in, int *h_out, int n) {
    int *d_in, *d_out;
    size_t bytes = n * sizeof(int);

    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);

    cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice);

    int blockSize = BLOCK_SIZE;
    int gridSize = (n + blockSize - 1) / blockSize;

    uncoalescedMemoryAccessKernel<<<gridSize, blockSize>>>(d_in, d_out, n);

    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
}

int main() {
    const int N = ARRAY_SIZE;
    int h_in[N], h_out[N];

    for (int i = 0; i < N; i++) {
        h_in[i] = i;
    }

    uncoalescedMemoryAccess(h_in, h_out, N);

    for (int i = 0; i < N; i++) {
        std::cout << h_out[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}
