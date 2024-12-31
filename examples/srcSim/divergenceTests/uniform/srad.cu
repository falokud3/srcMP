// Modified version of the diffusion coefficient calc in srad
// Inspired by paper GPUCheck: Detecting CUDA Thread Divergence with Static Analysis

#include <stdio.h>

// CUDA kernel with m branching
__global__ void modifiedDiffusionCoeffKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    float den = array[idx] * array[idx];
    float c = min(max(1.0f / (1.0f + den),0.0f),1.0f)
    if (c < 00000000.1) {
        array[idx] = 0;
    } else {
        array[idx] = c;
    }
    
}

int main() {
    int size = 1024;
    int bytes = size * sizeof(float);

    float *h_array = (float *)malloc(bytes);

    float *d_array;
    cudaMalloc(&d_array, bytes);

    int blockSize = 256;

    int gridSize = (size + blockSize - 1) / blockSize;

    modifiedDiffusionCoeffKernel<<<gridSize, blockSize>>>(d_array, size);

    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++) {
        if (i % 2 == 0) {
            float expectedValue = i * 2.0f;
            if (h_array[i] != expectedValue) {
                printf("Error at index %d: Expected %f but got %f\n", i, expectedValue, h_array[i]);
            }
        } else {
            float expectedValue = i * 3.0f + 1.0f;
            if (h_array[i] != expectedValue) {
                printf("Error at index %d: Expected %f but got %f\n", i, expectedValue, h_array[i]);
            }
        }
    }

    printf("Divergent kernel computation complete.\n");


    cudaFree(d_array);
    free(h_array);

    return 0;
}
