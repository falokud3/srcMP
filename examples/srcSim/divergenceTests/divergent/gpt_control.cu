// generated using chatGPT

#include <stdio.h>


// CUDA kernel with divergent branching
__global__ void divergentKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (idx % 2 == 0) {
            // If index is even, perform one computation
            array[idx] = idx * 2.0f;
        } else {
            // If index is odd, perform a different computation
            array[idx] = idx * 3.0f + 1.0f;
        }
    }
}

int main() {
    // Array size
    int size = 1024;
    int bytes = size * sizeof(float);

    // Host array
    float *h_array = (float *)malloc(bytes);

    // Device array
    float *d_array;
    cudaMalloc(&d_array, bytes);

    // Number of threads in a block
    int blockSize = 256;

    // Number of blocks in a grid
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the divergent kernel
    divergentKernel<<<gridSize, blockSize>>>(d_array, size);

    // Copy data back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    // Check the results
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

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
