// generated using chatGPT

#include <stdio.h>

// CUDA kernel to initialize an array and perform additional computation
__global__ void initializeAndComputeArray(float *array, float initialValue, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Initialize the array element
        array[idx] = initialValue;

        // Perform additional computation: compute the square of the value
        array[idx] = array[idx] * array[idx];
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

    // Initial uniform value
    float initialValue = 3.14f;

    // Number of threads in a block
    int blockSize = 256;

    // Number of blocks in a grid
    int gridSize = (size + blockSize - 1) / blockSize;

    // Launch the kernel
    initializeAndComputeArray<<<gridSize, blockSize>>>(d_array, initialValue, size);

    // Copy data back to host
    cudaMemcpy(h_array, d_array, bytes, cudaMemcpyDeviceToHost);

    // Check the results
    for (int i = 0; i < size; i++) {
        float expectedValue = initialValue * initialValue;
        if (h_array[i] != expectedValue) {
            printf("Error at index %d: Expected %f but got %f\n", i, expectedValue, h_array[i]);
        }
    }

    printf("Array initialized and computed with the square of %f\n", initialValue);

    // Free memory
    cudaFree(d_array);
    free(h_array);

    return 0;
}
