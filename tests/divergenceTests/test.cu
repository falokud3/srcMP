__global__ void modifiedDiffusionCoeffKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // if (idx >= size) return;

    const float den = array[idx] * array[idx];
    const float c = 0;
    if (true) {
        c = 1;
    } else {
        c = 2;
    }

    if (c < 0) {
        array[idx] = 0;
    } else if (c > 1) {
        array[idx] = 1;
    } else {
        array[idx] = c;
    }
    
}