__global__ void modifiedDiffusionCoeffKernel(float *array, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx >= size) return;

    const float den = array[idx] * array[idx];
    const float c = 0;
    if (true) {
        if (true) {
            c = 2;
        }
        c = 1;
    } else {
        c = 2;
    }

    if (c < 0) {
        array[idx] = 0;
    } else if (c > 1) {
        array[modifiedDiffusionCoeffKernel(idx)] = 1;
    } else {
        array[idx / 5] = c;
    }
    
}