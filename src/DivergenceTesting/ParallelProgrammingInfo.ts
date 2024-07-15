

export type SupportedLanguages = 'cu' | 'hip' | 'cl';

/**
 * The built in variables for the CUDA programming model
 * @see https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#built-in-variables
 * for details
 */
export const cudaBuiltIns = [
    'gridDim', 'blockIdx', 'blockDim', 'threadIdx', 'warpSize'
]

/**
 * Built in variables for the HIP programming model
 * @see https://github.com/ROCm/HIP/blob/develop/docs/reference/kernel_language.rst
 * for details
 */
export const hipBuiltIns =  [
    'gridDim', 'blockIdx', 'blockDim', 'threadIdx', 'warpSize'
]

/**
 * Built in functions for OpenCL
 * @see https://www.khronos.org/files/opencl30-reference-guide.pdf
 * for details
 */
export const openclBuiltIns = [
    'get_global_id', 'get_group_id', 'get_local_id', 'get_global_linear_id', ' get_local_linear_id'
]

