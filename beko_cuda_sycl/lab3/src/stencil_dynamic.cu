#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define NROWS 1024        
#define NCOLS 1024        
#define STENCIL_RADIUS 1  

#define BLOCK_SIZE 256

// Macro for checking CUDA errors.
#define CHECK_CUDA(call) {                                    \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error in " << __FILE__            \
                  << " at line " << __LINE__ << ": "          \
                  << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}

// Child kernel: processes one row using a 3-point stencil
__global__ void stencilKernel(const float* input, float* output, int ncols) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < ncols) {
        float sum = 0.0f;
        // Apply stencil to current element and its neighbors
        for (int r = -STENCIL_RADIUS; r <= STENCIL_RADIUS; r++) {
            int neighbor_col = col + r;
            // Check boundary conditions
            if (neighbor_col >= 0 && neighbor_col < ncols) {
                sum += input[neighbor_col];
            }
        }
        output[col] = sum;
    }
}

// Parent kernel: each thread handles one row and launches a child kernel
__global__ void parentKernel(const float* d_input, float* d_output, int nrows, int ncols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < nrows) {
        // TODO: configure child kernel launch parameters.
        int blockSizeChild = BLOCK_SIZE;
        int gridSizeChild = (ncols + blockSizeChild - 1) / blockSizeChild;

        // TODO: launch the child kernel to process a row
        stencilKernel<<<gridSizeChild, blockSizeChild>>>(
            &d_input[row * ncols], 
            &d_output[row * ncols], 
            ncols
        );

        // Wait for the child kernel to finish.
        cudaDeviceSynchronize();
    }
}

int main() {
    size_t matrixSize = NROWS * NCOLS * sizeof(float);
    float* h_input  = new float[NROWS * NCOLS];
    float* h_output = new float[NROWS * NCOLS];

    for (int i = 0; i < NROWS * NCOLS; i++) {
        h_input[i] = 1.0f;
    }

    // TODO: allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc((void**)&d_input, matrixSize));
    CHECK_CUDA(cudaMalloc((void**)&d_output, matrixSize));

    // TODO: copy input matrix from host to device
    CHECK_CUDA(cudaMemcpy(d_input, h_input, matrixSize, cudaMemcpyHostToDevice));

    // Launch the parent kernel. Each thread in this kernel will launch a child kernel
    // to process one row
    int blockSizeParent = BLOCK_SIZE;
    int gridSizeParent = (NROWS + blockSizeParent - 1) / blockSizeParent;
    parentKernel<<<gridSizeParent, blockSizeParent>>>(d_input, d_output, NROWS, NCOLS);

    // TODO: copy the processed matrix back to host memory
    CHECK_CUDA(cudaMemcpy(h_output, d_output, matrixSize, cudaMemcpyDeviceToHost));
    
    std::cout << "First row of result: ";
    for (int i = 0; i < NCOLS; i++) std::cout << h_output[i] << " ";
    std::cout << std::endl;

    // Clean up 
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));

    return 0;
}
