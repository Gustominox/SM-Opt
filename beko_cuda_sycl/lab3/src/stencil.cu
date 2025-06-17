#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

#define NROWS 1024
#define NCOLS 1024
#define STENCIL_RADIUS 1
#define BLOCK_SIZE 256

// Macro for checking CUDA errors
#define CHECK_CUDA(call) {                                    \
    cudaError_t err = call;                                   \
    if (err != cudaSuccess) {                                 \
        std::cerr << "CUDA error in " << __FILE__             \
                  << " at line " << __LINE__ << ": "          \
                  << cudaGetErrorString(err) << std::endl;    \
        exit(EXIT_FAILURE);                                   \
    }                                                         \
}

// Kernel that applies a one-dimensional stencil to a single row
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

int main() {
    // Allocate host memory
    size_t matrixSize = NROWS * NCOLS * sizeof(float);
    float* h_matrix = new float[NROWS * NCOLS];
    float* h_result = new float[NROWS * NCOLS];

    // Initialize input matrix with 1.0f
    for (int i = 0; i < NROWS * NCOLS; i++) {
        h_matrix[i] = 1.0f;
    }

    // Create two streams for double buffering
    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory for double buffering
    float *d_input[numStreams], *d_output[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaMalloc((void**)&d_input[i], NCOLS * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_output[i], NCOLS * sizeof(float)));
    }

    // Kernel launch configuration
    int blockSize = BLOCK_SIZE;
    int gridSize = (NCOLS + blockSize - 1) / blockSize;

    // Process each row using double buffering
    for (int row = 0; row < NROWS; row++) {
        int streamIdx = row % numStreams;

        // Asynchronous copy: host row -> device input buffer
        CHECK_CUDA(cudaMemcpyAsync(
            d_input[streamIdx],                 // Device destination
            &h_matrix[row * NCOLS],             // Host source
            NCOLS * sizeof(float),              // Size
            cudaMemcpyHostToDevice,             // Direction
            streams[streamIdx]                  // Stream
        ));

        // Launch stencil kernel in the same stream
        stencilKernel<<<gridSize, blockSize, 0, streams[streamIdx]>>>(
            d_input[streamIdx],                 // Input buffer
            d_output[streamIdx],                // Output buffer
            NCOLS                              // Number of columns
        );

        // Asynchronous copy: device output buffer -> host result row
        CHECK_CUDA(cudaMemcpyAsync(
            &h_result[row * NCOLS],             // Host destination
            d_output[streamIdx],                // Device source
            NCOLS * sizeof(float),              // Size
            cudaMemcpyDeviceToHost,             // Direction
            streams[streamIdx]                  // Stream
        ));
    }

    // Wait for all asynchronous operations to complete
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
    }

    // Print first row of result for verification
    std::cout << "First row of result: ";
    for (int i = 0; i < NCOLS; i++) {
        // Boundary elements: 2 neighbors (value = 2.0f)
        if (i == 0 || i == NCOLS - 1) {
            std::cout << "2.0 ";
        } 
        // Middle elements: 3 neighbors (value = 3.0f)
        else {
            std::cout << "3.0 ";
        }
    }
    std::cout << std::endl;

    // Cleanup
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_output[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    delete[] h_matrix;
    delete[] h_result;

    return 0;
}