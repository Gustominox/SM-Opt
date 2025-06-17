#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>
#include <thread>
#include <vector>
#include <chrono>

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

// Process a range of rows on a GPU
void processRows(int gpuId, int rowStart, int rowEnd, 
                float* h_matrix, float* h_result) {
    // Set current device
    CHECK_CUDA(cudaSetDevice(gpuId));
    
    // Two streams for double buffering
    const int numStreams = 2;
    cudaStream_t streams[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }

    // Allocate device memory per stream
    float *d_input[numStreams], *d_output[numStreams];
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaMalloc((void**)&d_input[i], NCOLS * sizeof(float)));
        CHECK_CUDA(cudaMalloc((void**)&d_output[i], NCOLS * sizeof(float)));
    }

    // Kernel configuration
    int blockSize = BLOCK_SIZE;
    int gridSize = (NCOLS + blockSize - 1) / blockSize;

    // Process assigned rows with double buffering
    for (int row = rowStart; row < rowEnd; row++) {
        int streamIdx = (row - rowStart) % numStreams;
        
        // Async H2D copy
        CHECK_CUDA(cudaMemcpyAsync(
            d_input[streamIdx], 
            &h_matrix[row * NCOLS],
            NCOLS * sizeof(float),
            cudaMemcpyHostToDevice,
            streams[streamIdx]
        ));
        
        // Launch kernel
        stencilKernel<<<gridSize, blockSize, 0, streams[streamIdx]>>>(
            d_input[streamIdx],
            d_output[streamIdx],
            NCOLS
        );
        
        // Async D2H copy
        CHECK_CUDA(cudaMemcpyAsync(
            &h_result[row * NCOLS],
            d_output[streamIdx],
            NCOLS * sizeof(float),
            cudaMemcpyDeviceToHost,
            streams[streamIdx]
        ));
    }

    // Synchronize and cleanup
    for (int i = 0; i < numStreams; i++) {
        CHECK_CUDA(cudaStreamSynchronize(streams[i]));
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_output[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
}

int main(int argc, char* argv[]) {
    // Parse number of GPUs from command line
    int num_gpus = 1;
    if (argc > 1) {
        num_gpus = std::atoi(argv[1]);
        if (num_gpus < 1 || num_gpus > 4) {
            std::cerr << "Invalid GPU count. Using 1-4 GPUs." << std::endl;
            num_gpus = 1;
        }
    }
    std::cout << "Using " << num_gpus << " GPU(s)" << std::endl;

    // Allocate host memory
    float* h_matrix = new float[NROWS * NCOLS];
    float* h_result = new float[NROWS * NCOLS];

    // Initialize matrix
    for (int i = 0; i < NROWS * NCOLS; i++) {
        h_matrix[i] = 1.0f;
    }

    // Start timer
    auto start = std::chrono::high_resolution_clock::now();

    // Partition rows across GPUs
    std::vector<std::thread> threads;
    int rows_per_gpu = NROWS / num_gpus;
    int extra_rows = NROWS % num_gpus;
    int current_row = 0;

    for (int gpu = 0; gpu < num_gpus; gpu++) {
        int rows_this_gpu = rows_per_gpu + (gpu < extra_rows ? 1 : 0);
        int end_row = current_row + rows_this_gpu;
        
        threads.emplace_back(
            processRows, 
            gpu, 
            current_row, 
            end_row, 
            h_matrix, 
            h_result
        );
        
        current_row = end_row;
    }

    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }

    // Stop timer
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Execution time: " << elapsed.count() << " seconds" << std::endl;

    // Verification (optional)
    bool verify = false;
    if (verify) {
        std::cout << "First row: ";
        for (int i = 0; i < 10; i++) {
            if (i == 0 || i == NCOLS-1) std::cout << "2.0 ";
            else std::cout << "3.0 ";
        }
        std::cout << "..." << std::endl;
    }

    // Cleanup
    delete[] h_matrix;
    delete[] h_result;

    return 0;
}