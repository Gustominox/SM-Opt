#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <iostream>
#include <vector>
#include <random>

#include <sycl/sycl.hpp>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 4096 // Matrix dimension set to 4096 as requested for the largest dataset
#endif

// Function to create a standard dense matrix with a specified percentage of zeros.
// Non-zero entries are generated randomly (values between 1.0 and 10.0).
// Uses sycl::malloc_shared for USM allocation.
float **create_matrix(int rows, int cols, int percentZeros, sycl::queue &q)
{
    float **matrix = sycl::malloc_shared<float*>(rows, q);
    if (!matrix)
    {
        perror("Failed to allocate dense rows (USM)");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = sycl::malloc_shared<float>(cols, q);
        if (!matrix[i])
        {
            perror("Failed to allocate dense cols (USM)");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = ((rand() % 100) < percentZeros)? 0.0f : (float)((rand() % 10) + 1);
        }
    }
    return matrix;
}

// Convert dense matrix to Compressed Sparse Column (CSC) format.
// Uses sycl::malloc_shared for USM allocation.
void convert_dense_to_csc(float **dense, float **values, int **row_idx, int **col_ptr, int rows, int cols, sycl::queue &q)
{
    int nnz = 0;
    // Allocate maximum possible size for values and row_idx (rows*cols)
    *values = sycl::malloc_shared<float>(rows * cols, q);
    *row_idx = sycl::malloc_shared<int>(rows * cols, q);
    *col_ptr = sycl::malloc_shared<int>((cols + 1), q); // col_ptr size is cols + 1
    if (!*values ||!*row_idx ||!*col_ptr)
    {
        perror("Failed to allocate CSC arrays (USM)");
        exit(EXIT_FAILURE);
    }

    (*col_ptr)[0] = 0; // Fix: Initialize first element to 0
    for (int j = 0; j < cols; j++)
    {
        (*col_ptr)[j + 1] = (*col_ptr)[j];
        for (int i = 0; i < rows; i++)
        {
            float v = dense[i][j];
            if (fabs(v) > EPSILON)
            {
                (*values)[nnz] = v;
                (*row_idx)[nnz] = i;
                nnz++;
                (*col_ptr)[j + 1]++;
            }
        }
    }
}

// Convert dense matrix to Compressed Sparse Row (CSR) format.
// Uses sycl::malloc_shared for USM allocation.
void convert_dense_to_csr(float **dense, float **values, int **col_idx, int **row_ptr, int rows, int cols, sycl::queue &q)
{
    int nnz = 0;
    // Allocate maximum possible size for values and col_idx (rows*cols)
    *values = sycl::malloc_shared<float>(rows * cols, q);
    *col_idx = sycl::malloc_shared<int>(rows * cols, q);
    *row_ptr = sycl::malloc_shared<int>((rows + 1), q); // row_ptr size is rows + 1
    if (!*values ||!*col_idx ||!*row_ptr)
    {
        perror("Failed to allocate CSR arrays (USM)");
        exit(EXIT_FAILURE);
    }

    (*row_ptr)[0] = 0; // Fix: Initialize first element to 0
    for (int i = 0; i < rows; i++)
    {
        (*row_ptr)[i + 1] = (*row_ptr)[i];
        for (int j = 0; j < cols; j++)
        {
            float v = dense[i][j];
            if (fabs(v) > EPSILON)
            {
                (*values)[nnz] = v;
                (*col_idx)[nnz] = j;
                nnz++;
                (*row_ptr)[i + 1]++;
            }
        }
    }
}

// Free functions for USM allocated memory.
void free_matrix(float **M, int rows, sycl::queue &q)
{
    for (int i = 0; i < rows; i++)
        sycl::free(M[i], q);
    sycl::free(M, q);
}

void free_csr(float *values, int *col_idx, int *row_ptr, sycl::queue &q)
{
    sycl::free(values, q);
    sycl::free(col_idx, q);
    sycl::free(row_ptr, q);
}

void free_csc(float *values, int *row_idx, int *col_ptr, sycl::queue &q)
{
    sycl::free(values, q);
    sycl::free(row_idx, q);
    sycl::free(col_ptr, q);
}

// Standard dense matrix multiplication for verification purposes.
// Operates on USM pointers, but execution is on the host CPU.
void standard_multiply(float **A, float **B, float **C, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// Compares two dense matrices for equality within a given epsilon.
// Operates on USM pointers, but execution is on the host CPU.
int compare_matrices(float **X, float **Y, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (fabs(X[i][j] - Y[i][j]) > EPSILON)
                return 0;
    return 1;
}

// Heterogeneous Sparse Matrix-Matrix Multiplication (SpMM).
// A is in CSR format, B^T (transpose of B) is in CSC format. Result C is dense.
// This function orchestrates the SYCL kernels to run simultaneously on CPU and GPU.
void sparse_multiply_csr_csc_hetero(
    float *A_val, int *A_col, int *A_row,
    float *Bt_val, int *Bt_row, int *Bt_col,
    float **C, int m, int n, int p)
{
    // Create SYCL queues for CPU and GPU devices.
    // Error handling for device selection is recommended in production code.
    sycl::queue q_cpu{sycl::cpu_selector_v};
    sycl::queue q_gpu{sycl::gpu_selector_v};

    std::cout << "Using CPU device: " << q_cpu.get_device().get_info<sycl::info::device::name>() << std::endl;
    std::cout << "Using GPU device: " << q_gpu.get_device().get_info<sycl::info::device::name>() << std::endl;


    // Simple static row-wise partitioning: half rows to CPU, half to GPU.
    // For sparse matrices, a more sophisticated load balancing strategy (e.g., dynamic scheduling
    // or profiling-guided static partitioning based on non-zero elements per row)
    // would be crucial for optimal performance and scalability.
    size_t rows_cpu = m / 2;
    size_t rows_gpu = m - rows_cpu;

    // Submit the CPU portion of the workload.
    // Each work-item computes one row of the result matrix C.
    auto e_cpu = q_cpu.submit([&](sycl::handler& h) {
        h.parallel_for<class HeteroCPU>(sycl::range{rows_cpu}, [=](sycl::id<1> idx) {
            int i = idx[0]; // Fix: Extract index from sycl::id<1>
            // Iterate over non-zero elements in row 'i' of matrix A
            for (int a = A_row[i]; a < A_row[i + 1]; a++)
            {
                int k = A_col[a]; // Column index in A, which is also the row index in B
                float vA = A_val[a]; // Value from A[i][k]
                // Traverse column 'k' of B^T (which corresponds to row 'k' of B)
                for (int b = Bt_col[k]; b < Bt_col[k + 1]; b++)
                {
                    int j = Bt_row[b]; // Row index in B^T, which is the column index in B (and C)
                    // Accumulate the product into the dense result matrix C[i][j].
                    // C[i][j] is directly accessible via USM pointer.
                    C[i][j] += vA * Bt_val[b];
                }
            }
        });
    });

    // Submit the GPU portion of the workload.
    // Each work-item computes one row of the result matrix C.
    auto e_gpu = q_gpu.submit([&](sycl::handler& h) {
        h.parallel_for<class HeteroGPU>(sycl::range{rows_gpu}, [=](sycl::id<1> idx) {
            // Row index for the GPU's assigned portion, with an offset to cover the remaining rows.
            int i = idx[0] + rows_cpu; // Fix: Extract index from sycl::id<1>
            // Iterate over non-zero elements in row 'i' of matrix A
            for (int a = A_row[i]; a < A_row[i + 1]; a++)
            {
                int k = A_col[a]; // Column index in A, which is also the row index in B
                float vA = A_val[a]; // Value from A[i][k]
                // Traverse column 'k' of B^T (which corresponds to row 'k' of B)
                for (int b = Bt_col[k]; b < Bt_col[k + 1]; b++)
                {
                    int j = Bt_row[b]; // Row index in B^T, which is the column index in B (and C)
                    // Accumulate the product into the dense result matrix C[i][j].
                    // C[i][j] is directly accessible via USM pointer.
                    C[i][j] += vA * Bt_val[b];
                }
            }
        });
    });

    // Wait for both CPU and GPU kernels to complete their execution.
    e_cpu.wait();
    e_gpu.wait();
}

int main()
{
    srand((unsigned)time(NULL)); // Seed random number generator
    int m = SIZE, n = SIZE, p = SIZE;
    int percentZeros = 90; // 90% sparsity

    // Create a SYCL queue for initial USM allocations.
    // This queue provides the context for sycl::malloc_shared.
    // Using default_selector_v will pick a suitable device (often GPU if available, else CPU).
    sycl::queue q_alloc{sycl::default_selector_v};
    std::cout << "Memory allocation context device: " << q_alloc.get_device().get_info<sycl::info::device::name>() << std::endl;

    // Create dense matrices A (m x n) and B (n x p) using USM.
    float **A = create_matrix(m, n, percentZeros, q_alloc);
    float **B = create_matrix(n, p, percentZeros, q_alloc);

    // Convert dense matrix A to CSR format using USM.
    float *A_val;
    int *A_col, *A_row;
    convert_dense_to_csr(A, &A_val, &A_col, &A_row, m, n, q_alloc);

    // Transpose dense matrix B into B_T (p x n) using USM.
    float **B_T = sycl::malloc_shared<float*>(p, q_alloc);
    if (!B_T) { perror("Failed to allocate B_T rows (USM)"); exit(EXIT_FAILURE); }
    for (int i = 0; i < p; i++)
    {
        B_T[i] = sycl::malloc_shared<float>(n, q_alloc);
        if (!B_T[i]) { perror("Failed to allocate B_T cols (USM)"); exit(EXIT_FAILURE); }
        for (int j = 0; j < n; j++)
        {
            B_T[i][j] = B[j][i];
        }
    }
    // Convert transposed dense matrix B_T to CSC format using USM.
    float *Bt_val;
    int *Bt_row, *Bt_col;
    convert_dense_to_csc(B_T, &Bt_val, &Bt_row, &Bt_col, p, n, q_alloc);
    free_matrix(B_T, p, q_alloc); // Free the temporary transposed dense matrix

    // Allocate the result matrix C_sparse (dense) using USM and initialize to zero.
    float **C_sparse = sycl::malloc_shared<float*>(m, q_alloc);
    if (!C_sparse) { perror("Failed to allocate C_sparse rows (USM)"); exit(EXIT_FAILURE); }
    for (int i = 0; i < m; i++)
    {
        C_sparse[i] = sycl::malloc_shared<float>(p, q_alloc);
        if (!C_sparse[i]) { perror("Failed to allocate C_sparse cols (USM)"); exit(EXIT_FAILURE); }
        memset(C_sparse[i], 0, p * sizeof(float)); // Initialize to zero
    }

    // Perform the heterogeneous sparse matrix multiplication.
    sparse_multiply_csr_csc_hetero(A_val, A_col, A_row,
                                   Bt_val, Bt_row, Bt_col,
                                   C_sparse, m, n, p);

    // Allocate a second result matrix C_std for verification using standard dense multiplication.
    float **C_std = sycl::malloc_shared<float*>(m, q_alloc);
    if (!C_std) { perror("Failed to allocate C_std rows (USM)"); exit(EXIT_FAILURE); }
    for (int i = 0; i < m; i++)
    {
        C_std[i] = sycl::malloc_shared<float>(p, q_alloc);
        if (!C_std[i]) { perror("Failed to allocate C_std cols (USM)"); exit(EXIT_FAILURE); }
        memset(C_std[i], 0, p * sizeof(float)); // Initialize to zero
    }
    // Perform standard dense matrix multiplication on the host for verification.
    standard_multiply(A, B, C_std, m, n, p);

    // Compare the results from sparse and standard multiplication.
    printf("Verification: %s\n",
           compare_matrices(C_sparse, C_std, m, p)? "Match!" : "Mismatch!");

    // Cleanup all USM allocated memory.
    free_csr(A_val, A_col, A_row, q_alloc);
    free_csc(Bt_val, Bt_row, Bt_col, q_alloc);
    free_matrix(C_sparse, m, q_alloc);
    free_matrix(C_std, m, q_alloc);
    free_matrix(A, m, q_alloc);
    free_matrix(B, n, q_alloc);

    return 0;
}