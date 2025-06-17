#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 1024
#endif

// Function to create a standard matrix with a specified percentage of zeros (value between 0-99).
// Non-zero entries are generated randomly (values between 1.0 and 10.0).
float **create_matrix(int rows, int cols, int percentZeros)
{
    float **matrix = malloc(rows * sizeof(float *));
    if (!matrix)
    {
        perror("alloc dense rows");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < rows; i++)
    {
        matrix[i] = malloc(cols * sizeof(float));
        if (!matrix[i])
        {
            perror("alloc dense cols");
            exit(EXIT_FAILURE);
        }
        for (int j = 0; j < cols; j++)
        {
            matrix[i][j] = ((rand() % 100) < percentZeros) ? 0.0f : (float)((rand() % 10) + 1);
        }
    }
    return matrix;
}

// Convert dense to CSR
void convert_dense_to_csr(float **dense, float **values, int **col_idx, int **row_ptr, int rows, int cols)
{
    int nnz = 0;
    *values = malloc(rows * cols * sizeof(float));
    *col_idx = malloc(rows * cols * sizeof(int));
    *row_ptr = malloc((rows + 1) * sizeof(int));
    if (!*values || !*col_idx || !*row_ptr)
    {
        perror("alloc CSR");
        exit(EXIT_FAILURE);
    }
    (*row_ptr)[0] = 0;
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

// Convert dense to CSC
void convert_dense_to_csc(float **dense, float **values, int **row_idx, int **col_ptr, int rows, int cols)
{
    int nnz = 0;
    *values = malloc(rows * cols * sizeof(float));
    *row_idx = malloc(rows * cols * sizeof(int));
    *col_ptr = malloc((cols + 1) * sizeof(int));
    if (!*values || !*row_idx || !*col_ptr)
    {
        perror("alloc CSC");
        exit(EXIT_FAILURE);
    }
    (*col_ptr)[0] = 0;
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

// Free functions
void free_matrix(float **M, int rows)
{
    for (int i = 0; i < rows; i++)
        free(M[i]);
    free(M);
}

void free_csr(float *values, int *col_idx, int *row_ptr)
{
    free(values);
    free(col_idx);
    free(row_ptr);
}

void free_csc(float *values, int *row_idx, int *col_ptr)
{
    free(values);
    free(row_idx);
    free(col_ptr);
}

// Standard multiply for verification
void standard_multiply(float **A, float **B, float **C, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
        for (int j = 0; j < p; j++)
            for (int k = 0; k < n; k++)
                C[i][j] += A[i][k] * B[k][j];
}

// Compare matrices
int compare_matrices(float **X, float **Y, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
        for (int j = 0; j < cols; j++)
            if (fabs(X[i][j] - Y[i][j]) > EPSILON)
                return 0;
    return 1;
}

// CUDA kernel for SpMM: One thread per row of output matrix
__global__ void spmm_csr_csc_kernel(
    const float *__restrict__ A_val, 
    const int *__restrict__ A_col, 
    const int *__restrict__ A_row,
    const float *__restrict__ Bt_val, 
    const int *__restrict__ Bt_row, 
    const int *__restrict__ Bt_col,
    float *__restrict__ C, 
    int m, int p) 
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) return;

    int start = A_row[row];
    int end = A_row[row + 1];
    
    // Initialize output row to zero
    for (int j = 0; j < p; j++) {
        C[row * p + j] = 0.0f;
    }

    // Process non-zero elements in row of A
    for (int idx = start; idx < end; idx++) {
        int colA = A_col[idx];
        float valA = A_val[idx];
        
        int bt_start = Bt_col[colA];
        int bt_end = Bt_col[colA + 1];
        
        // Accumulate contributions for each non-zero in B^T column
        for (int bt_idx = bt_start; bt_idx < bt_end; bt_idx++) {
            int colB = Bt_row[bt_idx];
            float valB = Bt_val[bt_idx];
            atomicAdd(&C[row * p + colB], valA * valB);
        }
    }
}

int main() {
    srand((unsigned)time(NULL));
    int m = SIZE, n = SIZE, p = SIZE;
    int percentZeros = 90;

    // Create dense matrices
    float **A = create_matrix(m, n, percentZeros);
    float **B = create_matrix(n, p, percentZeros);

    // Convert A to CSR
    float *A_val, *Bt_val;
    int *A_col, *A_row, *Bt_row, *Bt_col;
    convert_dense_to_csr(A, &A_val, &A_col, &A_row, m, n);

    // Create B^T and convert to CSC
    float **B_T = malloc(p * sizeof(float *));
    for (int i = 0; i < p; i++) {
        B_T[i] = malloc(n * sizeof(float));
        for (int j = 0; j < n; j++) {
            B_T[i][j] = B[j][i];
        }
    }
    convert_dense_to_csc(B_T, &Bt_val, &Bt_row, &Bt_col, p, n);
    free_matrix(B_T, p);

    // Allocate and initialize result matrix (1D for GPU)
    float *C_sparse_flat = calloc(m * p, sizeof(float));
    float **C_sparse = malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++) {
        C_sparse[i] = &C_sparse_flat[i * p];
    }

    // --- GPU Computation ---
    float *d_A_val, *d_Bt_val, *d_C;
    int *d_A_col, *d_A_row, *d_Bt_row, *d_Bt_col;
    size_t size;

    // Allocate and copy CSR (A)
    size = A_row[m] * sizeof(float);
    cudaMalloc(&d_A_val, size);
    cudaMemcpy(d_A_val, A_val, size, cudaMemcpyHostToDevice);

    size = A_row[m] * sizeof(int);
    cudaMalloc(&d_A_col, size);
    cudaMemcpy(d_A_col, A_col, size, cudaMemcpyHostToDevice);

    size = (m + 1) * sizeof(int);
    cudaMalloc(&d_A_row, size);
    cudaMemcpy(d_A_row, A_row, size, cudaMemcpyHostToDevice);

    // Allocate and copy CSC (B^T)
    int nnz_Bt = Bt_col[n];
    size = nnz_Bt * sizeof(float);
    cudaMalloc(&d_Bt_val, size);
    cudaMemcpy(d_Bt_val, Bt_val, size, cudaMemcpyHostToDevice);

    size = nnz_Bt * sizeof(int);
    cudaMalloc(&d_Bt_row, size);
    cudaMemcpy(d_Bt_row, Bt_row, size, cudaMemcpyHostToDevice);

    size = (n + 1) * sizeof(int);
    cudaMalloc(&d_Bt_col, size);
    cudaMemcpy(d_Bt_col, Bt_col, size, cudaMemcpyHostToDevice);

    // Allocate GPU output
    cudaMalloc(&d_C, m * p * sizeof(float));
    cudaMemset(d_C, 0, m * p * sizeof(float));

    // Launch kernel
    int blockSize = 256;
    int gridSize = (m + blockSize - 1) / blockSize;
    spmm_csr_csc_kernel<<<gridSize, blockSize>>>(d_A_val, d_A_col, d_A_row, 
                                                d_Bt_val, d_Bt_row, d_Bt_col, 
                                                d_C, m, p);

    // Copy result back
    cudaMemcpy(C_sparse_flat, d_C, m * p * sizeof(float), cudaMemcpyDeviceToHost);

    // --- Verification ---
    float **C_std = malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++) {
        C_std[i] = calloc(p, sizeof(float));
    }
    standard_multiply(A, B, C_std, m, n, p);

    printf("Verification: %s\n", 
           compare_matrices(C_sparse, C_std, m, p) ? "Match!" : "Mismatch!");

    // Cleanup
    cudaFree(d_A_val); cudaFree(d_A_col); cudaFree(d_A_row);
    cudaFree(d_Bt_val); cudaFree(d_Bt_row); cudaFree(d_Bt_col);
    cudaFree(d_C);
    free_csr(A_val, A_col, A_row);
    free_csc(Bt_val, Bt_row, Bt_col);
    free(C_sparse_flat); free(C_sparse);
    free_matrix(C_std, m);
    free_matrix(A, m); 
    free_matrix(B, n);

    return 0;
}