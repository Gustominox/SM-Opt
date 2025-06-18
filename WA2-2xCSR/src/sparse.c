#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 1024
#endif

// Function to print a dense matrix
void print_dense_matrix(float **matrix, int rows, int cols)
{
    for (int i = 0; i < rows; i++) // Iterate over rows
    {
        for (int j = 0; j < cols; j++) // Iterate over columns
        {
            printf("%.2f,", matrix[i][j]); // Print each element with 2 decimal places
        }
        printf("\n"); // Newline after each row
    }
}

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

// Sparse multiply: A in CSR, B^T in CSC -> C dense
void sparse_multiply_csr_csc(
    float *A_val, int *A_col, int *A_row,
    float *Bt_val, int *Bt_row, int *Bt_col,
    float **C, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int a = A_row[i]; a < A_row[i + 1]; a++)
        {
            int k = A_col[a]; // column index in A / row index in B
            float vA = A_val[a];
            // traverse column k of B^T => row k of B
            for (int b = Bt_col[k]; b < Bt_col[k + 1]; b++)
            {
                int j = Bt_row[b]; // column index j in B
                C[i][j] += vA * Bt_val[b];
            }
        }
    }
}

// Sparse multiply: A in CSR, B^T in CSR
void sparse_multiply_csr_csr(
    float *A_val, int *A_col, int *A_row,
    float *Bt_val, int *Bt_row, int *Bt_col,
    float **C, int m)
{
    for (int i = 0; i < m; i++)
    {
        for (int a = A_row[i]; a < A_row[i + 1]; a++)
        {
            int k = A_col[a];
            float vA = A_val[a];

            for (int b = Bt_row[k]; b < Bt_row[k + 1]; b++)
            {
                int j = Bt_col[b];
                // printf("Multiplying A[%d][%d] * B[%d][%d] => %.2f * %.2f\n", i, k, b, j, vA, Bt_val[b]);

                C[i][j] += vA * Bt_val[b];
            }
        }
    }
}

// Sparse multiply: A in CSR, B^T in CSC -> C dense
void sparse_multiply_csr_csc_2(
    float *A_val, int *A_col, int *A_row,
    float *B_val, int *B_row, int *B_col,
    float **C, int m)
{

    for (int i = 0; i < m; i++) // Iterate over rows of A
    {
        for (int a = A_row[i]; a < A_row[i + 1]; a++) // Traverse non-zero elements of row i in A
        {
            int k = A_col[a];
            float vA = A_val[a];

            // Traverse column k of B
            for (int b = B_col[i]; b < B_col[i + 1]; b++) // Traverse non-zero elements of column k in B
            {
                // int j = B_row[k]; // Row index in B (column index in C)
                printf("Multiplying A[%d][%d] * B[%d][%d] => %.2f * %.2f\n", i, k, k, b, vA, B_val[b]);

                C[i][k] += vA * B_val[b]; // Multiply and accumulate the result in C
            }
        }
    }
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

int main()
{
    srand((unsigned)time(NULL));
    // srand(1024);
    int m = SIZE, n = SIZE, p = SIZE;
    int percentZeros = 90;

    // Create dense matrices A (m x n) and B (n x p)
    float **A = create_matrix(m, n, percentZeros);
    float **B = create_matrix(n, p, percentZeros);

    // printf("Matrix A\n");
    // print_dense_matrix(A, m, n);

    // Convert A to CSR
    float *A_val;
    int *A_col, *A_row;
    convert_dense_to_csr(A, &A_val, &A_col, &A_row, m, n);

    // // Transpose B into B_T (p x n)
    // float **B_T = malloc(p * sizeof(float *));
    // for (int i = 0; i < p; i++)
    // {
    //     B_T[i] = malloc(n * sizeof(float));
    //     for (int j = 0; j < n; j++)
    //     {
    //         B_T[i][j] = B[j][i];
    //     }
    // }

    // printf("Matrix B\n");
    // print_dense_matrix(B, n, p);

    // // Convert B_T to CSC => Bt_val, Bt_row, Bt_col
    // float *Bt_val;
    // int *Bt_row, *Bt_col;
    // convert_dense_to_csr(B_T, &Bt_val, &Bt_col, &Bt_row, p, n);
    // free_matrix(B_T, p);

    float *B_val;
    int *B_row, *B_col;
    convert_dense_to_csr(B, &B_val, &B_col, &B_row, p, n);

    // Allocate result matrix C_sparse
    float **C_sparse = malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++)
    {
        C_sparse[i] = calloc(p, sizeof(float));
    }

    // Perform sparse multiplication
    sparse_multiply_csr_csr(A_val, A_col, A_row,
                            B_val, B_row, B_col,
                            C_sparse, m);

    // sparse_multiply_csr_csc_2(A_val, A_col, A_row,
    //                           B_val, B_row, B_col,
    //                           C_sparse, m);

    // Verification with standard multiply
    float **C_std = malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++)
    {
        C_std[i] = calloc(p, sizeof(float));
    }
    standard_multiply(A, B, C_std, m, n, p);

    // printf("Matrix C Sparse\n");
    // print_dense_matrix(C_sparse, n, p);

    // printf("Matrix C Standard\n");
    // print_dense_matrix(C_std, n, p);

    printf("%s\n",
           compare_matrices(C_sparse, C_std, m, p) ? "Match!" : "Mismatch!");

    // Cleanup
    free_csr(A_val, A_col, A_row);
    // free_csc(Bt_val, Bt_row, Bt_col);
    free_csc(B_val, B_row, B_col);
    free_matrix(C_sparse, m);
    free_matrix(C_std, m);
    free_matrix(A, m);
    free_matrix(B, n);
    return 0;
}
