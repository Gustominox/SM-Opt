#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 1024
#endif
// Function to create a float matrix with a specified percentage of zeros (value between 0-99).
// Non-zero entries are generated randomly (values between 1.0 and 10.0).
float **create_matrix(int rows, int cols, int percentZeros)
{
    float **matrix = (float **)malloc(rows * sizeof(float *));
    if (!matrix)
    {
        fprintf(stderr, "Memory allocation failed for matrix rows.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < rows; i++)
    {
        matrix[i] = (float *)malloc(cols * sizeof(float));

        if (!matrix[i])
        {
            fprintf(stderr, "Memory allocation failed for matrix[%d].\n", i);
            exit(EXIT_FAILURE);
        }

        for (int j = 0; j < cols; j++)
        {
            if ((rand() % 100) < percentZeros)
                matrix[i][j] = 0.0f;
            else
                matrix[i][j] = (float)((rand() % 10) + 1);
        }
    }
    return matrix;
}

void free_matrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

// Naive sparse matrix multiplication.
// Multiplies matrix A (of size m x n) by matrix B (of size n x p)
// to produce matrix C (of size m x p).
// This version checks for zero entries to avoid unnecessary multiplications.
// This is your base code. Now, improve it in any way you wish!
void sparse_multiply(float **A, float **B, float **C, int m, int n, int p)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < m; i++)
    {
#pragma omp parallel for schedule(static)
        for (int k = 0; k < n; k++)
        {
            if (fabs(A[i][k]) > EPSILON)
            {
#pragma omp parallel for schedule(static)
                for (int j = 0; j < p; j++)
                {
                    if (fabs(B[k][j]) > EPSILON)
                    {
                        C[i][j] += A[i][k] * B[k][j];
                    }
                }
            }
        }
    }
}

// Standard matrix multiplication (control method).
void standard_multiply(float **A, float **B, float **C, int m, int n, int p)
{
    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < p; j++)
        {
            for (int k = 0; k < n; k++)
            {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }
}

// Function to compare two float matrices using a tolerance.
// Returns 1 if the matrices are identical within the tolerance, otherwise 0.
int compare_matrices(float **mat1, float **mat2, int rows, int cols)
{
    for (int i = 0; i < rows; i++)
    {
        for (int j = 0; j < cols; j++)
        {
            if (fabs(mat1[i][j] - mat2[i][j]) > EPSILON)
                return 0;
        }
    }
    return 1;
}

int main()
{
    // Seed the random number generator.
    srand((unsigned int)time(NULL));

    // Define matrix dimensions.
    int m = SIZE; // Rows in matrix A.
    int n = SIZE; // Columns in matrix A and rows in matrix B.
    int p = SIZE; // Columns in matrix B.

    // Define the percentage of zeros to be inserted (e.g., 70% zeros).
    int percentZeros = 90;

    // Create matrices A and B with the specified sparsity.
    float **A = create_matrix(m, n, percentZeros);
    float **B = create_matrix(n, p, percentZeros);

    // Allocate result matrices for sparse and standard multiplication.
    float **C_sparse = (float **)malloc(m * sizeof(float *));
    float **C_std = (float **)malloc(m * sizeof(float *));
    if (!C_sparse || !C_std)
    {
        fprintf(stderr, "Memory allocation failed for result matrices.\n");
        exit(EXIT_FAILURE);
    }
    for (int i = 0; i < m; i++)
    {
        C_sparse[i] = (float *)malloc(p * sizeof(float));
        C_std[i] = (float *)malloc(p * sizeof(float));
        if (!C_sparse[i] || !C_std[i])
        {
            fprintf(stderr, "Memory allocation failed for result matrix row %d.\n", i);
            exit(EXIT_FAILURE);
        }

        // Initialize matrix C to zero.
        for (int j = 0; j < p; j++)
        {
            C_sparse[i][j] = 0.0f;
            C_std[i][j] = 0.0f;
        }
    }

    // Perform the sparse multiplication.
    sparse_multiply(A, B, C_sparse, m, n, p);

    // Perform the standard multiplication to create the control matrix.
    standard_multiply(A, B, C_std, m, n, p);

    if (compare_matrices(C_sparse, C_std, m, p))
        printf("The sparse multiplication result matches the control matrix.\n");
    else
        printf("The sparse multiplication result does NOT match the control matrix.\n");

    // Free all allocated memory.
    free_matrix(A, m);
    free_matrix(B, n);
    free_matrix(C_sparse, m);
    free_matrix(C_std, m);

    return 0;
}
