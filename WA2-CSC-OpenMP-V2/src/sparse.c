#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 1024
#endif

// Function to create a standard matrix with a specified percentage of zeros (value between 0-99).
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

// Function to convert a standard dense matrix to the CSC format
void convert_dense_to_csc(float **dense_matrix, float **values, int **row_indices, int **col_pointers, int rows, int cols)
{
    int non_zero_count = 0;

    // Allocate memory for the CSC representation
    *values = (float *)malloc(rows * cols * sizeof(float)); // Worst case, all non-zero
    *row_indices = (int *)malloc(rows * cols * sizeof(int));
    *col_pointers = (int *)malloc((cols + 1) * sizeof(int)); // One extra for the last column pointer

    if (!*values || !*row_indices || !*col_pointers)
    {
        fprintf(stderr, "Memory allocation failed for CSC matrix components.\n");
        exit(EXIT_FAILURE);
    }

    // Initialize the column pointers
    (*col_pointers)[0] = 0;

    // Iterate over the dense matrix and populate the CSC format
#pragma omp parallel for schedule(static)
    for (int j = 0; j < cols; j++)
    {
        (*col_pointers)[j + 1] = (*col_pointers)[j]; // Default to same as previous column

        for (int i = 0; i < rows; i++)
        {
            if (fabs(dense_matrix[i][j]) > EPSILON)
            {
                (*values)[non_zero_count] = dense_matrix[i][j]; // Non-zero value
                (*row_indices)[non_zero_count] = i;             // Row index of non-zero value
                non_zero_count++;
                (*col_pointers)[j + 1]++; // Update the column pointer
            }
        }
        }
}

// Free dynamically allocated dense matrix
void free_matrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

// Free dynamically allocated CSC matrix
void free_csc_matrix(float *values, int *row_indices, int *col_pointers)
{
    free(values);
    free(row_indices);
    free(col_pointers);
}

// CSC matrix multiplication
void sparse_multiply_csc(float *A_values, int *A_row_indices, int *A_col_pointers,
                         float *B_values, int *B_row_indices, int *B_col_pointers,
                         float **C, int p)
{
// Multiply each column of B by A in the CSC format
#pragma omp parallel for schedule(static)
    for (int j = 0; j < p; j++)
    {
#pragma omp parallel for schedule(static)

        for (int k = B_col_pointers[j]; k < B_col_pointers[j + 1]; k++)
        {
            int rowB = B_row_indices[k];
            float valueB = B_values[k];

            // Multiply with the corresponding column of A
#pragma omp parallel for schedule(static)

            for (int i = A_col_pointers[rowB]; i < A_col_pointers[rowB + 1]; i++)
            {
                int colA = A_row_indices[i];
                float valueA = A_values[i];

                // Update C[i][j]
                C[colA][j] += valueA * valueB;
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

// Function to print a matrix
void print_matrix(float **matrix, int rows, int cols)
{
    printf("\nMatrix (first 5 rows, 5 columns):\n");
    for (int i = 0; i < (rows < 5 ? rows : 5); i++)
    {
        for (int j = 0; j < (cols < 5 ? cols : 5); j++)
        {
            printf("%.2f ", matrix[i][j]);
        }
        printf("\n");
    }
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
    int percentZeros = 70;

    // Create matrices A and B in standard dense format
    float **A = create_matrix(m, n, percentZeros);
    float **B = create_matrix(n, p, percentZeros);

    // Convert dense matrices A and B to CSC format
    float *A_values, *B_values;
    int *A_row_indices, *A_col_pointers;
    int *B_row_indices, *B_col_pointers;
    convert_dense_to_csc(A, &A_values, &A_row_indices, &A_col_pointers, m, n);
    convert_dense_to_csc(B, &B_values, &B_row_indices, &B_col_pointers, n, p);

    // Allocate result matrix C (dense format)
    float **C_sparse = (float **)malloc(m * sizeof(float *));
    if (!C_sparse)
    {
        fprintf(stderr, "Memory allocation failed for result matrix C_sparse.\n");
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < m; i++)
    {
        C_sparse[i] = (float *)malloc(p * sizeof(float));
        if (!C_sparse[i])
        {
            fprintf(stderr, "Memory allocation failed for result matrix row %d.\n", i);
            exit(EXIT_FAILURE);
        }

        // Initialize matrix C_sparse to zero.
        for (int j = 0; j < p; j++)
        {
            C_sparse[i][j] = 0.0f;
        }
    }

    // Perform sparse matrix multiplication (CSC)
    sparse_multiply_csc(A_values, A_row_indices, A_col_pointers,
                        B_values, B_row_indices, B_col_pointers,
                        C_sparse, p);

    // Perform the standard multiplication to create the control matrix
    float **C_std = (float **)malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++)
    {
        C_std[i] = (float *)malloc(p * sizeof(float));
        for (int j = 0; j < p; j++)
        {
            C_std[i][j] = 0.0f;
        }
    }

    standard_multiply(A, B, C_std, m, n, p);

    // Compare the results
    if (compare_matrices(C_sparse, C_std, m, p))
        printf("The sparse multiplication result matches the control matrix.\n");
    else
        printf("The sparse multiplication result does NOT match the control matrix.\n");

    // Print the first 5 rows and 5 columns of C_sparse and C_std
    // print_matrix(C_sparse, m, p);
    // print_matrix(C_std, m, p);

    // Free all allocated memory
    free_csc_matrix(A_values, A_row_indices, A_col_pointers);
    free_csc_matrix(B_values, B_row_indices, B_col_pointers);
    free_matrix(C_sparse, m);
    free_matrix(C_std, m);
    free_matrix(A, m);
    free_matrix(B, n);

    return 0;
}
