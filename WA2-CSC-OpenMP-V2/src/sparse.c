#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <omp.h>

#define EPSILON 1e-6
#ifndef SIZE
#define SIZE 1024
#endif

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

void convert_dense_to_csc(float **dense_matrix, float **values, int **row_indices, int **col_pointers, int rows, int cols)
{
    int non_zero_count = 0;

    *values = (float *)malloc(rows * cols * sizeof(float));
    *row_indices = (int *)malloc(rows * cols * sizeof(int));
    *col_pointers = (int *)malloc((cols + 1) * sizeof(int));

    if (!*values || !*row_indices || !*col_pointers)
    {
        fprintf(stderr, "Memory allocation failed for CSC matrix components.\n");
        exit(EXIT_FAILURE);
    }

    (*col_pointers)[0] = 0;

    for (int j = 0; j < cols; j++)
    {
        (*col_pointers)[j + 1] = (*col_pointers)[j];
        for (int i = 0; i < rows; i++)
        {
            if (fabs(dense_matrix[i][j]) > EPSILON)
            {
                (*values)[non_zero_count] = dense_matrix[i][j];
                (*row_indices)[non_zero_count] = i;
                non_zero_count++;
                (*col_pointers)[j + 1]++;
            }
        }
    }
}

void free_matrix(float **matrix, int rows)
{
    for (int i = 0; i < rows; i++)
        free(matrix[i]);
    free(matrix);
}

void free_csc_matrix(float *values, int *row_indices, int *col_pointers)
{
    free(values);
    free(row_indices);
    free(col_pointers);
}

void sparse_multiply_csc(float *__restrict__ A_values, int *__restrict__ A_row_indices, int *__restrict__ A_col_pointers,
                         float *__restrict__ B_values, int *__restrict__ B_row_indices, int *__restrict__ B_col_pointers,
                         float **__restrict__ C, int p)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int j = 0; j < p; j++)
    {
        for (int k = B_col_pointers[j]; k < B_col_pointers[j + 1]; k++)
        {
            int rowB = B_row_indices[k];
            float valueB = B_values[k];

            for (int i = A_col_pointers[rowB]; i < A_col_pointers[rowB + 1]; i++)
            {
                int colA = A_row_indices[i];
                float valueA = A_values[i];

#pragma omp atomic
                C[colA][j] += valueA * valueB;
            }
        }
    }
}

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
    srand((unsigned int)time(NULL));

    int m = SIZE;
    int n = SIZE;
    int p = SIZE;
    int percentZeros = 90;

    float **A = create_matrix(m, n, percentZeros);
    float **B = create_matrix(n, p, percentZeros);

    float *A_values, *B_values;
    int *A_row_indices, *A_col_pointers;
    int *B_row_indices, *B_col_pointers;
    convert_dense_to_csc(A, &A_values, &A_row_indices, &A_col_pointers, m, n);
    convert_dense_to_csc(B, &B_values, &B_row_indices, &B_col_pointers, n, p);

    float **C_sparse = (float **)malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++)
    {
        C_sparse[i] = (float *)calloc(p, sizeof(float));
        if (!C_sparse[i])
        {
            fprintf(stderr, "Memory allocation failed for result matrix row %d.\n", i);
            exit(EXIT_FAILURE);
        }
    }

    sparse_multiply_csc(A_values, A_row_indices, A_col_pointers,
                        B_values, B_row_indices, B_col_pointers,
                        C_sparse, p);

    float **C_std = (float **)malloc(m * sizeof(float *));
    for (int i = 0; i < m; i++)
    {
        C_std[i] = (float *)calloc(p, sizeof(float));
    }

    standard_multiply(A, B, C_std, m, n, p);

    if (compare_matrices(C_sparse, C_std, m, p))
        printf("The sparse multiplication result matches the control matrix.\n");
    else
        printf("The sparse multiplication result does NOT match the control matrix.\n");

    free_csc_matrix(A_values, A_row_indices, A_col_pointers);
    free_csc_matrix(B_values, B_row_indices, B_col_pointers);
    free_matrix(C_sparse, m);
    free_matrix(C_std, m);
    free_matrix(A, m);
    free_matrix(B, n);

    return 0;
}
