#ifndef SYMNMF_H
#define SYMNMF_H

/*
 * Header file for the C implementation of symNMF functions.
 * Declares function prototypes used in symnmfmodule.c and implemented in symnmf.c.
 */

/* Define constants for convergence (moved from symnmf.c for potential shared use) */
#define EPSILON 1e-4
#define MAX_ITER 300
#define BETA 0.5

/* Define a small epsilon for numerical stability in division */
#define EPSILON_DIV 1e-10

/* Function to calculate the similarity matrix
 * data: 2D array of data points (n x d)
 * n: number of data points
 * d: dimension of data points
 * Returns: 2D array representing the similarity matrix (n x n)
 */
double** calculate_similarity_matrix(double** data, int n, int d);

/* Function to calculate the diagonal degree matrix
 * similarity_matrix: 2D array of the similarity matrix (n x n)
 * n: number of data points
 * Returns: 2D array representing the diagonal degree matrix (n x n)
 */
double** calculate_ddg_matrix(double** similarity_matrix, int n);

/* Function to calculate the normalized similarity matrix
 * similarity_matrix: 2D array of the similarity matrix (n x n)
 * ddg_matrix: 2D array of the diagonal degree matrix (n x n)
 * n: number of data points
 * Returns: 2D array representing the normalized similarity matrix (n x n)
 */
double** calculate_normalized_similarity_matrix(double** similarity_matrix, double** ddg_matrix, int n);

/* Function to optimize H using the iterative update rule
 * H: Initial H matrix (n x k)
 * W: Normalized similarity matrix (n x n)
 * n: number of data points
 * k: number of clusters
 * Returns: Optimized H matrix (n x k)
 */
double** optimize_h(double** H, double** W, int n, int k);

/* Helper function to free allocated memory for a 2D array
 * matrix: The 2D array to free
 * rows: The number of rows in the matrix
 */
void free_matrix(double** matrix, int rows);

/* Helper function to allocate memory for a 2D array
 * rows: The number of rows
 * cols: The number of columns
 * Returns: Allocated 2D array
 */
double** allocate_matrix(int rows, int cols);

/* Helper function to calculate the squared Euclidean distance between two vectors
 * vec1: The first vector
 * vec2: The second vector
 * d: The dimension of the vectors
 * Returns: The squared Euclidean distance
 */
double squared_euclidean_distance(double* vec1, double* vec2, int d);

/* Helper function to calculate the Frobenius norm squared of the difference between two matrices
 * matrix1: The first matrix
 * matrix2: The second matrix
 * rows: The number of rows
 * cols: The number of columns
 * Returns: The squared Frobenius norm of the difference
 */
double frobenius_norm_squared_difference(double** matrix1, double** matrix2, int rows, int cols);

/* Helper function to calculate matrix product C = A * B
 * A: First matrix (rows_A x cols_A)
 * B: Second matrix (rows_B x cols_B)
 * rows_A, cols_A, rows_B, cols_B: Dimensions of matrices A and B
 * Returns: Result matrix C (rows_A x cols_B)
 */
double** multiply_matrices(double** A, double** B, int rows_A, int cols_A, int rows_B, int cols_B);

/* Helper function to perform one iteration of the H update rule
 * H: Current H matrix (n x k)
 * W: Normalized similarity matrix (n x n)
 * n: number of data points
 * k: number of clusters
 * Returns: Updated H matrix (n x k)
 */
double** update_h_iteration(double** H, double** W, int n, int k);

/* Helper function to calculate the transpose of a matrix
 * matrix: The input matrix (rows x cols)
 * rows: The number of rows in the input matrix
 * cols: The number of columns in the input matrix
 * Returns: The transposed matrix (cols x rows)
 */
double** calculate_Ht_matrix(double** matrix, int rows, int cols);


#endif /* SYMNMF_H */
