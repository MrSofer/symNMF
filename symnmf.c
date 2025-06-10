#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h> /* Required for strcmp */
#include "symnmf.h" /* Include the header file */

/*
 * C implementation of the symNMF functions.
 * Includes functions for calculating similarity matrix, diagonal degree matrix,
 * normalized similarity matrix, and optimizing H.
 * Also includes main function for standalone execution.
 */

/* Helper function to allocate memory for a 2D array */
double **allocate_matrix(int rows, int cols)
{
    double **matrix;
    int i, j; /* Declare loop variable at the beginning of the block */

    matrix = (double **)calloc(rows, sizeof(double *));
    if (matrix == NULL)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }
    for (i = 0; i < rows; i++)
    {
        matrix[i] = (double *)calloc(cols, sizeof(double));
        if (matrix[i] == NULL)
        {
            /* Free previously allocated rows before exiting */
            for (j = 0; j < i; j++)
            {
                free(matrix[j]);
            }
            free(matrix);
            printf("An Error Has Occurred\n");
            exit(1);
        }
    }
    return matrix;
}

/* Helper function to free allocated memory for a 2D array */
void free_matrix(double **matrix, int rows)
{
    int i; /* Declare loop variable at the beginning of the block */
    if (matrix == NULL)
        return;
    for (i = 0; i < rows; i++)
    {
        free(matrix[i]);
    }
    free(matrix);
}

/* Helper function to calculate the squared Euclidean distance between two vectors */
double squared_euclidean_distance(double *vec1, double *vec2, int d)
{
    double sum = 0.0;
    int i;
    for (i = 0; i < d; i++)
    {
        sum += pow(vec1[i] - vec2[i], 2);
    }
    return sum;
}

/* Helper function to calculate the Frobenius norm squared of the difference between two matrices */
double frobenius_norm_squared_difference(double **matrix1, double **matrix2, int rows, int cols)
{
    double sum = 0.0;
    int i, j; /* Declare loop variables at the beginning of the block */
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            sum += pow(matrix1[i][j] - matrix2[i][j], 2);
        }
    }
    return sum;
}

/* Helper function to calculate matrix product C = A * B */
double **multiply_matrices(double **A, double **B, int rows_A, int cols_A, int rows_B, int cols_B)
{
    double **C;
    int i, j, l; /* Declare loop variables at the beginning of the block */

    /* Check if multiplication is possible */
    if (cols_A != rows_B)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }

    C = allocate_matrix(rows_A, cols_B);

    for (i = 0; i < rows_A; i++)
    {
        for (j = 0; j < cols_B; j++)
        {
            for (l = 0; l < cols_A; l++)
            { /* Or rows_B */
                C[i][j] += A[i][l] * B[l][j];
            }
        }
    }
    return C;
}

/* Function to calculate the similarity matrix */
double **calculate_similarity_matrix(double **data, int n, int d)
{
    double **affinity_matrix, dist_sq;
    int i, j; /* Declare loop variables at the beginning of the block */

    affinity_matrix = allocate_matrix(n, n);

    for (i = 0; i < n; i++)
    {
        for (j = 0; j < n; j++)
        {
            if (i == j)
            {
                affinity_matrix[i][j] = 0.0;
            }
            else
            {
                dist_sq = squared_euclidean_distance(data[i], data[j], d);
                affinity_matrix[i][j] = exp(-dist_sq / 2.0);
            }
        }
    }
    return affinity_matrix;
}

/* Function to calculate the diagonal degree matrix */
double **calculate_ddg_matrix(double **similarity_matrix, int n)
{
    double **degree_matrix, degree;
    int i, j; /* Declare loop variables at the beginning of the block */

    degree_matrix = allocate_matrix(n, n);

    for (i = 0; i < n; i++)
    {
        degree = 0.0;
        for (j = 0; j < n; j++)
        {
            degree += similarity_matrix[i][j];
        }
        degree_matrix[i][i] = degree;
    }
    return degree_matrix;
}

/* Function to calculate the normalized similarity matrix */
double **calculate_normalized_similarity_matrix(double **similarity_matrix, double **ddg_matrix, int n)
{
    double **inv_sqrt_ddg, **temp_matrix, **normalized_matrix;
    int i; /* Declare loop variable at the beginning of the block */

    /* Calculate D^(-1/2) */
    inv_sqrt_ddg = allocate_matrix(n, n);
    for (i = 0; i < n; i++)
    {
        if (ddg_matrix[i][i] > 0)
        { /* Avoid division by zero */
            inv_sqrt_ddg[i][i] = 1.0 / sqrt(ddg_matrix[i][i]);
        }
        else
        {
            /* Handle cases where degree is zero, though problem assumes valid data */
            inv_sqrt_ddg[i][i] = 0.0;
        }
    }

    /* Calculate D^(-1/2) * A */
    temp_matrix = multiply_matrices(inv_sqrt_ddg, similarity_matrix, n, n, n, n);

    /* Calculate (D^(-1/2) * A) * D^(-1/2) */
    normalized_matrix = multiply_matrices(temp_matrix, inv_sqrt_ddg, n, n, n, n);

    free_matrix(inv_sqrt_ddg, n);
    free_matrix(temp_matrix, n);

    return normalized_matrix;
}

/* Helper function to calculate the transpose of a matrix */
double** calculate_Ht_matrix(double** matrix, int rows, int cols) {
    double** transposed_matrix;
    int i, j; /* Declare loop variables at the beginning of the block */

    transposed_matrix = allocate_matrix(cols, rows);

    for (i = 0; i < rows; i++) {
        for (j = 0; j < cols; j++) {
            transposed_matrix[j][i] = matrix[i][j];
        }
    }
    return transposed_matrix;
}

/* Helper function to perform one iteration of the H update rule */
double** update_h_iteration(double** H, double** W, int n, int k) {
    double** H_new, **H_T, **HH_T, **HHT_H, **WH;
    int i, j; /* Declare loop variables at the beginning of the block */

    H_new = allocate_matrix(n, k);

    /* Calculate H^T */
    H_T = calculate_Ht_matrix(H, n, k);

    /* Calculate H * H^T */
    HH_T = multiply_matrices(H, H_T, n, k, k, n);

    /* Calculate (H * H^T) * H */
    HHT_H = multiply_matrices(HH_T, H, n, n, n, k);

    /* Calculate W * H */
    WH = multiply_matrices(W, H, n, n, n, k);

    /* Update H */
    for (i = 0; i < n; i++) {
        for (j = 0; j < k; j++) {
            if (HHT_H[i][j] != 0)
                H_new[i][j] = H[i][j] * (1 - BETA + BETA * (WH[i][j] / (HHT_H[i][j])));
            else
                H_new[i][j] = H[i][j] * (1 - BETA + BETA * (WH[i][j] / (HHT_H[i][j]+1e-6)));
        }
    }

    /* Free temporary matrices */
    free_matrix(H_T, k); /* Free H_T (k rows) */
    free_matrix(HH_T, n);
    free_matrix(HHT_H, n);
    free_matrix(WH, n);

    return H_new;
}

/* Function to optimize H using the iterative update rule */
double **optimize_h(double **H, double **W, int n, int k)
{
    double **H_current, **H_prev, **H_new, frobenius_diff;
    int iter, i, j; /* Declare loop variables at the beginning of the block */

    H_current = allocate_matrix(n, k);
    /* Copy initial H to H_current */
    for (i = 0; i < n; i++)
    {
        for (j = 0; j < k; j++)
        {
            H_current[i][j] = H[i][j];
        }
    }

    H_prev = allocate_matrix(n, k);

    for (iter = 0; iter < MAX_ITER; iter++)
    {
        /* Copy H_current to H_prev for convergence check */
        for (i = 0; i < n; i++)
        {
            for (j = 0; j < k; j++)
            {
                H_prev[i][j] = H_current[i][j];
            }
        }

        /* Perform one update iteration */
        H_new = update_h_iteration(H_current, W, n, k);

        /* Free the previous H_current matrix */
        free_matrix(H_current, n);
        H_current = H_new; /* H_current now points to the newly updated matrix */

        /* Check for convergence */
        frobenius_diff = frobenius_norm_squared_difference(H_current, H_prev, n, k);

        /* Check convergence condition */
        if (frobenius_diff < EPSILON)
        {
            break; /* Converged */
        }
    }

    /* Free the H_prev matrix */
    free_matrix(H_prev, n);

    /* Return the final optimized H matrix */
    return H_current;
}

/* Helper function to read data from a file
 * Reads comma-separated float values into a 2D double array.
 * Assumes a rectangular matrix format.
 * Returns the 2D array and updates n (rows) and d (cols) by reference.
 */
double **read_data_from_file(const char *file_name, int *n, int *d)
{
    FILE *file = fopen(file_name, "r");
    char line[4096]; /* Assuming a maximum line length */
    int i, j;        /* Declare loop variables at the beginning of the block */
    double **data;

    if (file == NULL)
    {
        printf("An Error Has Occurred\n");
        exit(1);
    }

    /* Read the first line to determine the number of columns (d) */
    if (fgets(line, sizeof(line), file) == NULL)
    {
        fclose(file);
        *n = 0;
        *d = 0;
        return NULL; /* Empty file or read error */
    }

    *d = 0;
    /* Count commas to determine number of columns */
    for (i = 0; line[i] != '\0'; i++)
    {
        if (line[i] == ',')
        {
            (*d)++;
        }
    }
    (*d)++; /* Add 1 for the last number */

    /* Reset file pointer to the beginning to read data points */
    fseek(file, 0, SEEK_SET);

    /* Count the number of lines (n) */
    *n = 0;
    while (fgets(line, sizeof(line), file) != NULL)
    {
        (*n)++;
    }

    /* Reset file pointer again */
    fseek(file, 0, SEEK_SET);

    data = allocate_matrix(*n, *d);

    /* Read data points */
    for (i = 0; i < *n; i++)
    {
        for (j = 0; j < *d; j++)
        {
            if (fscanf(file, "%lf%*c", &data[i][j]) != 1)
            {
                /* Error reading data */
                free_matrix(data, *n);
                fclose(file);
                printf("An Error Has Occurred\n");
                exit(1);
            }
        }
    }

    fclose(file);
    return data;
}

/* Helper function to print a matrix to standard output
 * Formats elements to 4 decimal places.
 */
void print_matrix(double **matrix, int rows, int cols)
{
    int i, j; /* Declare loop variables at the beginning of the block */
    for (i = 0; i < rows; i++)
    {
        for (j = 0; j < cols; j++)
        {
            printf("%.4f%s", matrix[i][j], (j == cols - 1) ? "" : ",");
        }
        printf("\n");
    }
}

/* Helper function to process the goal and get the result matrix */
double **process_goal_and_get_result(const char *goal, double **data, int n, int d)
{
    double **result_matrix = NULL, **similarity_matrix = NULL, **ddg_matrix = NULL;

    if (strcmp(goal, "sym") == 0)
    {
        result_matrix = calculate_similarity_matrix(data, n, d);
    }
    else if (strcmp(goal, "ddg") == 0)
    {
        similarity_matrix = calculate_similarity_matrix(data, n, d);
        if (similarity_matrix == NULL)
            return NULL; /* Error handled in calculate_... */
        result_matrix = calculate_ddg_matrix(similarity_matrix, n);
        free_matrix(similarity_matrix, n); /* Free intermediate matrix */
    }
    else if (strcmp(goal, "norm") == 0)
    {
        similarity_matrix = calculate_similarity_matrix(data, n, d);
        if (similarity_matrix == NULL)
            return NULL; /* Error handled in calculate_... */
        ddg_matrix = calculate_ddg_matrix(similarity_matrix, n);
        if (ddg_matrix == NULL)
        {
            free_matrix(similarity_matrix, n);
            return NULL; /* Error handled in calculate_... */
        }
        result_matrix = calculate_normalized_similarity_matrix(similarity_matrix, ddg_matrix, n);
        free_matrix(similarity_matrix, n); /* Free intermediate matrices */
        free_matrix(ddg_matrix, n);
    }
    else
    {
        /* Invalid goal - this case should ideally be caught before calling this function */
        return NULL;
    }

    return result_matrix;
}

/* Main function for standalone execution */
int main(int argc, char *argv[])
{
    char *goal, *file_name;
    double **data, **result_matrix;
    int n, d, result_rows, result_cols;

    /* Check for correct number of arguments */
    if (argc != 3)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    goal = argv[1];
    file_name = argv[2];

    /* Validate goal before reading data */
    if (strcmp(goal, "sym") != 0 && strcmp(goal, "ddg") != 0 && strcmp(goal, "norm") != 0)
    {
        printf("An Error Has Occurred\n");
        return 1;
    }

    data = read_data_from_file(file_name, &n, &d);

    if (data == NULL || n == 0 || d == 0)
    {
        /* read_data_from_file handles errors and exits, but check for empty data */
        if (data != NULL)
            free_matrix(data, n);
        printf("An Error Has Occurred\n");
        return 1;
    }

    result_matrix = process_goal_and_get_result(goal, data, n, d);

    /* Free the input data matrix as it's no longer needed */
    free_matrix(data, n);

    /* Print the result matrix */
    if (result_matrix != NULL)
    {
        /* For sym, ddg, norm, result is n x n */
        result_rows = n;
        result_cols = n;
        print_matrix(result_matrix, result_rows, result_cols);
        free_matrix(result_matrix, result_rows); /* Free the result matrix */
    }
    else
    {
        /* Error occurred during processing */
        printf("An Error Has Occurred\n");
        return 1;
    }

    return 0;
}
