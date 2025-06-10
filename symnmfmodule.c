#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "symnmf.h" /* Include the C header file */

/*
 * Python C API wrapper for the symNMF functions.
 * Exposes C functions to Python.
 */

/* Fill c matrix data with py matrix data */
double **populate_matrix(PyObject *py_matrix, double **c_matrix, int *n, int *d)
{
    int i, j;
    PyObject *row, *item;

    for (i = 0; i < *n; i++)
    {
        row = PyList_GetItem(py_matrix, i);
        if (!PyList_Check(row) || PyList_Size(row) != *d)
        {
            /* Free allocated memory before returning error */
            free_matrix(c_matrix, i + 1);
            return NULL;
        }
        for (j = 0; j < *d; j++)
        {
            item = PyList_GetItem(row, j);
            if (!PyFloat_Check(item) && !PyLong_Check(item))
            {
                /* Free allocated memory before returning error */
                free_matrix(c_matrix, i + 1);
                return NULL;
            }
            c_matrix[i][j] = PyFloat_AsDouble(item);
        }
    }
    return c_matrix;
}

/* Helper function to convert a Python list of lists (representing a matrix) to a C 2D array */
double **py_list_to_c_matrix(PyObject *py_matrix, int *n, int *d)
{
    PyObject *first_row;
    double **c_matrix;

    /* Check if the input is a list */
    if (!PyList_Check(py_matrix))
    {
        return NULL;
    }

    *n = PyList_Size(py_matrix); /* Number of rows */
    if (*n == 0)
    {
        *d = 0;
    }
    else
    {
        /* Assume all rows have the same number of columns */
        first_row = PyList_GetItem(py_matrix, 0);
        if (!PyList_Check(first_row))
        {
            return NULL;
        }
        *d = PyList_Size(first_row); /* Number of columns */
    }
    c_matrix = allocate_matrix(*n, *d);
    if (c_matrix == NULL)
    {
        return NULL;
    }
    c_matrix = populate_matrix(py_matrix, c_matrix, n, d);
    return c_matrix;
}

/* Helper function to convert a C 2D array to a Python list of lists */
PyObject *c_matrix_to_py_list(double **c_matrix, int n, int d)
{
    PyObject *py_matrix, *row, *item;
    int i, j;
    py_matrix = PyList_New(n);
    if (py_matrix == NULL)
    {
        return NULL;
    }

    for (i = 0; i < n; i++)
    {
        row = PyList_New(d);
        for (j = 0; j < d; j++)
        {
            item = PyFloat_FromDouble(c_matrix[i][j]);
            PyList_SetItem(row, j, item);
        }
        PyList_SetItem(py_matrix, i, row);
    }
    return py_matrix;
}

/* symnmf(H, W) function exposed to Python */
static PyObject *symnmf_symnmf(PyObject *self, PyObject *args)
{
    PyObject *py_H, *py_W, *py_final_H;
    int n_H, k, n_W, d_W;
    double **c_H, **c_W, **final_c_H;

    /* Set error string in advance */
    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    /* Parse arguments: two lists of lists (H and W) */
    if (!PyArg_ParseTuple(args, "OO", &py_H, &py_W))
        return NULL;

    /* Convert Python lists to C matrices */
    c_H = py_list_to_c_matrix(py_H, &n_H, &k);
    if (c_H == NULL)
        return NULL;

    c_W = py_list_to_c_matrix(py_W, &n_W, &d_W);
    if (c_W == NULL)
    {
        free_matrix(c_H, n_H); /* Free H if W conversion fails */
        return NULL;
    }

    /* Call the C optimization function */
    final_c_H = optimize_h(c_H, c_W, n_H, k);

    /* Free the input C matrices (they were copies) */
    free_matrix(c_H, n_H);
    free_matrix(c_W, n_W);

    if (final_c_H == NULL)
        return NULL;

    /* Convert the result back to a Python list of lists and free the result C matrix */
    py_final_H = c_matrix_to_py_list(final_c_H, n_H, k);
    free_matrix(final_c_H, n_H);
    PyErr_Clear();
    return py_final_H;
}

/* sym(data) function exposed to Python */
static PyObject *symnmf_sym(PyObject *self, PyObject *args)
{
    PyObject *py_data, *py_similarity_matrix;
    double **c_data, **similarity_matrix;
    int n, d;

    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    /* Parse arguments: one list of lists (data points) */
    if (!PyArg_ParseTuple(args, "O", &py_data))
        return NULL;

    /* Convert Python list to C matrix */
    c_data = py_list_to_c_matrix(py_data, &n, &d);
    if (c_data == NULL)
        return NULL;

    /* Calculate the similarity matrix */
    similarity_matrix = calculate_similarity_matrix(c_data, n, d);

    /* Free the input data matrix */
    free_matrix(c_data, n);

    if (similarity_matrix == NULL)
        return NULL;

    /* Convert the result back to a Python list of lists */
    py_similarity_matrix = c_matrix_to_py_list(similarity_matrix, n, n);

    /* Free the result C matrix */
    free_matrix(similarity_matrix, n);
    PyErr_Clear();
    return py_similarity_matrix;
}

/* ddg(data) function exposed to Python */
static PyObject *symnmf_ddg(PyObject *self, PyObject *args)
{
    PyObject *py_data, *py_ddg_matrix;
    double **c_data, **similarity_matrix, **ddg_matrix;
    int n, d;

    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    /* Parse arguments: one list of lists (data points) */
    if (!PyArg_ParseTuple(args, "O", &py_data))
        return NULL;

    /* Convert Python list to C matrix */
    c_data = py_list_to_c_matrix(py_data, &n, &d);
    if (c_data == NULL)
        return NULL;

    /* Calculate the similarity matrix (needed for DDG) */
    similarity_matrix = calculate_similarity_matrix(c_data, n, d);
    /* Free the input data matrix */
    free_matrix(c_data, n);

    if (similarity_matrix == NULL)
        return NULL;

    /* Calculate the diagonal degree matrix */
    ddg_matrix = calculate_ddg_matrix(similarity_matrix, n);
    /* Free the intermediate similarity matrix */
    free_matrix(similarity_matrix, n);

    if (ddg_matrix == NULL)
        return NULL;

    /* Convert the result back to a Python list of lists */
    py_ddg_matrix = c_matrix_to_py_list(ddg_matrix, n, n);
    /* Free the result C matrix */
    free_matrix(ddg_matrix, n);
    PyErr_Clear();
    return py_ddg_matrix;
}

/* norm(data) function exposed to Python */
static PyObject *symnmf_norm(PyObject *self, PyObject *args)
{
    PyObject *py_data, *py_normalized_matrix;
    int n, d;
    double **c_data, **similarity_matrix, **ddg_matrix, **normalized_matrix;

    PyErr_SetString(PyExc_RuntimeError, "An Error Has Occurred");
    /* Parse arguments: one list of lists (data points) */
    if (!PyArg_ParseTuple(args, "O", &py_data))
        return NULL;
    /* Convert Python list to C matrix */
    c_data = py_list_to_c_matrix(py_data, &n, &d);
    if (c_data == NULL)
        return NULL;
    /* Calculate the similarity matrix */
    similarity_matrix = calculate_similarity_matrix(c_data, n, d);
    /* Free the input data matrix */
    free_matrix(c_data, n);
    if (similarity_matrix == NULL)
        return NULL;
    /* Calculate the diagonal degree matrix */
    ddg_matrix = calculate_ddg_matrix(similarity_matrix, n);
    if (ddg_matrix == NULL)
    {
        free_matrix(similarity_matrix, n); /* Free similarity if DDG fails */
        return NULL;
    }
    /* Calculate the normalized similarity matrix then free used matrices*/
    normalized_matrix = calculate_normalized_similarity_matrix(similarity_matrix, ddg_matrix, n);
    free_matrix(similarity_matrix, n);
    free_matrix(ddg_matrix, n);
    if (normalized_matrix == NULL)
        return NULL;
    /* Convert the result back to a Python list of lists */
    py_normalized_matrix = c_matrix_to_py_list(normalized_matrix, n, n);
    /* Free the result C matrix */
    free_matrix(normalized_matrix, n);
    PyErr_Clear();
    return py_normalized_matrix;
}

/* Method definitions */
static PyMethodDef symnmf_methods[] = {
    {"symnmf", symnmf_symnmf, METH_VARARGS, "Performs symNMF optimization."},
    {"sym", symnmf_sym, METH_VARARGS, "Calculates the similarity matrix."},
    {"ddg", symnmf_ddg, METH_VARARGS, "Calculates the diagonal degree matrix."},
    {"norm", symnmf_norm, METH_VARARGS, "Calculates the normalized similarity matrix."},
    {NULL, NULL, 0, NULL} /* Sentinel */
};

/* Module definition */
static struct PyModuleDef symnmfmodule = {
    PyModuleDef_HEAD_INIT,
    "symnmfmodule", /* name of module */
    NULL,           /* module documentation, may be NULL */
    -1,             /* size of per-interpreter state of the module, or -1 if the module keeps state in global variables. */
    symnmf_methods};

/* Module initialization function */
PyMODINIT_FUNC PyInit_symnmfmodule(void)
{
    return PyModule_Create(&symnmfmodule);
}
