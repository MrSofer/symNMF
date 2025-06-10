"""
Python interface for the symNMF clustering algorithm.

This file handles command line arguments, data loading,
initialization of H for symNMF, and interfacing with the C extension.
"""

import sys
import numpy as np
import symnmfmodule # This will import the C extension module

# Set random seed for reproducibility as required
np.random.seed(1234)

def parse_arguments():
    """
    Parses command line arguments.

    Expected arguments:
    1. k (int): Number of required clusters.
    2. goal (str): Can be 'symnmf', 'sym', 'ddg', or 'norm'.
    3. file_name (str): Path to the input data file (.txt).

    Returns:
        tuple: (k, goal, file_name)
    """
    # Assuming arguments are always provided and valid as per instructions
    if len(sys.argv) != 4:
        print("An Error Has Occurred")
        exit(1)

    k = sys.argv[1]
    goal = sys.argv[2]
    file_name = sys.argv[3]

    # goal validation and k value validation (whole number and larger than 1)
    valid_goals = ['symnmf', 'sym', 'ddg', 'norm']
    if goal not in valid_goals:
        print("An Error Has Occurred")
        exit(1)

    return k, goal, file_name

def load_data(file_name):
    """
    Loads data points from the specified file.

    Args:
        file_name (str): Path to the input data file (.txt).

    Returns:
        np.ndarray: A numpy array of data points.
    """
    try:
        # Data points are expected to be comma-separated floats
        data = np.loadtxt(file_name, delimiter=',')
        if len(data) == 0:
            print("An Error Has Occurred")
            exit(1)
        return data
    except Exception:
        # Handle potential file reading errors
        print("An Error Has Occurred")
        exit(1)

def initialize_h(W, k):
    """
    Initializes the matrix H for symNMF.

    Initialization is random with values from [0, 2 * sqrt(m/k)],
    where m is the average of all entries of W.

    Args:
        W (np.ndarray): The normalized similarity matrix.
        k (int): The number of clusters.

    Returns:
        np.ndarray: The initialized H matrix.
    """
    n = W.shape[0]
    # Calculate the average of all entries in W
    m = np.mean(W)
    # Calculate the upper bound for uniform distribution
    upper_bound = 2 * np.sqrt(m / k)

    # Initialize H with random values from [0, upper_bound]
    H = np.random.uniform(0, upper_bound, size=(n, k))
    return H

def print_matrix(matrix):
    """
    Prints a matrix to standard output, formatted to 4 decimal places.

    Args:
        matrix (np.ndarray): The matrix to print.
    """
    # Iterate through rows
    for row in matrix:
        # Format each element to 4 decimal places and join with commas
        print(','.join([f'{x:.4f}' for x in row]))

def main():
    """
    Main function to execute the symNMF process based on arguments.
    """
    sk, goal, file_name = parse_arguments()
    data = load_data(file_name)

    # Determine which C function to call based on the goal
    if goal == 'sym':
        # Call the C function for similarity matrix
        similarity_matrix = symnmfmodule.sym(data.tolist()) # Convert numpy array to list
        print_matrix(np.array(similarity_matrix)) # Convert list back to numpy for printing
    elif goal == 'ddg':
        # Call the C function for diagonal degree matrix
        ddg_matrix = symnmfmodule.ddg(data.tolist())
        print_matrix(np.array(ddg_matrix))
    elif goal == 'norm':
        # Call the C function for normalized similarity matrix
        norm_matrix = symnmfmodule.norm(data.tolist())
        print_matrix(np.array(norm_matrix))
    elif goal == 'symnmf':
        try:
            fk = float(sk)
        except:
            print("An Error Has Occurred")
            exit(1)

        k = int(fk)

        if len(data) <= k or k != fk or k <= 1:
            print("An Error Has Occurred")
            exit(1)
        # For symnmf, first get the normalized similarity matrix W
        W = np.array(symnmfmodule.norm(data.tolist()))
        # Initialize H
        H = initialize_h(W, k)
        # Call the C function for symNMF optimization
        final_H = symnmfmodule.symnmf(H.tolist(), W.tolist()) # Pass initial H and W
        print_matrix(np.array(final_H))

if __name__ == "__main__":
    main()
