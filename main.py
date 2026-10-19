# To use the program just run it in a reasonably new version of python 3 and wait for the result.
# Switch between convergences by entering either "1" or "2" as the convergence type.

from scipy.io import loadmat
import numpy as np


def check_for_convergence_first(x_k, x_k_plus_one, tol=1e-7):
    if np.linalg.norm(x_k_plus_one - x_k, ord=1).astype(np.float32) < tol:
        return True
    else:
        return False


def check_for_convergence_second(x_k, matrix, eigenvalue, tol=1e-7):
    if np.linalg.norm(matrix.dot(x_k) - eigenvalue * x_k, ord=1).astype(np.float32) < tol:
        return True
    else:
        return False


def power_method(matrix, num_iterations=1000, tol=1e-7, convergence_type=1):
    x_k_initial = np.full((matrix.shape[0], 1), 1 / matrix.shape[0])
    x_k = x_k_initial.astype(np.float32)
    iterations = 0
    if matrix.shape[0] != matrix.shape[1]:
        return -1, -1, -1, False, "Not a square matrix."

    for i in range(num_iterations):

        # x_k - last iteration of the vector
        # x_k_plus_one - new iteration of the vector
        x_k_plus_one = matrix.dot(x_k).astype(np.float32)

        # Estimate the new eigenvalue
        eigenvalue = np.dot(np.transpose(x_k), x_k_plus_one) / np.dot(np.transpose(x_k), x_k).astype(np.float32)

        # Reduce the size of the numbers in the vector for optimized calculations
        x_k_plus_one = x_k_plus_one / np.linalg.norm(x_k_plus_one, ord=1).astype(np.float32)

        # Check for convergence the first way
        if convergence_type == 1:
            if check_for_convergence_first(x_k_plus_one, x_k, tol):
                return eigenvalue, x_k.flatten(), iterations, True, "Success"

            # if i % 10 == 0:
            #     print(i)
            #     print(np.linalg.norm(x_k_plus_one - x_k, ord=1))

        # Check for convergence the second way
        elif convergence_type == 2:
            if check_for_convergence_second(x_k, matrix, eigenvalue, tol):
                return eigenvalue, x_k.flatten(), iterations, True, "Success"

            # if i % 10 == 0:
            #     print(i)
            #     print(np.linalg.norm(matrix.dot(x_k) - eigenvalue * x_k, ord=1))

        else:
            return -1, -1, -1, False, "Wrong convergence parameter."

        x_k = x_k_plus_one
        iterations += 1

    return -1, -1, -1, False, "Did not converge."


mat_file_path = 'web-Google.mat'

data = loadmat('web-Google.mat')
structured_array = data['Problem']
matrix = structured_array['A']

smallest_element = np.min(matrix[0][0])
print("Smallest element in matrix: ", smallest_element)


res_eigenvalue, res_eigenvector, res_iterations, success, message = power_method(matrix[0][0], 1000, 1e-7, 1)

if success:
    # Get the 5 largest elements
    indices_of_elements = np.argsort(res_eigenvector)[-5:]
    five_largest_elements = res_eigenvector[indices_of_elements]

    # Combine indices and elements into pairs (index, element)
    result = list(zip(indices_of_elements[::-1], five_largest_elements[::-1]))

    # Print the result
    print("5 largest elements:", result)
    print(f"Dominant Eigenvalue: {res_eigenvalue}")
    print("Number of iterations: ", res_iterations)
    print(f"Dominant Eigenvector: {res_eigenvector}")
else:
    print(message)
