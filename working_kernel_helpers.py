"""
This code was implemented by gjtrowbridge:
https://github.com/gjtrowbridge/simple-mkl-python/blob/master/helpers.py
"""
import numpy as np

#Gets an array of all kernel matrices for each kernel
def get_all_kernels(X, kernel_functions):
    n = X.shape[0]
    M = len(kernel_functions)

    #Initialize array that will store all the kernels
    kernel_matrices = []

    #Loops through all kernel functions
    for m in range(M):
        kernel_func = kernel_functions[m]
        kernel_matrices.append(np.empty((n, n)))

        #Creates kernel matrix
        for i in range(n):
            for j in range(n):
                kernel_matrices[m][i, j] = kernel_func(X[i], X[j])

    #Returns all kernel matrices
    return kernel_matrices

#Gets the weighted combined kernel matrix entries
def get_combined_kernel(kernel_matrices, weights):
    M = len(kernel_matrices)
    n = kernel_matrices[0].shape[0]

    combined_kernel_matrix = np.zeros((n, n))
    #sum up values from different kernels with their weights
    for m in range(M):
        combined_kernel_matrix += kernel_matrices[m] * weights[m] *1.0

    return combined_kernel_matrix

def get_combined_kernel_function(kernel_functions, weights):
    M = len(kernel_functions)
    def combined_kernel(u, v):
        result = 0
        for m in range(M):
            result += kernel_functions[m](u, v) * weights[m]
        return result

    return combined_kernel

#Four commonly used kernels for features defined below
def create_linear_kernel(u, v):
    """ Returns inner product of u and v. """

    return np.inner(u, v)


def create_poly_kernel(degree, gamma, intercept=0.0):
    """ Returns polynomial kernel of specified degree and coeff gamma. """

    def poly_kernel_func(u, v):
        return (gamma*np.inner(u, v) + intercept) ** degree

    return poly_kernel_func


def create_rbf_kernel(gamma):
    """ Returns the gaussian/rbf kernel with specified gamma. """

    def rbf_kernel_func(u, v):
        tmp=np.linalg.norm(u-v)**2
        #print tmp
        tmp/=(2*gamma**2)
        #print tmp
        tmp*=-1
        #print tmp
        return np.exp(tmp)
        #return np.exp(-gamma*np.sum(np.abs(u - v) ** 2))

    return rbf_kernel_func


def create_sigmoid_kernel(gamma, intercept=0.0):
    """ Returns the sigmoid/tanh kernel with specified gamma. """

    def sigmoid_kernel_func(u, v):
        return np.arctan(gamma*np.inner(u, v) + intercept)

    return sigmoid_kernel_func
