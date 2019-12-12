import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import random

np.random.seed(42) 


def epsilon_similarity_graph(X: np.ndarray, sigma=1, epsilon=0.01):
    """ X (n x d): coordinates of the n data points in R^d.
        sigma (float): width of the kernel
        epsilon (float): threshold
        Return:
        adjacency (n x n ndarray): adjacency matrix of the graph.
    """
    dist = np.array(squareform(pdist(X)))
    adjacency = np.exp(-dist**2/(2*sigma**2))
    adjacency[adjacency <= epsilon] = 0
    
    return adjacency

def compute_laplacian(adjacency: np.ndarray, normalize: bool):
    """ Return:
        L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
    """
    degree = np.sum(adjacency, axis=1)
    degree_matrix = np.diag(degree)
    laplacian = degree_matrix - adjacency
    
    if normalize:
        degree_sqrt = np.power(degree, -1/2)
        degree_sqrt_matrix = np.diag(degree_sqrt)
        laplacian = degree_sqrt_matrix @ laplacian @ degree_sqrt_matrix
        
    return laplacian

def spectral_decomposition(laplacian: np.ndarray):
    """ Return:
        lamb (np.array): eigenvalues of the Laplacian
        U (np.ndarray): corresponding eigenvectors.
    """    
    # As we know that the Laplacian is real symmetric matrix, we can use eigh from np.linalg.
    return np.linalg.eigh(laplacian)


def laplacian_eigenmaps(X:np.ndarray, dim: int, sigma: float, epsilon: float, normalize: bool, use_similarity=False):
    """ Return:
        coords (n x dim array): new coordinates for the data points."""
    if use_similarity:
        adjacency = epsilon_similarity_graph(X, sigma, epsilon)
    else:
        adjacency = X
    laplacian = compute_laplacian(adjacency, normalize)
    _, U = spectral_decomposition(laplacian)
    U = U[:,1:dim+1]
    return U

#proj = laplacian_eigenmaps(X_mnist, dim, sigma=8, epsilon=0.01, normalize=True)
#np.save('spectral_clustering_embeddings.npy',proj)
