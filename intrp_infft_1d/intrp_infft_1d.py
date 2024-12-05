import numpy as np
import pandas as pd
from nfft import nfft as adjoint #CHANGE IN CONVENTION
from nfft import nfft_adjoint as nfft #CHANGE IN CONVENTION
from scipy.linalg import lu

def ndft_mat(x,N):
    #non-equispaced discrete Fourier transform Matrix
    k = -(N // 2) + np.arange(N)
   
    return np.asmatrix(np.exp(2j * np.pi * np.outer(k,x[:,np.newaxis])).T)

def change_last_true_to_false(arr):
    
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    
    return arr

def fjr(N):
    
    x = np.linspace(-1/2,1/2,N,endpoint=False)
    w = ((np.sin((N/2) * np.pi * x) / np.sin(np.pi * x)) ** 2) * np.divide(2*(1 + np.exp(-2 * np.pi * 1j * x)),N ** 2)
    w[x.shape[0] // 2] = 1
    
    return w

def sobg(z,a,b,g):
    
    w = np.divide((0.25 - z ** 2) ** b, g + np.abs(z) ** (2 * a))
    c = sum(abs(w)) ** (-1)
    
    return w * c

def sobk(N,a,b,g):
    
    x = np.linspace(-1/2,1/2,N,endpoint=False)
    k = np.linspace(-N/2,N/2,N,endpoint=False)
    w = sobg(k/N,a,b,g)
    s = np.divide(1 + np.exp(-2 * np.pi * 1j * x),2 * np.sum(adjoint(x,w))) * adjoint(x,w) 
    #s[x.shape[0]//2] = 1
    
    return s

def infft(x, y, N, AhA=None, w=None, return_adjoint=False, approx=False):
    
    if w is None:
        w = np.ones(N) / N
        Warning("No weight function input; normalized uniform weight for all frequencies")

    if AhA is None and approx == False:
        A = ndft_mat(x,N)
        AhA = A.H @ A
        Warning("No self-adjoint matrix specified; calculating based on input observations")
    
    if approx == False:
        L,U = lu(AhA,permute_l=True)
        fk = nfft(x,y,N) @ (np.diag(w) - np.diag(w) @ L @ np.linalg.pinv(np.eye(N) + U @ np.diag(w) @ L) @ U @ np.diag(w))
    else:
        fk = (nfft(x,y,N) @ np.diag(w)) @ np.linalg.pinv(len(x) * np.diag(w) + np.eye(N))
    
    if return_adjoint == True:
        fj = np.real(adjoint(x,fk))
        res_abs = np.sum(np.abs(y - fj) ** 2)
        res_rel = res_abs / np.sum(y ** 2)
    else:
        fj = None
        res_abs = None
        res_rel = None

    return fk, fj, res_abs, res_rel

def ndft_mat_nd(spatial_points, num_frequencies_per_dim):
    """
    Constructs the non-equispaced discrete Fourier transform (NDFT) matrix for N dimensions using matmul.

    Parameters:
        spatial_points (np.ndarray): Spatial points, shape (M, D) or (M,) for 1D.
        num_frequencies_per_dim (int or list[int]): Number of frequency points per dimension.
                                                    Can be an integer (1D) or list for N-D.

    Returns:
        np.ndarray: Transformation matrix A of shape (M, total_frequencies).
    """
    # Ensure spatial_points is a numpy array
    spatial_points = np.asarray(spatial_points)

    # Handle 1D case: If num_frequencies_per_dim is an integer, convert it to a list
    if isinstance(num_frequencies_per_dim, int):
        num_frequencies_per_dim = [num_frequencies_per_dim]

    # Handle the case where spatial_points is 1D (reshape to M x 1 for N-D compatibility)
    if spatial_points.ndim == 1:
        spatial_points = spatial_points[:, np.newaxis]  # Shape (M, 1)

    # Extract dimensions
    M, D = spatial_points.shape  # M: Number of spatial points, D: Dimensions
    if len(num_frequencies_per_dim) != D:
        raise ValueError("Length of num_frequencies_per_dim must match the dimensionality of spatial_points.")

    # Generate frequency points for each dimension
    frequency_grids = [-(N // 2) + np.arange(N) for N in num_frequencies_per_dim]
    if D == 1:  # Special case for 1D
        frequency_points = frequency_grids[0][:, np.newaxis]  # Shape (N, 1)
    else:
        frequency_points = np.array(np.meshgrid(*frequency_grids, indexing="ij")).reshape(D, -1).T  # Shape (total_frequencies, D)
    
    # Compute the transformation matrix
    # Outer product in N dimensions -> dot product between spatial and frequency points using matmul
    phase_matrix = 2j * np.pi * np.matmul(spatial_points, frequency_points.T)  # Shape (M, total_frequencies)
    transformation_matrix = np.exp(phase_matrix)
    
    return transformation_matrix



