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



