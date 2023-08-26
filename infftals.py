import numpy as np
import pandas as pd
from nfft import nfft as nfft
from nfft import nfft_adjoint as adjoint
from scipy.linalg import dft
import matplotlib.pyplot as plt
import imageio

from scipy.stats import chi2
from scipy.linalg import lu

def ndft_mat(x,N):
    """non-equispaced discrete Fourier transform Matrix"""
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
    #w = (1/N**2) *(np.sin((N/2) * x) / np.sin(x/2)) ** 2
    w[x.shape[0] // 2] = 1
    
    # for n in np.arange(20,500,5):
    #     w1 = (np.divide(2*(1 + np.exp(-2 * np.pi * 1j * x)),n ** 2) * (np.sin((n/2) * np.pi * x) / np.sin(np.pi * x)) ** 2)
    #     w1[x.shape[0] // 2] = 1
    #     plt.plot(w1)
    # plt.show()
    
    #w /= sum(w)
    return w

def sobg(z,a,b,g):
    w = np.divide((0.25 - z ** 2) ** b, g + np.abs(z) ** (2 * a))
    c = sum(abs(w)) ** (-1)
    return w * c

def sobk(N,a,b,g):
    x = np.linspace(-1/2,1/2,N,endpoint=False)
    k = np.linspace(-N/2,N/2,N,endpoint=False)
    w = sobg(k/N,a,b,g)
    s = np.divide(1 + np.exp(-2 * np.pi * 1j * x),2 * np.sum(nfft(x,w))) * nfft(x,w) 
    #s[x.shape[0]//2] = 1
    return s

def nat_norm(f,w):
    return np.sum(abs(f)**2 / w)

def create_frame(iter):
    plt.savefig(f'./img/img_{iter}.png',transparent=False,facecolor='white')
    plt.close()

def infft(x, y, N, AhA,w=1, maxiter = 1, L=1, tol = 1e-16, is_verbose = True, create_gif=False):
    iter = 0
    res = []
    M = x.shape[0]
    r0 = y.copy()
    f = np.zeros(N,dtype = "complex128")
    L,U = lu(AhA,permute_l=True)

    while iter < maxiter: #np.abs(rnm1 - rnm2) / rnm1 > tol and 
        
        #f += np.divide(adjoint(x,r0,N)*w,(M*w + 1))
        f += adjoint(x,r0,N) @ (np.diag(w) - np.diag(w) @ L @ np.linalg.pinv(np.eye(N) + U @ np.diag(w) @ L) @ U @ np.diag(w))
        r1 = r0 - nfft(x,f*w)
        r0 = r1.copy()
        res.append(r0)
        iter += 1

        if is_verbose == True: 
            print(iter, ' {:.5E}'.format(np.sum(r0 ** 2)))

    return f, r0, res

# ## This part is just reading the data
# df = pd.read_csv('T.Suelo.csv')
# Ln = df.shape[0]
# smplR = 1800
# data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
# inverse_mat = np.zeros_like(data_raw,dtype="complex128")
# residue_mat = np.zeros_like(data_raw,dtype="float64")
# rec_mat = np.zeros_like(data_raw,dtype="float64")
# mni = np.zeros((df.shape[1]-1,1))
# #data_raw, Ln = ensure_even(data_raw, Ln)
# N = 1024
# t = np.linspace(-0.5,0.5,Ln,endpoint=False)
# tf = np.linspace(-0.5,0.5,N,endpoint=False)
# mn = []
# inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
# #w = fjr(N)
# w = sobk(N,1,2,1e-2)

# for ii in range(df.shape[1]-1):
#     idx = data_raw[:,ii] != -9999
#     if sum(idx) % 2 != 0:
#         idx = change_last_true_to_false(idx)
#         Ln = sum(idx)
#     else:
#         Ln = sum(idx)

#     x = t[idx]
#     mn.append(np.mean(data_raw[idx,ii]))
#     A = ndft_mat(x,N)
#     AhA = A.H @ A
    
#     f0 = adjoint(x,(data_raw[idx,ii] - mn[ii]), N)

#     h_hat, r, res = infft(x, data_raw[idx,ii] - mn[ii], N, AhA, w = w, maxiter = 1, tol = 5e-16, create_gif=False)
#     y_pred = nfft(t,h_hat) + mn[ii]
#     fit = np.sum((y_pred[idx] - data_raw[idx,ii])**2) / np.sum((data_raw[idx,ii])**2)
#     print(fit)

#     plt.plot(t,nfft(t,h_hat) + mn[ii],color='orange')
#     plt.scatter(x,data_raw[idx,ii],s = np.ones(len(x))*0.05)
#     plt.xlabel("Normalized time values $t \in [-0.5,0.5)$")
#     plt.ylabel("Temperature ($^\circ$C)")
#     plt.title("Interpolative iNFFT on irregularly sampled remote sensor data")
#     plt.show()

#     inverse_mat[:,ii] = h_hat
#     plt.close()
#     print(ii)

# print("inverse NFFTs complete")
# #k = 10
# #mnk = np.mean(inverse_mat,axis=0)
# #U, S, V = np.linalg.svd(inverse_mat - mnk)
# #rec_inv = (U[:,:k]*S[:k]) @ V[:,:k].T + mnk
# rec_inv = inverse_mat.copy()

# # Reconstructing the data, measuring the residuals
# for ii in range(df.shape[1]-1):
#     idx = data_raw[:,ii] != -9999
    
#     rec_mat[:,ii] = np.real(nfft(t, rec_inv[:,ii]) / t.shape[0] + mn[ii])
#     residue_mat[idx,ii] = rec_mat[idx,ii] - data_raw[idx,ii]
#     print(ii)

# np.save("inverse_mat.npy", inverse_mat)
# np.save("inverse_res.npy",residue_mat)
# np.save("data_raw.npy",data_raw)
# np.save("rec_mat.npy",rec_mat)
# print("finished")

# # y = data_raw[:,0]
# # idx = y != -9999

# # idx = change_last_true_to_false(idx)
# # #Ln = sum(idx)
# # x = t[idx]

# # N = 48
# # w = fjr(np.linspace(-N/2, N/2,N,endpoint=False),N)

# # mn = np.mean(y[idx])

# # f0 = adjoint(x,y[idx]-mn, N)

# # f, r, res = infft(x, y[idx] - mn, N, w = 1, f0=f0, maxiter = 1500, tol = 5e-8, create_gif=True) 



