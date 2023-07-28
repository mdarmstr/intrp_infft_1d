import numpy as np
import pandas as pd
from nfft import nfft as nfft
from nfft import nfft_adjoint as adjoint
import scipy.optimize
import matplotlib.pyplot as plt
import imageio

from scipy.stats import chi2

def change_last_true_to_false(arr):
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    return arr

def fjr(x,N):
    w = (np.divide(2*(1 + np.exp(-2 * np.pi * 1j * x)),N ** 2) * (np.sin((N/2) * np.pi * x) / np.sin(np.pi * x)) ** 2)
    w[N // 2] = 1
    return w

def nat_norm(f,w):
    return np.sum(abs(f)**2 / w)

def create_frame(iter):
    plt.savefig(f'./img/img_{iter}.png',transparent=False,facecolor='white')
    plt.close()

def infft(x, y, N, w=1, f0 = None, maxiter = 10, L=10, tol = 1e-16, is_verbose = True, create_gif=False):
    
    res = []
    nnz = x.shape[0]

    if f0 is None:
        f = np.zeros(N,dtype=np.complex128)
    else:
        f = f0.copy()
    
    r = y - nfft(x, f) / nnz
    p = adjoint(x,r,N) 

    rnm1 = np.linalg.norm(r) ** 2
    rnm2 = 0
    iter = 0

    while np.abs(rnm1 - rnm2) / rnm1 > tol and iter < maxiter:
        if iter > 0:
            rnm1 = rnm2
        
        pnm = np.linalg.norm(p * w) ** 2
        alf = rnm1 / pnm
        f += alf * w * p
        r = y - nfft(x, f) / nnz
        rnm2 = np.linalg.norm(r) ** 2
        bta = rnm2 / rnm1
        p = bta * p + adjoint(x,r, N)
        #nat2 = np.linalg.norm((f - f0) * w)

        res.append(rnm2)

        if is_verbose == True: 
            print(iter, ' {:.5E}'.format(rnm2),' {:.5E}'.format((rnm1 - rnm2)/rnm1))
        
        if create_gif == True:
            plt.plot(r,alpha=0.25)
            plt.plot(nfft(x,f) / len(x), alpha = 0.75)
            plt.ylim((min(y),max(y)))
            plt.title(f'iNFFT Iteration {iter}')
            create_frame(iter)

        iter += 1

    if create_gif == True:
            frames = []

            for t in range(iter):
                image = imageio.v2.imread(f'./img/img_{t}.png')
                frames.append(image)
        
            imageio.mimsave('./output.gif', frames, duration = 0.01)

    return f, r, res

## This part is just reading the data
df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
inverse_mat = np.zeros_like(data_raw,dtype="complex128")
residue_mat = np.zeros_like(data_raw,dtype="float64")
rec_mat = np.zeros_like(data_raw,dtype="float64")
mni = np.zeros((df.shape[1]-1,1))
#data_raw, Ln = ensure_even(data_raw, Ln)
t = np.linspace(-0.5,0.5,Ln,endpoint=False)
mn = []
N = 500
inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")

for ii in range(df.shape[1]-1):
    idx = data_raw[:,ii] != -9999
    if sum(idx) % 2 != 0:
        idx = change_last_true_to_false(idx)
        Ln = sum(idx)
    else:
        Ln = sum(idx)

    x = t[idx]
    mn.append(np.mean(data_raw[idx,ii]))
    
    f0 = adjoint(x,data_raw[idx,ii] - mn[ii], N)

    h_hat, r, res = infft(x, data_raw[idx,ii] - mn[ii], N, f0=f0, w = 1, maxiter = 50, tol = 1e-7, create_gif=False)
        
    inverse_mat[:,ii] = h_hat
    print(ii)

print("inverse NFFTs complete")
#k = 10
#mnk = np.mean(inverse_mat,axis=0)
#U, S, V = np.linalg.svd(inverse_mat - mnk)
#rec_inv = (U[:,:k]*S[:k]) @ V[:,:k].T + mnk
rec_inv = inverse_mat.copy()

# Reconstructing the data, measuring the residuals
for ii in range(df.shape[1]-1):
    idx = data_raw[:,ii] != -9999
    
    rec_mat[:,ii] = np.real(nfft(t, rec_inv[:,ii]) / t.shape[0] + mn[ii])
    residue_mat[idx,ii] = rec_mat[idx,ii] - data_raw[idx,ii]
    print(ii)

np.save("inverse_mat.npy", inverse_mat)
np.save("inverse_res.npy",residue_mat)
np.save("data_raw.npy",data_raw)
np.save("rec_mat.npy",rec_mat)
print("finished")

# y = data_raw[:,0]
# idx = y != -9999

# idx = change_last_true_to_false(idx)
# #Ln = sum(idx)
# x = t[idx]

# N = 48
# w = fjr(np.linspace(-N/2, N/2,N,endpoint=False),N)

# mn = np.mean(y[idx])

# f0 = adjoint(x,y[idx]-mn, N)

# f, r, res = infft(x, y[idx] - mn, N, w = 1, f0=f0, maxiter = 1500, tol = 5e-8, create_gif=True) 



