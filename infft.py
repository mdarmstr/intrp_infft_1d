import pandas as pd
import numpy as np
import nfft as nfft
import matplotlib.pyplot as plt
import imageio

#(c) Universidad de Granada, Michael Sorochan Armstrong, Jos\'e Camacho 2023.

def nfft_inverse(x, y, N, w = 1, maxiter = 5000, eps = 1e-3, is_verbose = True, create_gif = False):
    
    res = []
    nnz = N
    f = np.zeros(N, dtype=np.complex128)

    r = y - nfft.nfft(x,f) / nnz   
    p = nfft.nfft_adjoint(x, r, N)
    r_norm = np.linalg.norm(r)
    r_norm_2 = 0
    iter = 0

    while np.abs(r_norm - r_norm_2) > eps:
        if iter > 0:
            r_norm = r_norm_2
        
        p_norm = np.linalg.norm(p * w)
        alpha = r_norm / p_norm
        f += alpha * w * p
        r = y - nfft.nfft(x, f) / nnz #When this was f, it was converging to a reasonable solution, contrary to the paper. Is it contrary to the paper?
        r_norm_2 = np.linalg.norm(r)
        beta = r_norm_2 / r_norm
        p = beta * p + nfft.nfft_adjoint(x, r, N)
        res.append(r_norm)
        
        if is_verbose == True: 
            print(iter, ' {:.5E}'.format(r_norm),' {:.5E}'.format(np.abs(r_norm-r_norm_2)))
        
        if create_gif == True:
            plt.plot(r,alpha=0.25)
            plt.plot(nfft.nfft(x,f) / len(x), alpha = 0.75)
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

    return f, res

def create_frame(iter):
    plt.savefig(f'./img/img_{iter}.png',transparent=False,facecolor='white')
    plt.close()

def change_last_true_to_false(arr):
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    return arr

#We need to perform the inverse transform first, before performing the reconstruction.
#np.save(file, arr, allow_pickle=True, fix_imports=True)

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
inverse_mat = np.zeros_like(data_raw,dtype="complex128")

mni = np.zeros((df.shape[1]-1,1))
#data_raw, Ln = ensure_even(data_raw, Ln)
t = np.linspace(-0.5,0.5,Ln,endpoint=False)

for ii in range(df.shape[0]-1):
    idx = data_raw[:,ii] != -9999
    if sum(idx) % 2 != 0:
        idx = change_last_true_to_false(idx)
        Ln = sum(idx)
    else:
        Ln = sum(idx)

    f_hat = data_raw[idx,ii] - np.mean(data_raw[idx,ii])
    x = t[idx]

    w = np.ones_like(f_hat)
    w /= sum(w)

    h_hat,res = nfft_inverse(x, f_hat, Ln, w = 1, maxiter = 2000, eps=5e-3, create_gif=False)
        
    inverse_mat[idx,ii] = h_hat
    print(ii)

np.save("inverse_mat.npy", inverse_mat)
print("finished")
    


