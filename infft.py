import pandas as pd
import numpy as np
import nfft as nfft
import matplotlib.pyplot as plt
import imageio

#(c) Universidad de Granada, Michael Sorochan Armstrong, Jos\'e Camacho 2023.

def nfft_inverse(x, y, N, w = 1, maxiter = 5000, eps = 1e-3, is_freq = False, is_verbose = True, create_gif = False):
    
    res = []
    nnz = np.count_nonzero(w)

    if is_freq:
        y = np.real(nfft.nfft(x,y) / nnz) #This cannot work because it's not a true inverse. So how can I do statistics in the freq domain?
    
    f = np.zeros(N,dtype="complex128")

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

def ensure_even(data_raw, Ln):
    if Ln % 2 != 0:
        last_elements = data_raw[-1,:]
        data_out = np.append(data_raw,[last_elements],axis=0)
        Ln += 1
    
    return data_out, Ln

def change_last_true_to_false(arr):
    arr = np.asarray(arr)
    indices = np.where(arr)[0]
    if len(indices) > 0:
        last_true_index = indices[-1]
        arr[last_true_index] = False
    return arr

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
#data_mean = data_raw.copy()
#data_mean[data_mean==-9999] = np.mean(data_mean[data_mean!=-9999])
data_freq = np.zeros_like(data_raw,dtype="complex64")
mni = np.zeros((df.shape[1]-1,1))
#data_raw, Ln = ensure_even(data_raw, Ln)
t = np.linspace(-0.5,0.5,Ln,endpoint=False)

idx = []

for ii in range(df.shape[1]-1):
    idx.append(data_raw[:,ii] != -9999)

    if sum(idx[ii]) % 2 != 0: #input must be even.
        idx[ii] = change_last_true_to_false(idx[ii])
    
    mni[ii] = np.mean(data_raw[idx[ii],ii])
    data_freq[idx[ii],ii] = nfft.nfft(t[idx[ii]],data_raw[idx[ii],ii] - mni[ii]) / sum(idx[ii])

U,S,V = np.linalg.svd(data_freq,full_matrices=False,compute_uv=True) #really, really slow.
k = 2
data_recon = np.outer((U[:,0:k] * S[0:k]), V[:,0:k].T)

f_hat = data_recon[:,0]

w = np.ones_like(f_hat)
w /= sum(w)

h_hat,res = nfft_inverse(t[idx[0]], f_hat, sum(idx[0]), w= 1, maxiter = 2000, eps=5e-8, is_freq=True, create_gif=True)

plt.plot(f_hat)
plt.plot(nfft.nfft(t[idx[0]],h_hat) / sum(idx[0]))
plt.show()
