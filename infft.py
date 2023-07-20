import numpy as np
import pandas as pd
from nfft import nfft as nfft
from nfft import nfft_adjoint as adjoint
import scipy.optimize
import matplotlib.pyplot as plt

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

def infft(x, y, N, w=1, f0 = None, maxiter = 10, L=10, tol = 1e-16, is_verbose = True):
    
    res = []

    if f0 is None:
        f = np.zeros(N,dtype=np.complex128)
    else:
        f = f0.copy()
    
    r = y - nfft(x, f) / N
    p = adjoint(x,r,N) 

    rnm1 = np.linalg.norm(r) ** 2
    rnm2 = 0
    nat1 = np.linalg.norm(f*w)
    nat2 = 0
    iter = 0

    while np.abs(nat1 - nat2) > tol and iter < maxiter:
        if iter > 0:
            nat1 = nat2
        
        pnm = np.linalg.norm(p * w) ** 2
        alf = rnm1 / pnm
        f += alf * w * p
        r = y - nfft(x, f) / N
        rnm2 = np.linalg.norm(r) ** 2
        bta = rnm2 / rnm1
        p = bta * p + adjoint(x,r, N)
        nat2 = np.linalg.norm((f - f0) * w)

        res.append(nat2)

        if is_verbose == True: 
            print(iter, ' {:.5E}'.format(nat2),' {:.5E}'.format(nat1 - nat2))

        iter += 1

    return f, r, res

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

y = data_raw[:,0]
idx = y != -9999

idx = change_last_true_to_false(idx)
Ln = sum(idx)
x = t[idx]

N = 256
w = fjr(np.linspace(-N/2, N/2,N,endpoint=False),N)

mn = np.mean(y[idx])

f0 = adjoint(x,y[idx], N)

f, r, res = infft(x, y[idx], N, w = w, f0=f0, maxiter = 15, tol = 1e-8) 

#plt.plot(nfft(x,f) / Ln + mn)
plt.plot(x,y[idx])
#plt.plot(res)
plt.plot(t, nfft(t,f) / Ln)
#plt.plot(f)

plt.show()


