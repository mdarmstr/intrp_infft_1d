import pandas as pd
import numpy as np
import nfft as nfft
import matplotlib.pyplot as plt

def nfft_inverse(x, y, N, w = 1, L=100):
    res = []
    f = np.zeros(N, dtype=np.complex128)
    r = y - nfft.nfft(x, f)
    p = nfft.nfft_adjoint(x, r, N)
    r_norm = np.sum(abs(r)**2 * w)
    for l in range(L):
        p_norm = np.sum(abs(p)**2 * w)
        alpha = r_norm / p_norm
        f += alpha * w * p
        r = y - nfft.nfft(x, f)
        r_norm_2 = np.sum(abs(r)**2 * w)
        beta = r_norm_2 / r_norm
        p = beta * p + nfft.nfft_adjoint(x, w * r, N)
        r_norm = r_norm_2
        #res.append(r_norm.copy()) 
        print(l, r_norm)
    return f, res

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
t = np.arange(0,Ln*smplR,smplR)

data = df.iloc[:,1].to_numpy()
f_hat = data[data != -9999].copy()
f_hat = f_hat[:-1].copy()

mn = np.mean(f_hat)

x = t[data != -9999].copy()
x = x[:-1].copy()

x = (x - np.min(x)) / (np.max(x) - np.min(x)) -0.5

# x = -0.5 + np.random.rand(10000)

# # define Fourier coefficients
# N = 10000
# k = - N // 2 + np.arange(N)
# f_hat = np.random.randn(N)

f = nfft.nfft(x, f_hat)
h_hat,res = nfft_inverse(x, f, len(x), L = 500)

plt.plot(f_hat)
plt.plot(np.real(h_hat) + mn)
plt.show()