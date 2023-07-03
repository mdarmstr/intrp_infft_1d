import pandas as pd
import numpy as np
import nfft as nfft
import matplotlib.pyplot as plt
import imageio

def nfft_inverse(x, y, N, w = 1, maxiter=100,eps=1e-3):
    res = []
    f = np.zeros(N, dtype=np.complex128)
    r = y - nfft.nfft(x, f) / len(x)
    p = nfft.nfft_adjoint(x, r, N)
    r_norm = np.linalg.norm(r * w)
    r_norm_2 = 1e-6
    iter = 1
    while np.abs(r_norm - r_norm_2) > eps:
        r_norm = r_norm_2
        p_norm = np.linalg.norm(p)
        alpha = r_norm / p_norm
        f += alpha * w * p
        r = y - nfft.nfft(x, f) / len(x)
        r_norm_2 = np.linalg.norm(r*w)
        beta = r_norm_2 / r_norm
        p = beta * p + nfft.nfft_adjoint(x, w * r, N)
        res.append(r_norm) 
        print(iter, ' {:.5E}'.format(r_norm),' {:.5E}'.format(r_norm-r_norm_2))
        iter += 1
        plt.plot(r)
        plt.plot(nfft.nfft(x,f) / len(x))
        plt.title(f'Iteration {iter}')
        create_frame(iter)
    return f, res

def create_frame(iter):
    plt.savefig(f'./img/img_{l}.png',transparent=False,facecolor='white')
    plt.close()

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
t = np.linspace(-0.5,0.5,Ln,endpoint=False)
fr = np.fft.fftfreq(Ln,d=smplR)
fr = np.fft.fftshift(fr).copy()

data = df.iloc[:,1].to_numpy()
f_hat = data[data != -9999].copy()
f_hat = f_hat[:-1].copy()

mn = np.mean(f_hat)

f_hat -= mn

x = (t - np.min(t)) / (np.max(t) - np.min(t)) -0.5
x = x[data != -9999].copy()
x = x[:-1].copy()

# N = 1024
# x = -0.5 + np.random.rand(N)
# f_hat = np.sin(10 * 2 * np.pi * x) + .1*np.random.randn( N ) #Add some  'y' randomness to the sample

# k = - N // 2 + np.arange(N)

#f = nfft.nfft(x, f_hat)
h_hat,res = nfft_inverse(x, f_hat, len(x), maxiter = 2000,eps=5e-3)

plt.plot(f_hat)
plt.plot(nfft.nfft(x,h_hat) / len(x))
plt.show()