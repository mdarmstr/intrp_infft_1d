import pandas as pd
import numpy as np
import nfft as nfft
import matplotlib.pyplot as plt
import imageio


def nfft_inverse(x, y, N, w = 1, L=500):
    res = []
    f = np.zeros(N, dtype=np.complex128)
    r = y - nfft.nfft(x, f) / len(x)
    p = nfft.nfft_adjoint(x, r, N)
    r_norm = np.linalg.norm(r)
    for l in range(L):
        p_norm = np.linalg.norm(p*w)
        alpha = r_norm / p_norm
        f += alpha * w * p
        r = y - nfft.nfft(x, f) / len(x)
        r_norm_2 = np.linalg.norm(r)
        beta = r_norm_2 / r_norm
        p = beta * p + nfft.nfft_adjoint(x, w * r, N)
        r_norm = r_norm_2
        res.append(r_norm.copy()) 
        print(l, r_norm)
        plt.plot(r)
        plt.plot(nfft.nfft(x,f) / len(x))
        plt.title(f'Iteration {l}')
        create_frame(l)
    return f, res

def create_frame(l):
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

x = t[data != -9999].copy()
x = x[:-1].copy()

#x = np.fft.fftshift(x) - Ln

# N = 1024
# x = -0.5 + np.random.rand(N)
# f_hat = np.sin(10 * 2 * np.pi * x) + .1*np.random.randn( N ) #Add some  'y' randomness to the sample

# k = - N // 2 + np.arange(N)

#f = nfft.nfft_adjoint(x, f_hat,Ln,truncated=False)
h_hat,res = nfft_inverse(x, f_hat - mn, len(x), L = 1000)

frames = []
for t in range(len(res)):
    image = imageio.v2.imread(f'./img/img_{t}.png')
    frames.append(image)

imageio.mimsave('./example.gif',frames,duration=5)

# plt.scatter(x,f_hat)
# plt.scatter(x,np.real(h_hat))
# plt.show()