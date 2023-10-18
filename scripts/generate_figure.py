import numpy as np
import matplotlib.pyplot as plt
from infft import *
plt.style.use('tableau-colorblind10')

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
inverse_mat = np.zeros_like(data_raw,dtype="complex128")
residue_mat = np.zeros_like(data_raw,dtype="float64")
rec_mat = np.zeros_like(data_raw,dtype="float64")
mni = np.zeros((df.shape[1]-1,1))

N = 1024
t = np.linspace(-0.5,0.5,Ln,endpoint=False)
inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
#w = fjr(N)
w = sobk(N,1,2,1e-2)

dat = data_raw[:,0]

idx = dat != -9999
if sum(idx) % 2 != 0:
    idx = change_last_true_to_false(idx)


dat_clean = dat[idx].copy()

A1 = ndft_mat(t[idx],N)
AhA1 = A1.H @ A1
ftot, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w)
ytot = adjoint(t,ftot) + np.mean(dat[idx])

ax = plt.gca()

#Scatter plot of observed data
ax.scatter(t[idx],dat[idx],s=0.1,c='k')

#Scatter plot of iNFFT
infft_line, = ax.plot(t,ytot,c='C4',label='iNFFT')

#Scatter plot of truncated iFFT
fapprox, _, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w,approx=True)
yapp = adjoint(t,fapprox) + np.mean(dat[idx])

ifft_line, = ax.plot(t,yapp,c='C1',label='iFFT')
ax.legend(handles=[infft_line,ifft_line])

plt.xlabel("Normalized time values $t \in [-0.5,0.5)$")
plt.ylabel("Temperature ($^\circ$C)")
plt.title("Interpolative iNFFT on irregularly sampled remote sensor data")
plt.savefig('result_figure.png')

print('MAPE: iNFFT')
print(len(idx)**(-1) * np.sum(np.abs(np.divide(ytot[idx] - dat[idx],dat[idx]))))

print('MAPE: iFFT')
print(len(idx)**(-1) * np.sum(np.abs(np.divide(yapp[idx] - dat[idx],dat[idx]))))
