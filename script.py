import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from infft import *

df = pd.read_csv('T.Suelo.csv')
Ln = df.shape[0]
smplR = 1800
data_raw = df.iloc[0:,1:].to_numpy() #keep the missing values
inverse_mat = np.zeros_like(data_raw,dtype="complex128")
residue_mat = np.zeros_like(data_raw,dtype="float64")
rec_mat = np.zeros_like(data_raw,dtype="float64")
mni = np.zeros((df.shape[1]-1,1))
#data_raw, Ln = ensure_even(data_raw, Ln)
N = 1024
t = np.linspace(-0.5,0.5,Ln,endpoint=False)
tf = np.linspace(-0.5,0.5,N,endpoint=False)
mn = []
inverse_mat = np.zeros((N,df.shape[1]-1),dtype="complex128")
#w = fjr(N)
w = sobk(N,1,2,1e-2)

for ii in range(df.shape[1]-1):
    idx = data_raw[:,ii] != -9999
    if sum(idx) % 2 != 0:
        idx = change_last_true_to_false(idx)
        Ln = sum(idx)
    else:
        Ln = sum(idx)

    x = t[idx]
    mn.append(np.mean(data_raw[idx,ii]))
    A = ndft_mat(x,N)
    AhA = A.H @ A
    
    f0 = nfft(x,(data_raw[idx,ii] - mn[ii]), N)

    fk, fj, res_abs, res_rel = infft(x, data_raw[idx,ii] - mn[ii], N, AhA, w = w)
    y_pred = adjoint(t,fk) + mn[ii]
    fit = np.sum((y_pred[idx] - data_raw[idx,ii])**2) / np.sum((data_raw[idx,ii])**2)
    print(fit)

    plt.plot(t,adjoint(t,fk) + mn[ii],color='orange')
    plt.scatter(x,data_raw[idx,ii],s = np.ones(len(x))*0.05)
    plt.xlabel("Normalized time values $t \in [-0.5,0.5)$")
    plt.ylabel("Temperature ($^\circ$C)")
    plt.title("Interpolative iNFFT on irregularly sampled remote sensor data")
    plt.show()

    inverse_mat[:,ii] = fk + mn[ii]
    plt.close()
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