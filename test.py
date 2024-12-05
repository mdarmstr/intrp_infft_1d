import numpy as np
import matplotlib.pyplot as plt
from intrp_infft_1d.intrp_infft_1d import *
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


# Example Usage
# Spatial points (1D, irregularly sampled)
spatial_points = t[idx] # 44098 points in 1D

# Frequency points (1D)
frequency_points = np.linspace(-500, 500, 1024)  # 1024 frequency points

A = ndft_mat_nd(spatial_points, N)
print(A.conj().T @ A)

print(np.allclose(A.conj().T @ A, A1.conj().T @ A1, atol=1e-8))