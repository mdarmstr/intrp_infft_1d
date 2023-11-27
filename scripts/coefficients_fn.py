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

test_gma = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]
test_fre = [16,32,64,128,256,512,1024,2048]

dat = data_raw[:,10]

idx = dat != -9999
if sum(idx) % 2 != 0:
    idx = change_last_true_to_false(idx)

results_mat = np.zeros((len(test_gma),len(test_fre)))

for ii in range(len(test_gma)):
    for jj in range(len(test_fre)):
        
        N = test_fre[jj]
        gma = test_gma[ii]
        w = sobk(N,1,2,gma)
        
        A = ndft_mat(t[idx],N)
        AhA = A.H @ A
        f_pred, _, _, res_rel = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA,w=w,return_adjoint=True)
        results_mat[ii,jj] = 1 - res_rel
        print(f'gma {ii} and N {jj} complete')

plt.imshow(results_mat,origin='upper')
plt.yticks(ticks=np.arange(len(test_gma)), labels=np.log10(test_gma))
plt.xticks(ticks=np.arange(len(test_fre)), labels=np.log2(test_fre))

plt.xlabel('$log_2N$')
plt.ylabel('$log_{10}\gamma$')
plt.title('Reconstruction accuracy')
plt.savefig('int_coeff_gamma.png')

dat_clean = dat[idx].copy()

#Cross-validation
M = dat_clean.shape[0]
k = np.ceil(0.2 * M)
start = np.random.randint(0,M-(k-1))
end = int(start + k)
bl_train = np.ones(M,dtype=bool)
bl_test = np.zeros(M,dtype=bool)

bl_train[start:end] = False
bl_test[start:end] = True

t2 = t[idx].copy()

for ii in range(len(test_gma)):
    for jj in range(len(test_fre)):
        
        N = test_fre[jj]
        gma = test_gma[ii]
        w = sobk(N,1,2,gma)
        
        A = ndft_mat(t2[bl_train],N)
        AhA = A.H @ A
        f_pred, _, _, _ = infft(t2[bl_train], dat_clean[bl_train] - np.mean(dat_clean[bl_train]),N=N,AhA=AhA,w=w)
        y_recn = adjoint(t[idx],f_pred) + np.mean(dat_clean[bl_train]) #reconstruction only evaluated on the artificially removed data
        results_mat[ii,jj] = 1 - len(bl_test)**(-1) * np.sum(np.abs(np.divide(y_recn[bl_test] - dat_clean[bl_test],dat_clean[bl_test])))
        print(f'gma {ii} and N {jj} complete')
        

plt.imshow(results_mat,origin='upper')
plt.yticks(ticks=np.arange(len(test_gma)), labels=np.log10(test_gma))
plt.xticks(ticks=np.arange(len(test_fre)), labels=np.log2(test_fre))

plt.xlabel('$log_2N$')
plt.ylabel('$log_{10}\gamma$')
plt.title('External reconstruction accuracy, $20\%$ contiguous block')
plt.savefig('ext_coeff_gamma.png')


