import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from infftals import *

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

#Let's start with the random sampling - recover the data
dat = data_raw[:,34]

idx = dat != -9999
if sum(idx) % 2 != 0:
    idx = change_last_true_to_false(idx)
    Ln = sum(idx)
else:
    Ln = sum(idx)

dat_clean = dat[idx].copy()

#what's the full thing look like?
A1 = ndft_mat(t,N)
AhA1 = A1.H @ A1
ftot, _, _ = infft(t[idx], dat[idx] - np.mean(dat[idx]),N=N,AhA=AhA1,w=w)
ytot = nfft(t[idx],ftot) + np.mean(dat[idx])

ps = [0.1,0.2,0.3]

reps = 7

results_rand_corr = np.zeros((len(ps),reps))
results_rand_pred = np.zeros((len(ps),reps))

for ii in range(len(ps)):
    for jj in range(reps):
        bl_train = np.random.choice(a = [True,False], size=dat_clean.shape[0], p = [1-ps[ii],ps[ii]])
        bl_test = np.invert(bl_train)
        t2 = t[idx].copy()
        A = ndft_mat(t2[bl_train],N)
        AhA = A.H @ A
        f_pred, _, _ = infft(t2[bl_train], dat_clean[bl_train] - np.mean(dat_clean[bl_train]),N=N,AhA=AhA,w=w)
        y_recn = nfft(t[idx],f_pred) + np.mean(dat_clean[bl_train])
        results_rand_corr[ii,jj] = pearsonr(np.real(y_recn),np.real(ytot)).statistic
        results_rand_pred[ii,jj] = 1 - np.real(np.sum((y_recn[bl_test] - dat_clean[bl_test]) ** 2) / np.sum(dat_clean[bl_test] ** 2))
        print(jj)

corr_mean_rand = np.mean(results_rand_corr,axis=1)
corr_std_rand = np.std(results_rand_corr,axis=1)

pred_mean_rand = np.mean(results_rand_pred,axis=1)
pred_std_rand = np.std(results_rand_pred,axis=1)

print("random done")

#Random contiguous block missing
M = dat_clean.shape[0]

results_blck_corr = np.zeros((len(ps),reps))
results_blck_pred = np.zeros((len(ps),reps))

for ii in range(len(ps)):
    for jj in range(reps):
        k = np.ceil(ps[ii] * M)
        start = np.random.randint(0,M-(k-1))
        end = int(start + k)
        bl_train = np.ones(M,dtype=bool)
        bl_test = np.zeros(M,dtype=bool)

        bl_train[start:end] = False
        bl_test[start:end] = True

        t2 = t[idx].copy()
        A = ndft_mat(t2[bl_train],N)
        AhA = A.H @ A
        f_pred, _, _ = infft(t2[bl_train], dat_clean[bl_train] - np.mean(dat_clean[bl_train]),N=N,AhA=AhA,w=w)
        y_recn = nfft(t[idx],f_pred) + np.mean(dat_clean[bl_train])
        results_blck_corr[ii,jj] = pearsonr(np.real(y_recn),np.real(ytot)).statistic
        results_blck_pred[ii,jj] = 1 - np.real(np.sum((y_recn[bl_test] - dat_clean[bl_test]) ** 2) / np.sum(dat_clean[bl_test] ** 2))
        print(jj)

corr_mean_blck = np.mean(results_blck_corr,axis=1)
corr_std_blck = np.std(results_blck_corr,axis=1)

pred_mean_blck = np.mean(results_blck_pred,axis=1)
pred_std_blck = np.std(results_blck_pred,axis=1)

data = {
    'correlation_mean_random': corr_mean_rand,
    'correlation_stdv_random': corr_std_rand,
    'prediction_mean_random': pred_mean_rand,
    'prediction_stdv_random': pred_std_rand,
    'correlation_mean_blck': corr_mean_blck,
    'correlation_stdv_blck': corr_std_blck,
    'prediction_mean_blck': pred_mean_blck,
    'prediction_stdv_blck': pred_std_blck
}

df = pd.DataFrame(data)

df.to_csv('cross_val.csv',index=False)

print("all done")









