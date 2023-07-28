import numpy as np
import matplotlib.pyplot as plt
import nfft as nfft

from scipy.stats import chi2

def analyse_res(mat,p=0.05):
    chi2thres = chi2.ppf(1-p,1)
    stdres = np.std(mat[mat != -9999].reshape(-1))
    is_out = np.zeros_like(mat,dtype="bool")

    for ii in range(mat.shape[1]-1):
        idx = mat[:,ii] != -9999
        is_out[idx,ii] = (mat[idx,ii] / stdres) ** 2 > chi2thres

    return is_out

def get_res_mat(mt,rw):
    res = np.zeros_like(mt,dtype="double")
    for ii in range(mt.shape[-1] -1):
        idx = rw[:,ii] != -9999
        res[idx,ii] = rw[idx,ii] - mt[idx,ii]

    return res

def plot_res(data,mt,outliers):
    x = np.arange(0,data.shape[0],1)
    sz = np.ones_like(x)

    for ii in range(data.shape[1]):
        idx = data[:,ii] != -9999
        idx_norm = np.logical_and(idx, np.invert(outliers[:,ii]))
        idx_outs = np.logical_and(idx,outliers[:,ii])
        plt.scatter(x[idx_norm],data[idx_norm,ii],sz[idx_norm],marker=",")
        plt.scatter(x[idx_outs],data[idx_outs,ii],sz[idx_outs],marker=",")
        plt.plot(x,mt[:,ii],color="tab:purple",alpha=0.75)
        plt.show()
        print("hello")
        plt.close()

mt = np.load("rec_mat.npy")
rw = np.load("data_raw.npy")

res = get_res_mat(mt,rw)
otlrs = analyse_res(res,p=0.05)

plot_res(rw,mt,otlrs)





