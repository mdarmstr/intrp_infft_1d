import numpy as np
import matplotlib.pyplot as plt
from intrp_infft_1d.intrp_infft_1d import *

j = 1000
N = 64
x = np.linspace(-0.5,0.5,j,endpoint=False)
idx = np.random.choice(a=[True,False],size=j)
w = sobk(N,1,2,1e-2)

y = np.sin(x*20) + np.random.rand(j)

#Can only handle even number of input observations
if sum(idx) % 2 != 0:
    idx = change_last_true_to_false(idx)

A = ndft_mat(x[idx],N) #convention has been reversed - this is equivalent to A @ A.H as described in the paper
AhA = A.H @ A

fk, _, _, _, = infft(x[idx],y[idx],N=N,AhA=AhA,w=w,return_adjoint=False)
fj_interp = adjoint(x,fk)

plt.scatter(x[idx],y[idx],s=0.1)
plt.plot(x,fj_interp)
plt.show()
