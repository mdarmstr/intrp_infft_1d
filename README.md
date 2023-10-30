# Inverse Interpolative Non-Uniform Fast Fourier Transform

The type-I Discrete Non-Uniform Transform (NDFT) is defined as the following operation:
$$h_k = \sum_{M/2-1}^{M/2} f(x_j) e^{-2\pi\mathbf{i}k\frac{x_j}{M}}$$

Where $h_k$ refers to a uniform set of Fourier coefficients, for a specified number $N$ of trigonometric polynomials. $f(x_j)$ refers to obsevational data in the time domain taken at irregularly sampled points: $x_j$ which are measured in the domain $[-\frac{M}{2}, \frac{M}{2})$

Which can be performed in $\mathcal{O}(N log N + |log (1/\epsilon) | M)$ complexity using the Non-Uniform Fast Fourier Transform described by Potts et al (Potts, Daniel, Gabriele Steidl, and Manfred Tasche. "Fast Fourier transforms for nonequispaced data: A tutorial." Modern Sampling Theory: Mathematics and Applications (2001): 247-270.). Note that for consistency with common convention, we define the forward transform as above, and the adjoint via a sign change in the exponential term. This is opposite to the convention utilised by Potts in the mathematical body of literature.

Unlike the Fast Fourier Transform and its inverse in the equidistant case, the normalisation for an inverse transform is not known explicitly and relies on the implicit inverse of the self-adjoint product of the forward and adjoint transformation matices: $AA^H$. In order to calculate a series of Fourier coefficients that is invertible via its adjoint product, the coefficients must account for the inverse of $AA^H$. This is done using the LU decomposition of the inverse product, via a weighted filter to avoid fitting to a discontinuous function. 

Formally, this package performs the inverse adjoint non-uniform fast fourier transform as defined by our convention via the minimisation of the cost function:

$${argmin}_{\hat{h}_k}||f(x_j) - A^H\hat{h}_k||_W$$

## Instructions for interpolation

A basic use case can be found in `scripts/basic_usage.py`

### Inputs
To use the infft function, requires at minimum 3 inputs: 
- `x` as a 1-dimensional numpy array with length $j$ containing discontinuous time series data normalized such that $-0.5 \leq x < 0.5$ 
- `y` containing the responses at each time point in `x` of length $j$
- `N` as an integer value representing the total number of $k$ Fourier coefficients

and the optional inputs:
- `AhA` the self-adjoint product of the non-uniform discrete Fourier transform matrix (see: `scripts/basic_usage.py`)
- `w` a $k$ dimensional numpy array indicating the weight function.

The `return_adjoint` argument will return the calculated values **at the points indexed by `x`**, and will not interpolate missing values.

If `approx=True` the inverse transform will assume uniformity in the data, on average.

### Outputs
- `fk` the $N$ weighted Fourier coefficients
- `fj` if `return_adjoint = True`, then the reconstructed data at the originally measured points
- `res_abs`: absolute squared error
- `res_rel`: relative squared error

## Brief descriptions of functions

- `ndft_mat(x,N)`: used to generate non-uniform self-adjoint matrix. Input `x` as discontinuous numpy array of length $j$, and `N` as the number of Fourier coefficients. From: [dependency](https://github.com/jakevdp/nfft)[^1].
- `change_last_true_to_false(arr)`: changes the last entry of a 1-dimensional boolean array `arr` to `False`.
- `fjr(N)`: generates the modified Fejer kernel for use as a weight function for `N` input Fourier Coefficients.
- `sobg(z,a,b,g)`: subroutine for `sobk`
- `sobk(N,a,b,g)`: generates the modified Sobolev kernel for user inputs `a`, `b`, and `g`. See: [reference](https://www-user.tu-chemnitz.de/~potts/paper/potts_kunis04.pdf)[^2]
- `infft(x,y,N,AhA=None,w=None,return_adjoint=None,approx=False)`: the interpolative non-uniform fast fourier transform. See **Instructions for interpolation** for usage.
- `adjoint(x,k)`: equivalent to an un-normalized inverse transform in the equidistant case, although note a reversal in convention from [^2]. Used to interpolate over the nominal continuous range of `x` similarly normalised to the discontinuous case. From: [dependency](https://github.com/jakevdp/nfft)[^1].

# Credit
Michael Sorochan Armstrong (mdarmstr@go.ugr.es) and José Camacho Páez (josecamacho@ugr.es) from the Computational Data Science Lab (CoDaS) at the University of Granada. Please, note that the software is provided "as is" and we do not accept any responsibility or liability. Should you find any bug or have suggestions, please contact the authors. For copyright information, please see the license file.

# Installation instructions
In progress - please see `requirements.txt` for a list of dependencies

[^1]: NFFT package written by @jakevdp [https://github.com/jakevdp/nfft](https://github.com/jakevdp/nfft)
[^2]: Kunis, Stefan, and Daniel Potts. "Stability results for scattered data interpolation by trigonometric polynomials." SIAM Journal on Scientific Computing 29.4 (2007): 1403-1419.



