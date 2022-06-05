import sys
import numpy as np
import matplotlib.pyplot as plt
# from astropy.io import fits
# from multiprocessing import Pool
# import jax.numpy as jnp
# from jax import jit, grad
# import gc
import patsy
import scipy
plt.style.use('default')
plt.rcParams.update({'font.size':16})

from src import compute_kernels

'''
defining the radial grid and B-splines
'''
quant = np.load('gyreEigenFunctions/eigs00.npz')
z = quant['z']
rho = quant['rho']
these_knots = np.append(z[::55],z[-1])
fj = patsy.bs(z, knots=these_knots, degree=3).T
RSUN = 696e6

#.....defining kx,ky grid for a cube of 401 pixels in X and Y; distance between adjacent pixels - 0.972 Mm
k_grid = np.fft.fftshift(np.fft.fftfreq(401,0.972*1e6))*2*np.pi

kRmin, kRmax = 600, 1500
kmin, kmax = kRmin/RSUN, kRmax/RSUN

#.....only want to compute kernels for |q|R <= 300
absq_range = np.array([0,300])/RSUN

#.....computing kernels for f-f coupling; i.e., n = n' = 0
n, n_ = 0,0

CartKerns = compute_kernels.CartKerns(z, n, n_, k_grid, kmin, kmax, absq_range, fj, rho)

QXX = np.arange(16) ; QYY = np.arange(-15,16)
QXY = [[i,j] for idx1,i in enumerate(QXX) for idx2,j in enumerate(QYY)]

flow_kern, Lorentz_kern = CartKerns.compute_kernels(QXY[0])
