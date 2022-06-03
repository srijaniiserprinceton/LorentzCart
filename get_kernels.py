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

kRmin  = 600; kRmax  = 1500

#.....defining kx,ky grid for a cube of 401 pixels in X and Y; distance between adjacent pixels - 0.972 Mm
kx = ky = np.fft.fftshift(np.fft.fftfreq(401,0.972*1e6))*2*np.pi

kxg,kyg = np.meshgrid(kx,ky,indexing='ij')
abskg = np.sqrt(kxg**2+kyg**2)

#.....computing kernels for f-f coupling; i.e., n = n' = 0
n,npr = 0,0

#.....i only want to compute kernels for 600 < |k|R < 1500 for f-f coupling as defined in the first line of this cell
k_mask = np.where((abskg*RSUN < kRmax)*(abskg*RSUN > kRmin),1,0)
nmodes = np.where(k_mask==True)[0].size

#.....only want to compute kernels for |q|R <= 300
absq_range = np.array([0,300])/RSUN

QXX = np.arange(16) ; QYY = np.arange(-15,16)
QXY = [[i,j] for idx1,i in enumerate(QXX) for idx2,j in enumerate(QYY)]

flow_kern = compute_kernels.compute_kernels(QXY[0])
