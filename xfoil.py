# Basic xfoil module that extracts values of alpha/mach/re from xfoil.mat and returns a linearly interpolated estimate of the cp.
# Locations can be found in x, as soon as init has been called. Initialization is also done on first call to cp.

import numpy as np
import scipy.io as sio
from scipy.interpolate import RegularGridInterpolator
import functools

_xfoil_all = []
x = []

def _xfoil_prepare():
    global x, _xfoil_all
    
    cpx = sio.loadmat('xfoil.mat')
    x = cpx['x']
    alpha = np.linspace(1, 6, 21)
    mach = np.linspace(0.3, 0.6, 13)
    re = np.linspace(1e6, 1e7, 19)

    _xfoil_all = [None] * x.size

    for i in range(0, x.size):
        _xfoil_all[i] = RegularGridInterpolator((alpha, mach, re), cpx['cp'][:, :, :, i])

def init():
    _xfoil_prepare()

# From the documentation: The LRU feature performs best when maxsize is a power-of-two. This is 2^14
@functools.lru_cache(maxsize=16384)
def cp(alpha, mach, re):
    global _xfoil_all
    
    if len(_xfoil_all) == 0:
        _xfoil_prepare()
    
    cp = [None] * x.size
    
    for i in range(0, x.size):
        cp[i] = _xfoil_all[i](np.array([alpha, mach, re]))
    
    return np.array(cp)