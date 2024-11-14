import os
import pandas as pd
import numpy as np
from astropy.constants import c, m_e, e
home = os.path.expanduser('~')

src_path = '{}/lyc_tigm/src'.format(home)
data_path = '{}/lyc_tigm/data'.format(home)
res_path = '{}/lyc_tigm/results'.format(home)
pdata_path = '{}/lyc_tigm/processed_data'.format(home)


# For now, input redshifts and perturbations
zref = 2.4
zmin = 1.7
n_mc = 500


# Determines Einstein A coefficients via a wavelength and an oscillator strength
def einsteina(lam, fosci):
    #pi=3.14159265
    #e=4.803204e-10
    #me=9.10938188e-28
    #c=2.99792458e10

    x = 8.0 * np.pi**2 * e.esu**2 * fosci / (3 * np.power(lam/1e8, 2) * m_e * c).cgs
    return x.value
    #return 8.0*pi*pi*e*e/(3*np.power(lam/1.0e+8, 2)*me*c)*fosci

# Oscillator strength for a transition for a state n, where n=2 for Lya
def fosc(n):
    x = 2**9
    y = n**5 * np.power(n-1,2*n-4, dtype=np.float64)
    z = 3 * np.power(n+1,2*n+4, dtype=np.float64)
    return x * y / (2 * z)


# Calculate Einstein A coefficients for the Lyman series lines
lyman_coeff = pd.DataFrame(index=range(31), columns=['WAVE', 'A'])
lyman_coeff['WAVE'] = [
    1215.67,
    1025.72,
    972.537,
    949.743,
    937.8034,
    930.748,
    926.226,
    911.267/0.987127,
    911.267/0.989472,
    911.267/0.99120656,
    911.267/0.9925259,
    917.180,
    916.429,
    915.824,
    915.329,
    914.919,
    914.576,
    914.286,
    914.039,
    913.826,
    913.641,
    913.480,
    913.339,
    913.215,
    913.104,
    913.006,
    912.918,
    912.839,
    912.768,
    912.703,
    912.645,
]
lyman_coeff.loc[:4, 'A'] = [
    6.265e+8,
    1.672e+8,
    6.818e+7,
    3.437e+7,
    1.973e+7
]
for i in np.arange(5, 31):
    lyman_coeff.loc[i, 'A'] = einsteina(lyman_coeff.loc[i, 'WAVE'], fosc(i+2))
