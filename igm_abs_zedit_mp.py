import os, subprocess, sys
import shutil
import glob
from copy import copy
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
from scipy.special import gammaln

from astropy.io import fits, ascii
from astropy.table import Table
from astropy import units as u
from astropy.cosmology import FlatLambdaCDM
from astropy.constants import c, m_e, e
import matplotlib.pyplot as plt

import line_profiler
import multiprocessing as mp
from numpy.random import random, poisson, seed

from vars import *

def poisson_distr(xm):
    oldm = -1.
    if xm < 12.0:
        if xm != oldm:
            oldm = xm
            g = np.exp(-1*xm)
        em = -1
        t = 1.0
        while (t > g):
            em+=1
            t*=np.random.random()
    else:
        if xm != oldm:
            oldm = xm
            sq = np.sqrt(2.*xm)
            alxm = np.log(xm)
            g = xm*alxm - gammaln(xm+1.0)
        #y = np.tan(np.pi * np.random.random())
        #em = sq*y + xm
        em = -1
        t = 0.0
        while(np.random.random() > t):
            while(em < 0.0):
                y = np.tan(np.pi * np.random.random())
                em = sq*y + xm
            em = np.floor(em)
            t=0.9*(1.0 + y*y)*np.exp(em*alxm - gammaln(em+1.0) - g)

        return em
                                  
            
    
# Calculate number of absorbers in two density regimes, integrating over NHI and redshift
def number_abs(zref, zmin, A_hi, A_lo, beta_hi, beta_lo, gamma_hi, gamma_lo, Nhimin, Nhimax, Nlomin, Nlomax):
    
    expNhi=A_hi/(1+beta_hi)/(1+gamma_hi)*(
        (np.power(Nhimax, 1+beta_hi, dtype=np.float64) - np.power(Nhimin, 1+beta_hi, dtype=np.float64))*
        (np.power((1+zref), 1+gamma_hi, dtype=np.float64) - np.power((1+zmin), 1+gamma_hi, dtype=np.float64)))
    expNlo=A_lo/(1+beta_lo)/(1+gamma_lo)*(
        (np.power(Nlomax, 1+beta_lo, dtype=np.float64) - np.power(Nlomin, 1+beta_lo, dtype=np.float64))*
        (np.power((1+zref), 1+gamma_lo, dtype=np.float64) - np.power((1+zmin), 1+gamma_lo, dtype=np.float64)))

    return (expNhi, expNlo)

# Calculate number of absorbers in the CGM, integrating over NHI and redshift
def number_abs_cgm(zref, A, beta, gamma, Nmin, Nmax, dz_cgm=0.01):
    expNcgm = A/(1+beta)/(1+gamma)*(
        (np.power(Nmax, 1+beta, dtype=np.float64) - np.power(Nmin, 1+beta, dtype=np.float64))*
        (np.power((1+zref), 1+gamma, dtype=np.float64) - np.power((1+zref-dz_cgm), 1+gamma, dtype=np.float64)))
    return expNcgm

def sigma_ll_b_arr(lam, lam_lyman_line, A, b): # array version

    sigmas = np.empty(len(lam_lyman_line))
    
    k = 1.38066e-16

    lam = lam# * u.AA
    lam_ll = lam_lyman_line# * u.AA
    nu = 299792458.0 / (lam / 1e10)
    nu_ll = 299792458.0 / (lam_ll / 1e10)
    delta_nu = (1e13 / lam_ll) * b # b in km/s divided by lam_ll in km, resulting in Hz

    p_HI = np.exp(-1 * ((nu - nu_ll)/delta_nu)**2) / (np.sqrt(np.pi) * delta_nu)
    sigma_dop = 3 * (lam_ll / 1e8)**2 / (8*np.pi) * A * p_HI
    
    omega = 2 * np.pi * nu
    omega_ll = 2 * np.pi * nu_ll

    sigma_damp = 3 * (lam_ll/1e8)**2 / (8 * np.pi) * ((omega/omega_ll)**4 / ((omega-omega_ll)**2 + A**2 * (omega/omega_ll)**6 / 4))

    ix_1 = np.where((nu - nu_ll) <= delta_nu)[0]
    ix_2 = np.where((~((nu - nu_ll) <= delta_nu)) & (sigma_dop < sigma_damp))[0]
    ix_3 = np.where((~((nu - nu_ll) <= delta_nu)) & (~(sigma_dop < sigma_damp)))[0]
    if len(ix_1) > 0:
        sigmas[ix_1] = sigma_dop[ix_1]
    if len(ix_2) > 0:
        sigmas[ix_2] = sigma_damp[ix_2]
    if len(ix_3) > 0:
        sigmas[ix_3] = sigma_dop[ix_3]

    return sigmas.sum()
    

def plot_spec(psi, z, nb, show=True):
    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(psi.WAVE, psi.CONT, color='k', drawstyle='steps-mid',linestyle='-', marker='', label='Source')
    ax.plot(psi.WAVE, psi.VSD, color='r', drawstyle='steps-mid',linestyle=':', marker='', label='Attenuated (total)')
    ax.plot(psi.WAVE, psi.VSD_LyC, color='b', drawstyle='steps-mid',linestyle=':', marker='', label='Attenuated (LyC)')
    ax.plot(psi.WAVE, psi.VSD_LL, color='green', drawstyle='steps-mid',linestyle=':', marker='', label='Attenuated (lines)')
    ax.plot(psi.WAVE, psi.VSD_hi, color='purple', drawstyle='steps-mid',linestyle=':', marker='', label='Attenuated (Nhi)')
    ax.plot(psi.WAVE, psi.VSD_lo, color='cyan', drawstyle='steps-mid',linestyle=':', marker='', label='Attenuated (Nlo)')

    ax.axvline(912., linestyle='--', alpha=0.7, label='Lyman Limit')

    ax.legend(loc='lower right')

    ax.set_xlim([880, 1215])

    plt.savefig(f'{pdata_path}/mc/{zref:.2f}/spec_{nb:03d}.png')
    if show:
        plt.show()
    plt.close()

def init_pool_processes():
    seed()

def absorb_one_wrapper(args):
    result = absorb_one(*args)
    return result

def absorb_one(waves, zref, zmin, lyman_coeff, regime='hi'):

    if regime == 'hi':
        A = 4.473e7
        beta=-1.479
        gamma=1.0  
        Nmin=1.38039e+15
        Nmax=1.0e+22
    elif regime == 'lo':
        A= 4.250e9   
        beta=-1.656
        gamma=2.50
        Nmin=1.0e+12
        Nmax=1.38039e+15
    elif regime == 'cgm':
        A = 3.49e6
        beta = -1.365
        gamma = 1.0
        Nmin = 1e12
        Nmax = 1e22
        delcgm = 0.01
    else:
        raise Exception()
    bmode = 26.
    num_lls_tau=0

    random_number = random()
    # column density of this absorber
    nh = (np.power(Nmax, 1+beta, dtype=np.float64) - np.power(Nmin, 1+beta)) * random_number + (
        np.power(Nmin, 1+beta))
    nh = np.power(nh, 1./(1+beta))

    # redshift of this absorber
    random_number = random()
    zh = (np.power(1+zref, 1+gamma) - np.power(1+zmin, 1+gamma)) * random_number + np.power(1+zmin, 1+gamma)
    zh = np.power(zh, 1./(1+gamma)) - 1.0
    print(f'Adding LLS, z={zh:.2f} nh1={nh:.2e}')

    # If redshift is far enough away and at high column density, count as LLS
    if (zh > zref-0.034) and (nh >= 1.58e17) and (regime=='hi'):
        num_lls_tau+=1

    # Calculate "b"
    while True:
        bxtmp = random() * 100.
        bmax = np.power(bmode, -5) * np.exp(-1)
        bytmp = random() * bmax
        if (bytmp <= (np.power(bxtmp, -5) * np.exp(-1 * np.power(bmode, 4) / np.power(bxtmp, 4)))):
            b = bxtmp
            break
    # At each wavelength
    ts = np.zeros(len(waves))
    tforests = np.zeros(len(waves))
    teffs = np.zeros(len(waves))
    waves_lyman = lyman_coeff.WAVE.values
    As_lyman = lyman_coeff.A.values
    for i,lam in enumerate(waves):
        # Compute photoelectric cross-section (LyC)
        lam_shift = lam * (1+zref) / (1+zh)
        sigma = 6.3e-18 * (lam_shift / 912)**3
        # If the wavelength element is redward of the LL of the absorber, set cross section to zero
        if (lam_shift > 912.):
            sigma=0.
        teff = nh*sigma
        # Lyman series absorption
        tforest = 0
        sigmaforest = 0
        rel_lyman_ixs = np.where(np.abs(lam_shift - waves_lyman) <=10)[0]
        if len(rel_lyman_ixs) > 0:
            sigmaforest = sigma_ll_b_arr(lam_shift, waves_lyman[rel_lyman_ixs], As_lyman[rel_lyman_ixs], b)
            
        tforest = nh*sigmaforest
        ttotal = teff+tforest
        ts[i] = ttotal
        tforests[i] = tforest
        teffs[i] = teff
    return teffs, tforests

def absorb(pspec, zref, zmin, lyman_coeff, p, cgm=True):

    # constants for distribution functions
    A_hi = 4.473e7
    A_lo= 4.250e9   
    beta_hi=-1.479
    beta_lo=-1.656
    gamma_hi=1.0  
    gamma_lo=2.50
    Nhimin=1.38039e+15
    Nhimax=1.0e+22
    Nlomin=1.0e+12
    Nlomax=Nhimin
    bmode = 26.
    num_lls_tau=0

    # Generate distribution of absorbers
    expNhi, expNlo = number_abs(zref, zmin, A_hi, A_lo, beta_hi, beta_lo, gamma_hi, gamma_lo, Nhimin, Nhimax, Nlomin, Nlomax)
    print(expNhi, expNlo)

    # Random integer from a Poisson distribution with mean expNhi, expNlo
    Nhi = poisson(expNhi)#poisson_distr(expNhi)
    Nlo = poisson(expNlo)#poisson_distr(expNlo)
    print(Nhi, Nlo)

    # Lyman limit absorption
    # Draw absorbers from distribution of column density and redshift
    # Absorb according to absorber's nhi and zhi
    # Start with Nhi, generate until you have up to Nhi aborbers
    args = []
    for i in range(Nhi):
        args.append((pspec.WAVE.values, zref, zmin, lyman_coeff, 'hi'))

    results = np.array(p.map(absorb_one_wrapper, args))
    ts = results[:,0] + results[:,1]
    teffs = results[:,0]
    tforests = results[:,1]

    pspec.VSD = pspec.VSD * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_igm = pspec.VSD_igm * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_hi = pspec.VSD_hi * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_LyC = pspec.VSD_LyC * (np.exp(-1 * teffs)).prod(axis=0)
    pspec.VSD_LyC_igm = pspec.VSD_LyC_igm * (np.exp(-1 * teffs)).prod(axis=0)
    pspec.VSD_LL = pspec.VSD_LL * (np.exp(-1 * tforests)).prod(axis=0)
    pspec.VSD_LL_igm = pspec.VSD_LL_igm * (np.exp(-1 * tforests)).prod(axis=0)
    
    # Repeat for Nlo
    args = []
    for i in range(Nlo):
        args.append((pspec.WAVE.values, zref, zmin, lyman_coeff, 'lo'))

    results = np.array(p.map(absorb_one_wrapper, args))
    ts = results[:,0] + results[:,1]
    teffs = results[:,0]
    tforests = results[:,1]

    pspec.VSD = pspec.VSD * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_igm = pspec.VSD_igm * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_lo = pspec.VSD_lo * (np.exp(-1 * ts)).prod(axis=0)
    pspec.VSD_LyC = pspec.VSD_LyC * (np.exp(-1 * teffs)).prod(axis=0)
    pspec.VSD_LyC_igm = pspec.VSD_LyC_igm * (np.exp(-1 * teffs)).prod(axis=0)
    pspec.VSD_LL = pspec.VSD_LL * (np.exp(-1 * tforests)).prod(axis=0)
    pspec.VSD_LL_igm = pspec.VSD_LL_igm * (np.exp(-1 * tforests)).prod(axis=0)

    # Repeat for CGM
    if cgm:
        A = 3.49e6
        beta = -1.365
        gamma = 1.0
        Nmin = 1e12
        Nmax = 1e22
        dz_cgm = 0.01
        # Generate distribution of absorbers
        expNcgm = number_abs_cgm(zref, A, beta, gamma, Nmin, Nmax, dz_cgm=dz_cgm)
        print(expNcgm)
        # Random integer from a Poisson distribution with mean expNcgm
        Ncgm = poisson(expNcgm)
        print(Ncgm)

        # LyC / Lyman series absorption
        # Create N=Ncgm absorbers from distribution of column density and redshift
        # Absorb according to absorber's nhi and zhi
        args = []
        for i in range(Ncgm):
            args.append((pspec.WAVE.values, zref, zref-dz_cgm, lyman_coeff, 'cgm'))

        results = np.array(p.map(absorb_one_wrapper, args))
        ts = results[:,0] + results[:,1]
        teffs = results[:,0]
        tforests = results[:,1]

        pspec.VSD = pspec.VSD * (np.exp(-1 * ts)).prod(axis=0)
        pspec.VSD_cgm = pspec.VSD_cgm * (np.exp(-1 * ts)).prod(axis=0)
        pspec.VSD_LyC = pspec.VSD_LyC * (np.exp(-1 * teffs)).prod(axis=0)
        pspec.VSD_LL = pspec.VSD_LL * (np.exp(-1 * tforests)).prod(axis=0)

    return pspec


if __name__ == '__main__':

    # Define the input spectrum
    lambda_min = 500
    lambda_max = 1250
    delta_lambda = 0.1
    lambdas = np.arange(500, 1250, 0.1)
    pspec = pd.DataFrame({'WAVE': lambdas,
                          'VSD': 1.0,
                          'VSD_igm': 1.0,
                          'VSD_LyC': 1.0,
                          'VSD_LyC_igm': 1.0,
                          'VSD_LL': 1.0,
                          'VSD_LL_igm': 1.0,
                          'VSD_LL': 1.0,
                          'VSD_hi': 1.0,
                          'VSD_lo': 1.0,
                          'VSD_cgm': 1.0,
                          'CONT': 1.0,
                          })

    # skip applying absorption from the galaxy's own dust

    # redshift the spectrum
    # this is assuming f_nu i believe
    pspec.loc[:, 'WAVE_OBS'] = lambdas * (1+zref)
    pspec.loc[:, 'VSD_OBS'] = pspec.VSD

    cpus = mp.cpu_count()
    p = mp.Pool(processes = cpus, initializer=init_pool_processes)

    # apply MC realization of absorption to each sightline
    for i in range(n_mc):
        psi = pspec.copy()
        psi = absorb(psi, zref, zmin, lyman_coeff, p, cgm=True)

        # Save absorbed spectrum
        odir = f'{pdata_path}/mc/{zref:.2f}'
        if not os.path.isdir(odir):
            os.mkdir(odir)
        plot_spec(psi, zref, i, show=False)
        psi.to_pickle(f'{pdata_path}/mc/{zref:.2f}/spec_{i:03d}.pkl')
