""" Units conversion for astro DM decay """

import numpy as np

def wavelength_to_mass(lam):
    """ 
    convert two-photon decay wavelength in microns to 
    progeniter mass in ev, via lam = 4pi/m
    """
    return 2.46/lam

def fluxscale_to_invsec(limit, rho_s, r_s):
    """ 
    Convert flux scale limit in MJy/(sr micron kpc GeV/cm^3)
    do a decay rate in 1/sec. This assumes the D-factors 
    were normalized by the given rho_s and r_s values.
    The "flux scale" G is 
      dphi/dnu dOmega = 
          (G/4pi) (df/dnu * W) D
    where D here is dimensionless, its overall scale being factored
    out as rho_s r_s, df/dnu is the decay spectrum and W the 
    instrumental response function.  G is thus related to the 
    decay rate Gamma as G = Gamma rho_s r_s.  

    Double check all these units!!
    """
    return limit*(4.71e-23)/(rho_s*r_s)

def decayrate_to_axion_g(rate, m):
    """ 
    convert decay rate in sec^-1 and axion mass in eV to the
    axion-two photon coupling in GeV^-1 
    """ 
    return 648.9*np.sqrt(rate)/(m**1.5)