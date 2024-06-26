""" Units conversion for astro DM decay """


import numpy as np


c_kms = 299792.458 # km/sec


def wavelength_to_mass(lam):
    """ 
    convert two-photon decay wavelength in microns to 
    dm mass in ev, via lam = 4pi/m
    """
    return 2.47968/lam # eV

def mass_to_wavelength(m):
    """ 
    convert dark matter mass in eV to the two-photon 
    decay wavelength in microns via lam = 4pi/m
    """
    return 2.47968/m # micron

def fluxscale_to_invsec(limit):
    """ 
    The code computes the decay rate G via 
        Phi = dphi/(dnu dOmega)
            = (G/4pi) (df/dnu * W) D
    The quantites are input in the following units:
        D = Din GeV/cm^3 kpc
        (df/dnu * W) = S_in micron
    Note that though df/dnu is 'per frequency', we express it and 
    then evaluate it in the code as a function of wavelength, so
    its units are in microns, the wavelenth unit used in the code. 
        Phi = Phi_in MJy 
    This is "really" MJy/sr, but the sr is in a sense conventional
    so we drop it in this comparison.  Another way to see it: suppose
    the instrument gave us a measure of Phi in MJy/sr over a solid 
    angle Omega.  Then we would compute  
        Phi Omega = (G/4pi) (df/dnu * W) D Omega
    and both sides are now in honest-to-God MJy, no sr involved. 
    But this is exactly equilavent to the above.  So G is computed 
    by the code to be 
        G_out = (4 pi Phi_in) / (S_in D_in)
    and the units of G_out are 
        G = G_out (MJy cm^3 / micron GeV kpc) 
    We want to convert this into 1/sec, 
        G = Ginvsec (1/sec)
        Ginvsec = G_out (MJy cm^3 sec/ micron GeV kpc) 
    The quantity in parentesis is unitless, and is the scale 
    factor used here. See unit-converstion.nb for evaluation. 
    """
    return limit*(6.06401e-22) # 1/sec

def invsec_to_fluxscale(rate):
    """ 
    convert decay rate in 1/sec to the flux scale 
    used in the code, see fluxscale_to_invsec for details
    """
    return rate/(6.06401e-22) # MJy cm^3 / micron GeV kpc

def decayrate_to_axion_g(rate, m):
    """ 
    convert decay rate in sec^-1 and axion mass in eV to the
    axion-two photon coupling in GeV^-1 
    """ 
    return 363.788*np.sqrt(rate)/(m**1.5) # 1/GeV

def axion_g_to_decayrate(g, m):
    """ 
    convert axion-two photon coupling in GeV^-1 and axion mass 
    in eV to the decay rate in sec^-1    
    """ 
    return (m**3)*(g/363.788)**2 # 1/sec
