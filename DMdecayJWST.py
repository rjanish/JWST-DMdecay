""" top-level assumptions and parameters for JWST DM search """

import numpy as np

import astropy.coordinates as coord 
import astropy.units as u

# parameters to normalize D factor and galactic DM distrbution
# any density distribution is given as rho(r) = rho_s f(r/r_s)
# for a purely dimensionless distribution f(x)
rho_s = 0.18 # GeV/cm^3
r_s = 24.0 # kpc

# velocity distribution 
sigma_v_kms = 160.0 # km/s, the asympotitc circular velocity 
                    #       is v0 = sqrt(2) sigma = 220 km/s
vesc_kms = 510.0    # km/s
c_kms = 3.0e5       # km/s
sigma_v = sigma_v_kms/c_kms
vesc = vesc_kms/c_kms

# define galactocentric corrdinate frame 
r_sun = 8.1    # kpc, distance from the sun to the galactic center
z_sun = 0.021  # kpc, distance of sun above galactic plane
v_sun = np.asarray(
    [2.9, 245.6, 7.78]) # km/s, velocity of sun wrt galactic center
ra_gc = 266.4051    # deg, icrs coords of galactic center
dec_gc = -28.936175 # deg, icrs coords of galactic center
galcen = coord.Galactocentric(
            galcen_coord=coord.SkyCoord(ra=ra_gc*u.deg, 
                                        dec=dec_gc*u.deg),
            galcen_distance=r_sun*u.kpc,
            galcen_v_sun=v_sun*u.km/u.s,
            z_sun = z_sun*u.pc)
vec_sun = np.asarray(
    [-r_sun, 0.0, z_sun]) # kpc, cartesain position of sun in galcen fame 

# paths 
download_dir = "data"
data_dir = "data/mastDownload/JWST"
resolution_dir = "data/resolution"
maxres_path = "data/resolution/JWST-NIRSPEC-max-resolution.dat"
AxionLimits_dir = "/home/rjanish/physics/AxionLimits/limit_data/AxionPhoton"




