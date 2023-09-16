""" top-level assumptions and parameters for JWST DM search """


# parameters to normalize D factor and galactic DM distrbution
# any density distribution is given as rho(r) = rho_s f(r/r_s)
# for a purely dimensionless distribution f(x)
rho_s = 0.18 # GeV/cm^3
r_s = 24.0 # kpc

# distance from the sun to the galactic center
r_sun = 8.1 # kpc

# paths 
download_dir = "data"
data_dir = "data/mastDownload/JWST"
resolution_dir = "data/resolution"
maxres_path = "data/resolution/JWST-NIRSPEC-max-resolution.dat"
AxionLimits_dir = "/home/rjanish/physics/AxionLimits/limit_data/AxionPhoton"




