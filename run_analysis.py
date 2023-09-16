#!/usr/bin/env python3
""" Run JWST DM search """

import sys
import os 

import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import astropy.io as io
import astropy.table as table
import astropy.coordinates as coord
import astropy.units as u

import MWDMhalo as mw


rho_s = 0.18 # GeV/cm^3
r_s = 24.0 # kpc
r_sun = 8.1 # kpc

def wavelength_to_mass(lam):
    """ convert wavelength in microns to mass in ev """
    return 2.46/lam

def fluxscale_to_invsec(limit, rho_s, r_s):
    """ 
    convert flux scale limit in MJy/(sr micron kpc GeV/cm^3)
    do a decay rate in 1/sec. This assumes the D-factors 
    were normalized by the given rho_s and r_s values 
    """
    return limit*(4.71e-23)/(rho_s*r_s)

def decayrate_to_axion_g(rate, m):
    """ 
    convert decay rate in sec^-1 and mass in eV to the
    axion-two photon coupling in GeV^-1 
    """ 
    return 648.9*np.sqrt(rate)/(m**1.5)


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
JWSTdata_dir = os.path.join(project_dir, "data/mastDownload/JWST")



def process_target_list(datadir):
    """ 
    compile table of targets and important targe data 
    """
    # find all data files 
    datafile_paths = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        datafile_paths += [os.path.join(current_path, f) 
                           for f in current_filenames]
    # start table with metadata from data files  
    Ntargets = len(datafile_paths)
    targets = table.Table(
        names=("name", "ra", "dec", "instrument", 
               "detector", "filter", "grating", "int_time",
               "lambda_min", "lambda_max", "path"), 
        dtype=(str, float, float, str, 
               str, str, str, float,
               float, float, str))
    data = []
    for path in datafile_paths:
        with io.fits.open(path) as hdul:
            targets.add_row(
                (hdul[0].header["TARGNAME"], hdul[0].header["TARG_RA"], 
                 hdul[0].header["TARG_DEC"], hdul[0].header["INSTRUME"], 
                 hdul[0].header["DETECTOR"], hdul[0].header["FILTER"], 
                 hdul[0].header["GRATING"], hdul[0].header["EFFINTTM"],
                 hdul[1].data["WAVELENGTH"][0],
                 hdul[1].data["WAVELENGTH"][-1], path))
            data_i = {}
            data_i["lam"] = hdul[1].data["WAVELENGTH"]
            data_i["sky"] = hdul[1].data["BACKGROUND"]
            data_i["error"] = hdul[1].data["BKGD_ERROR"]  
            data_i["grating"] = hdul[0].header["GRATING"].strip()
                #check other error entries
            data.append(data_i)
    # compute D factors 
    coords = coord.SkyCoord(ra=targets["ra"]*u.degree,
                            dec=targets["dec"]*u.degree)
    targets["b"] = coords.galactic.b.degree
    targets["l"] = coords.galactic.l.degree
    Ds = np.zeros(Ntargets)
    for index in range(Ntargets):
        Ds[index] = mw.compute_halo_Dfactor(
            targets[index]["b"], targets[index]["l"],
            mw.NFWprofile, r_sun/r_s)
        data[index]["D"] = Ds[index]
    targets["D"] = Ds
    # save metadata table
    targets.write("targets.html", format="ascii.html", overwrite=True)
    targets.write("targets.dat", format="ascii.csv", overwrite=True)
    return data, targets 


if __name__ == "__main__":
    if os.getcwd() != project_dir:
        print("ERROR: only run from {}".format(project_dir))
        exit()
    run_name = sys.argv[1]    

    data, targets = process_target_list(JWSTdata_dir)
    # read in resolutions
    res_path = os.path.join(project_dir, 
                            "data/resolution",
                            "JWST-NIRSPEC-max-resolution.dat")
    max_res_table = io.ascii.read(res_path)
    max_res = {line["grating"]:line["max_res"] for line in max_res_table}
    for row in data:
        row["max_res"] = max_res[row["grating"]]

    # find conservative limit 
    chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux)
    chisq_threshold = 4
    frac_step = 0.5e-3

    lam_min = np.min(targets["lambda_min"])
    lam_max = np.max(targets["lambda_max"])
    dlam = lam_max*frac_step
    lam_test = np.arange(lam_min, lam_max+dlam, dlam)
    
    limit = np.zeros(lam_test.shape)
    for index, lam_test_i in enumerate(lam_test):
        sol = opt.root_scalar(chisq_nb, 
                              args=(lam_test_i, chisq_threshold), 
                              bracket=[0, 100])
        limit[index] = sol.root

    m = wavelength_to_mass(lam_test)
    limit_decayrate = fluxscale_to_invsec(limit, rho_s, r_s)    
    limit_g = decayrate_to_axion_g(limit_decayrate, m)    
    
    save_path = os.path.join(project_dir,
                             "JWST-NIRSPEC-limits-{}.dat".format(run_name))
    header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(run_name))
    np.savetxt(save_path, np.column_stack((m, limit_decayrate, limit_g)),
               header=header)

    fig, ax = plt.subplots()
    ax.plot(m, limit_g)
    ax.set_yscale('log')
    plt.show()



