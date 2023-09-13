#!/usr/bin/env python3
""" Download data from JWST archives """

import os  

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits 
import astropy.table as table
import astropy.coordinates as coord
import astropy.units as u

import MWDMhalo as mw


rho_s = 0.18 # GeV/cm^3
r_s = 24.0 # kpc
r_sun = 8.1 # kpc

project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
JWSTdata_dir = os.path.join(project_dir, "data/mastDownload/JWST")


def get_target_list(datadir):
    """ 
    compile table of targets and important targe data 
    """
    # find all data files 
    datafile_paths = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        datafile_paths += [os.path.join(current_path, f) for f in current_filenames]
    # start table with metadata from data files  
    Ntargets = len(datafile_paths)
    targets = table.Table(
        names=("name", "ra", "dec", "instrument", 
               "detector", "filter", "grating", "int_time"), 
        dtype=(str, float, float, str, 
               str, str, str, float))
    for filename in datafile_paths:
        with fits.open(filename) as hdul:
            targets.add_row(
                (hdul[0].header["TARGNAME"], hdul[0].header["TARG_RA"], 
                 hdul[0].header["TARG_DEC"], hdul[0].header["INSTRUME"], 
                 hdul[0].header["DETECTOR"], hdul[0].header["FILTER"], 
                 hdul[0].header["GRATING"], hdul[0].header["EFFINTTM"]))
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
    targets["D"] = Ds

    print(targets)

if __name__ == "__main__":
    # hdul=fits.open(datafile_path)
    # wavelength = hdul[1].data["WAVELENGTH"]
    # sky = hdul[1].data["BACKGROUND"]
    # sky_error = hdul[1].data["BKGD_ERROR"]  # check the other error entries

    # fig, ax = plt.subplots()
    # # ax.plot(wavelength, sky, color='k', alpha=0.6)

    # ax.plot(np.log(wavelength), np.log(sky), color='k', alpha=0.6)
    # plt.show()
    get_target_list(JWSTdata_dir)
