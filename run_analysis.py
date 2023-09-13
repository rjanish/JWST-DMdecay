#!/usr/bin/env python3
""" Download data from JWST archives """

import os  

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits 
import astropy.table as table

import MWDMhalo as mw


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
JWSTdata_dir = os.path.join(project_dir, "/data/mastDownload/JWST")

def get_target_list(datadir):
    """ 
    compile table of targets and important targe data 
    """
    # find all data files 
    data_filenames = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        data_filenames += current_filenames
    # start table with metadata from data files  
    targets = table.Table(
        names=("name", "ra", "dec", "instrument", 
               "detector", "filter", "grating", "int_time"), 
        dtype=(str, float, float, str, 
               str, str, str, float))
    for filename in data_filenames:
        with fits.open(datafile_path) as hdul:
            targets.add_row(
                (hdul[0].header["TARGNAME"], hdul[0].header["TARG_RA"], 
                 hdul[0].header["TARG_DEC"], hdul[0].header["INSTRUME"], 
                 hdul[0].header["DETECTOR"], hdul[0].header["FILTER"], 
                 hdul[0].header["GRATING"], hdul[0].header["EFFINTTM"]))


if __name__ == "__main__":
    hdul=fits.open(datafile_path)
    wavelength = hdul[1].data["WAVELENGTH"]
    sky = hdul[1].data["BACKGROUND"]
    sky_error = hdul[1].data["BKGD_ERROR"]  # check the other error entries

    fig, ax = plt.subplots()
    # ax.plot(wavelength, sky, color='k', alpha=0.6)

    ax.plot(np.log(wavelength), np.log(sky), color='k', alpha=0.6)
    plt.show()
