#!/usr/bin/env python3
""" Download data from JWST archives """

import os  

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits 

import MWDMhalo as mw


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
JWSTdata_dir = os.path.join(project_dir, "/data/mastDownload/JWST")

def get_target_list(datadir):
    data_filenames = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        data_filenames += current_filenames

    for filename in data_filenames:
        hdul=fits.open(datafile_path)
        ra = hdul[0].header["RA"]
        dec = hdul[0].header["DEC"]


if __name__ == "__main__":
    hdul=fits.open(datafile_path)
    wavelength = hdul[1].data["WAVELENGTH"]
    sky = hdul[1].data["BACKGROUND"]
    sky_error = hdul[1].data["BKGD_ERROR"]  # check the other error entries

    fig, ax = plt.subplots()
    # ax.plot(wavelength, sky, color='k', alpha=0.6)

    ax.plot(np.log(wavelength), np.log(sky), color='k', alpha=0.6)
    plt.show()
