#!/usr/bin/env python3
""" Download data from JWST archives """

import numpy as np
import matplotlib.pyplot as plt
import astropy.io.fits as fits 


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
JWSTdata_dir = "{}/data/mastDownload/JWST".format(project_dir)

datafile_path = ("{}/jw04426-o001_t001_nirspec_g140m-f100lp/"
                 "jw04426-o001_t001_nirspec_g140m-f100lp_x1d.fits".format(JWSTdata_dir))

if __name__ == "__main__":
    hdul=fits.open(datafile_path)
    wavelength = hdul[1].data["WAVELENGTH"]
    sky = hdul[1].data["BACKGROUND"]
    sky_error = hdul[1].data["BKGD_ERROR"]  # check the other error entries

    fig, ax = plt.subplots()
    # ax.plot(wavelength, sky, color='k', alpha=0.6)

    ax.plot(np.log(wavelength), np.log(sky), color='k', alpha=0.6)
    plt.show()
