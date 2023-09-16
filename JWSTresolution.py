#!/usr/bin/env python3
""" parse and condense JWST spectral resolution calibration files """


import os 

import numpy as np
import astropy.io.fits as fits 
import astropy.table as table

import DMdecayJWST as assume


if __name__ == "__main__":
    # process sepectral resolution calibrations 
    calibration_paths = ["{}/{}".format(assume.resolution_dir, f) 
                         for f in os.listdir(assume.resolution_dir)
                         if f[-9:]=="disp.fits"]  
    res_max = table.Table(dtype=[("grating", str), 
                                 ("max_res", float),
                                 ("min_fracres", float)])
    for path in calibration_paths:
        with fits.open(path) as hdul:
            grating = hdul[0].header["COMPNAME"].strip()
            lam = hdul[1].data["WAVELENGTH"]
            R = hdul[1].data["R"]
            res = lam/R
            res_max.add_row([grating, np.max(res), np.min(1.0/R)])
            save_path = ("{}/JWST-NIRSPEC-{}-resolution-resolved.dat"
                         "".format(assume.resolution_dir, grating))
            header = ("spectral resolution vs wavelength "
                      "for JWST NIRSPEC grating {}\n"
                      "wavelength [microns]    resolution [microns]")
            np.savetxt(save_path, 
                       np.column_stack((lam, res)),
                       header=header)
    res_max.write(assume.maxres_path, overwrite=True,
                  format="ascii.commented_header")

