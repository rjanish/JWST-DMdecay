#!/usr/bin/env python3
""" Download data from JWST archives """


import os 

import numpy as np
import astropy.io.fits as fits 
import astropy.table as table


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
resolution_dir = os.path.join(project_dir, "data/resolution")

if __name__ == "__main__":
    if os.getcwd() != project_dir:
        print("ERROR: only run from {}".format(project_dir))
        exit()

    # process sepectral resolution calibrations 
    calibration_paths = [os.path.join(resolution_dir, f) 
                         for f in os.listdir(resolution_dir)
                         if f[-9:]=="disp.fits"]  
    res_max = table.Table(dtype=[("grating", str), ("max_res", float)])
    for path in calibration_paths:
        with fits.open(path) as hdul:
            grating = hdul[0].header["COMPNAME"].strip()
            lam = hdul[1].data["WAVELENGTH"]
            R = hdul[1].data["R"]
            res = lam/R
            res_max.add_row([grating, np.max(res)])
            save_path = os.path.join(
                resolution_dir,
                "JWST-NIRSPEC-{}-resolution-resolved.dat".format(grating))
            header = ("spectral resolution vs wavelength "
                      "for JWST NIRSPEC grating {}\n"
                      "wavelength [microns]    resolution [microns]")
            np.savetxt(save_path, 
                       np.column_stack((lam, res)),
                       header=header)
    res_max_path = os.path.join(resolution_dir, 
                                "JWST-NIRSPEC-max-resolution.dat")
    res_max.write(res_max_path, overwrite=True,
                  format="ascii.commented_header")

