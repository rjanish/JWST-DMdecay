""" parse and condense JWST spectral resolution calibration files """


import os 
import sys 
import tomli 

import numpy as np
import astropy.io.fits as fits 
import astropy.table as table


if __name__ == "__main__":
    config_path = sys.argv[1]    
    print(F"parsing {config_path}")
    with open(config_path, "rb") as f: 
        config = tomli.load(f)
    res_dir = config["paths"]["resolution_dir"]
    base = config["paths"]["resolvedres_base"]
    # process sepectral resolution calibrations 
    calibration_paths = [F"{res_dir}/{f}" 
                         for f in os.listdir(res_dir)
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
            save_path = F"{res_dir}/{base}-{grating}.dat"
            header = ("spectral resolution vs wavelength "
                      "for JWST NIRSPEC grating {}\n"
                      "wavelength [microns]    resolution [microns]")
            np.savetxt(save_path, 
                       np.column_stack((lam, res)),
                       header=header)
    res_max.write(F"{res_dir}/{config['paths']['maxres_path']}", 
                  overwrite=True,
                  format="ascii.commented_header")

