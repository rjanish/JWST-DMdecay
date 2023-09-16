""" 
Parse JWST datfiles
"""


import os 

import numpy as np
import astropy.io as io
import astropy.table as table
import astropy.coordinates as coord
import astropy.units as u

import DMdecayJWST as assume 
import MWDMhalo as mw


def process_target_list(datadir):
    """ 
    Parse a directory tree of JWST datafile, compile table of 
    targets and important metadata data, compute D factors, 
    and extract all info needed for the DM analysis.  
    """
    # read in resolutions
    max_res_table = io.ascii.read(assume.maxres_path)
    max_res = {line["grating"]:line["max_res"] for line in max_res_table}
        # For now simply take the max resolution for each 
        # grating as though uniform, and assume this is larger 
        # than the DM intrisic width.  This is very close to true. 
        # We can extend later to use the wavelength-dependent 
        # resolution and convolve a finite DM line profile, but the
        # results will not change appreciably 
    # find all data files 
    datafile_paths = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        datafile_paths += [os.path.join(current_path, f) 
                           for f in current_filenames
                           if f[-5:]==".fits"]
    # start table with metadata from data files  
    Ntargets = len(datafile_paths)
    targets = table.Table(
        names=("name", "ra", "dec", "instrument", 
               "detector", "filter", "grating", "int_time",
               "lambda_min", "lambda_max", "path"), 
        dtype=(str, float, float, str, 
               str, str, str, float,
               float, float, str))
    print("found {} datafiles".format(len(datafile_paths)))
    print("parsing...\n")
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
            mw.NFWprofile, assume.r_sun/assume.r_s)
        data[index]["D"] = Ds[index]
    targets["D"] = Ds
    # add resolutions 
    for row in data:
        row["max_res"] = max_res[row["grating"]]
    return data, targets 

