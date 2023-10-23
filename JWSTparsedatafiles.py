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


def process_target_list(datafile_paths):
    """ 
    Parse list of JWST datafiles, compile table of 
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
    print("parsing...")
    data = []
    for path in datafile_paths:
        with io.fits.open(path) as hdul:
            try:  # filter and grating exist only for NIRSpec
                filter_name = hdul[0].header["FILTER"].strip()
                grating_name = hdul[0].header["GRATING"].strip()
            except:
                filter_name = "None"
                grating_name = "None"
            targets.add_row(
                (hdul[0].header["TARGNAME"], hdul[0].header["TARG_RA"], 
                 hdul[0].header["TARG_DEC"], hdul[0].header["INSTRUME"], 
                 hdul[0].header["DETECTOR"], filter_name, 
                 grating_name, hdul[0].header["EFFINTTM"],
                 hdul[1].data["WAVELENGTH"][0],
                 hdul[1].data["WAVELENGTH"][-1], path))
            data_i = {}
            data_i["lam"] = hdul[1].data["WAVELENGTH"]
            data_i["sky"] = hdul[1].data["BACKGROUND"]
            data_i["error"] = hdul[1].data["BKGD_ERROR"]  
            data_i["grating"] = grating_name
            data_i["name"] = hdul[0].header["TARGNAME"]
                #check other error entries
            data.append(data_i)
    # add resolutions 
    for row in data:
        try:
            row["max_res"] = max_res[row["grating"]]
        except:
            row["max_res"] = 0.005 # guess for miri
    print("done\n")
    # compute D factors 
    coords = coord.SkyCoord(ra=targets["ra"]*u.degree,
                            dec=targets["dec"]*u.degree,
                            distance=1.0*u.kpc)  # placeholder distance
    targets["b"] = coords.galactic.b.degree
    targets["l"] = coords.galactic.l.degree
    Ds = np.zeros(Ntargets)
    for index in range(Ntargets):
        Ds[index] = mw.compute_halo_Dfactor(
            targets[index]["b"], targets[index]["l"],
            mw.NFWprofile, assume.r_sun/assume.r_s)*assume.rho_s*assume.r_s
        data[index]["D"] = Ds[index]
    targets["D"] = Ds
    # compute min distance from galactic center
    b_rad = targets["b"]*(np.pi/180.0)
    l_rad = targets["l"]*(np.pi/180.0) 
    targets["b_impact"] = assume.r_sun*np.sqrt(1-np.cos(b_rad)**2*np.cos(l_rad)**2)
    # compute relative velocity and apply doppler shift 
    coords_galcen = coords.transform_to(assume.galcen)
    coords_vec = np.asarray([coords_galcen.x.value,
                             coords_galcen.y.value,
                             coords_galcen.z.value])
    diff_vec = coords_vec.T - assume.vec_sun
    len_diff_vec = np.sqrt(np.sum(diff_vec**2, axis=1))
    diff_hat = (diff_vec.T/len_diff_vec).T
    v_parallel = np.sum(diff_hat*assume.v_sun, axis=1)
    targets["v_rel"] = v_parallel # km/s, velocity of the sun along the target
                                  # line-of-sight in galactocentric frame
    for index in range(Ntargets):
        data[index]["lam"] *= (1.0 - targets[index]["v_rel"]/assume.c_kms)
        data[index]["max_res"] *= (1.0 - targets[index]["v_rel"]/assume.c_kms)
        # apply doppler shift
        # postive v_rel indicates sun is moving towards the line-of-sight
        # (no need to rescale the spectra, as both the observed and 
        #  model spectra are rescaled identically)
    return data, targets
    

def get_fits_from_tree(datadir):
    """ get all *.fits files from directory tree starting at datadir """
    datafile_paths = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        datafile_paths += [os.path.join(current_path, f) 
                           for f in current_filenames
                           if f[-5:]==".fits"]
    return datafile_paths