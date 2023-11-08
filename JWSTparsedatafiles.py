""" 
Parse JWST datfiles
"""


import os 
import tomli

import numpy as np
import astropy.io as io
import astropy.table as table
import astropy.coordinates as coord
import astropy.units as u

import DMdecayJWST as parse 
import MWDMhalo as mw


def process_target_list(assume):
    """ 
    Parse list of JWST datafiles, compile table of 
    targets and important metadata data, compute D factors, 
    and extract all info needed for the DM analysis.  
    """
    datafile_paths = assume["run_data"]["paths"]
    # read in resolutions
    max_res_table = io.ascii.read(assume["paths"]["maxres_path"])
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
    print("parsing {} datafiles...".format(len(datafile_paths)))
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
            mw.NFWprofile, assume["mw_halo"]["r_sun"]/assume["mw_halo"]["r_s"])*assume["mw_halo"]["rho_s"]*assume["mw_halo"]["r_s"]
        data[index]["D"] = Ds[index]
    targets["D"] = Ds
    # compute min distance from galactic center
    b_rad = targets["b"]*(np.pi/180.0)
    l_rad = targets["l"]*(np.pi/180.0) 
    targets["b_impact"] = assume["mw_halo"]["r_sun"]*np.sqrt(1-np.cos(b_rad)**2*np.cos(l_rad)**2)
    # compute relative velocity and apply doppler shift 
    coords_galcen = coords.transform_to(assume["mw_halo"]["galcen"])
    coords_vec = np.asarray([coords_galcen.x.value,
                             coords_galcen.y.value,
                             coords_galcen.z.value])
    diff_vec = coords_vec.T - assume["mw_halo"]["vec_sun"]
    len_diff_vec = np.sqrt(np.sum(diff_vec**2, axis=1))
    diff_hat = (diff_vec.T/len_diff_vec).T
    v_parallel = np.sum(diff_hat*assume["mw_halo"]["v_sun"], axis=1)
    targets["v_rel"] = v_parallel # km/s, velocity of the sun along the target
                                  # line-of-sight in galactocentric frame
    for index in range(Ntargets):
        data[index]["lam"] *= (1.0 - targets[index]["v_rel"]/assume["mw_halo"]["c_kms"])
        data[index]["max_res"] *= (1.0 - targets[index]["v_rel"]/assume["mw_halo"]["c_kms"])
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


def parse_configs(config_filenames):
    assume = {}
    for filename in config_filenames:
        with open(filename, "rb") as f: 
            print(F"parsing {filename}")
            assume.update(tomli.load(f))
    # compute DM halo data
    assume["mw_halo"]["v_sun"] = np.asarray(assume["mw_halo"]["v_sun"])
    assume["mw_halo"]["sigma_v"] = (
        assume["mw_halo"]["sigma_v_kms"]/assume["mw_halo"]["c_kms"])
    assume["mw_halo"]["galcen"] = coord.Galactocentric(
        galcen_coord=coord.SkyCoord(ra=assume["mw_halo"]["ra_gc"]*u.deg, 
                                    dec=assume["mw_halo"]["dec_gc"]*u.deg),
        galcen_distance=assume["mw_halo"]["r_sun"]*u.kpc,
        galcen_v_sun=assume["mw_halo"]["v_sun"]*u.km/u.s,
        z_sun = assume["mw_halo"]["z_sun"]*u.pc)
    assume["mw_halo"]["vec_sun"] = np.asarray(
        [-assume["mw_halo"]["r_sun"], 
         0.0,
         assume["mw_halo"]["z_sun"]]) 
         # kpc, cartesain position of sun in galcen fame 
    # construct run paths 
    assume["run_data"]["paths"] = [
        "{}/{}".format(assume["paths"]["data_dir"], f) 
        for f in assume["run_data"]["filenames"]]
    return assume


def parse_sub(assume): 
    """ Truncate the blue GN-z11 spectrum at the start of the red one """
    data, targets = process_target_list(assume)
    gnz11_split = assume["run_setup"]["gnz11_split"] # specific to gnz-11
    gnz11_min = assume["run_setup"]["gnz11_min"] # specific to gnz-11
    for spec in data:
        if (spec["lam"][0] < gnz11_split) and (spec["name"] == "GN-z11"):
            select = (gnz11_min < spec["lam"]) & (spec["lam"] < gnz11_split)
            spec["lam"] = spec["lam"][select]
            spec["sky"] = spec["sky"][select]
            spec["error"] = spec["error"][select]
    return data, targets 