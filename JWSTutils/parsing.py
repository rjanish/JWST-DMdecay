"""
Process JWST datafiles, extracting relevant info for DM search 
"""


import astropy.io as io
import astropy.coordinates as coord
import astropy.units as u


def process_datafiles(datafile_paths, res_path):
    """ 
    Parse list of JWST datafiles and extract all info needed for the 
    DM analysis. The spectra in the solar baryocentric frame. 
      datafile_paths : list of paths to datafiles
      res_path : path to file containing table of spectral resolutions 
    """
    # read in resolutions
    res_table = io.ascii.read(res_path)
    res_dict = {line["grating"]:line["max_res"] for line in res_table}
        # For now simply take the max resolution for each 
        # grating as though uniform, and assume this is larger 
        # than the DM intrinsic width.  This is a good approximation. 
    print("processing {} datafiles...".format(len(datafile_paths)))
    out = []
    for path in datafile_paths:
        out_i = {}
        with io.fits.open(path) as hdul:
            out_i["lam"] = hdul[1].data["WAVELENGTH"]
            out_i["sky"] = hdul[1].data["BACKGROUND"]
            out_i["error"] = hdul[1].data["BKGD_ERROR"]
            out_i["name"] = hdul[0].header["TARGNAME"]
            out_i["ra"] = hdul[0].header["TARG_RA"] 
            out_i["dec"] = hdul[0].header["TARG_DEC"]
            out_i["skycoord"] = coord.SkyCoord(ra=out_i["ra"]*u.degree,
                                               dec=out_i["dec"]*u.degree,
                                               distance=1.0*u.kpc)  
                                               # placeholder distance
            out_i["b"] = out_i["skycoord"].galactic.b.degree
            out_i["l"] = out_i["skycoord"].galactic.l.degree
            out_i["instrument"] = hdul[0].header["INSTRUME"]
            out_i["detector"] = hdul[0].header["DETECTOR"]
            out_i["int_time"] = hdul[0].header["EFFINTTM"]
            out_i["lambda_min"] = out_i["lam"][0]
            out_i["lambda_max"] = out_i["lam"][-1]
            out_i["path"] = path
            try:  # filter and grating exist only for NIRSpec
                filter_name = hdul[0].header["FILTER"].strip()
                grating_name = hdul[0].header["GRATING"].strip()
                res = res_dict[grating_name]
            except:
                filter_name = "None"
                grating_name = "None"  
                res = np.nan # need to get miri resolutions
            out_i["grating"] = grating_name
            out_i["filter"] = filter_name
            out_i["res"] = res
        out.append(out_i)
    return out


def table_from_list_of_dicts(list_of_dicts, exclude):
    col_names = [k for k in list_of_dicts[0].keys() if k not in exclude]
    