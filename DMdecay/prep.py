""" 
Prepare data and setup analysis 
"""


import os 
import tomli
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
import astropy.io as io
import astropy.table as table
import astropy.coordinates as coord
import astropy.units as u

from . import conversions as convert
from . import halo 


def parse_configs(config_filename):
    """ read toml config file specifying an analysis run """
    print(F"parsing {config_filename}")
    with open(config_filename, "rb") as f: 
        out = tomli.load(f)
    # DM halo setup
    out["halo"]["v_sun"] = np.asarray(out["halo"]["v_sun"])
    out["halo"]["sigma_v"] = (
        out["halo"]["sigma_v_kms"]/convert.c_kms)
    out["halo"]["galcen"] = coord.Galactocentric(
        galcen_coord=coord.SkyCoord(ra=out["halo"]["ra_gc"]*u.deg, 
                                    dec=out["halo"]["dec_gc"]*u.deg),
        galcen_distance=out["halo"]["r_sun"]*u.kpc,
        galcen_v_sun=out["halo"]["v_sun"]*u.km/u.s,
        z_sun = out["halo"]["z_sun"]*u.pc)
    out["halo"]["pos_sun"] = np.asarray([-out["halo"]["r_sun"], 
                                         0.0,
                                         out["halo"]["z_sun"]]) # kpc
    # run setup 
    out["run"]["paths"] = ["{}/{}".format(out["system"]["data_dir"], f) 
                           for f in out["run"]["filenames"]]
    return out


def doppler_correction(spec, frame):
    """
    compute relative velocity and apply doppler shift 
    """    
    coords_galcen = spec["skycoord"].transform_to(frame["galcen"])
    coords_vec = np.asarray([coords_galcen.x.value,
                             coords_galcen.y.value,
                             coords_galcen.z.value])
    diff_vec = coords_vec.T - frame["pos_sun"]
    len_diff_vec = np.sqrt(np.sum(diff_vec**2))
    diff_hat = (diff_vec.T/len_diff_vec).T
    v_parallel = np.sum(diff_hat*frame["v_sun"])
    v_rel_kms = v_parallel # velocity of the sun along the target 
                           # line-of-sight in galactocentric frame in 
                           # km/s, positive value indicates sun is 
                           # moving towards the line-of-sight
    v_rel = v_rel_kms/convert.c_kms
    spec["lam"] *= (1.0 - v_rel)
    spec["res"] *= (1.0 - v_rel)
    spec["v_rel"] = v_rel
    # no need to rescale the spectra, as both the observed and 
    # model spectra are rescaled identically
    return


def add_Dfactor(spec, frame):
    D_unitless = halo.compute_halo_Dfactor(spec["b"], spec["l"], 
                                           halo.NFWprofile, 
                                           frame["r_sun"]/frame["r_s"])
    spec["D"] = D_unitless*frame["rho_s"]*frame["r_s"]
    return 
    

def get_fits_from_tree(datadir):
    """ get all *.fits files from directory tree starting at datadir """
    datafile_paths = []
    for current_path, current_dirnames, current_filenames in os.walk(datadir):
        datafile_paths += [os.path.join(current_path, f) 
                           for f in current_filenames
                           if f[-5:]==".fits"]
    return datafile_paths


def get_mass_samples(data, configs):    
    # generate mass sampling
    l_initial = np.min([spec["lambda_min"] for spec in data])
    l_final = np.max([spec["lambda_max"] for spec in data])
    test_lams = [l_initial]
    while test_lams[-1] < l_final:
        dlam_i = 2*np.min([
            halo.sigma_from_fwhm(spec["res"], test_lams[-1], 
                                 configs["halo"]["sigma_v"]) 
            for spec in data])*configs["analysis"]["inflate"] 
        test_lams.append(test_lams[-1] + dlam_i)
    test_lams = np.asarray(test_lams[1:-1]) # strip to stay inside data bounds
    return test_lams


@dataclass
class SpecSet:
    """ A set of blank sky spectra"""
    N_specs: int
    flux: list
    error: list
    lam: list
    N_pts: np.ndarray
    lam_limits: np.ndarray
    galactic_coords: np.ndarray
    int_time: np.ndarray
    inst_res: np.ndarray
    v_rel: np.ndarray
    D: np.ndarray

    def plot(self, ax=None, error_band=0.5, mask=None,
             distinct_plot_args=None, **common_plot_args):
        """ plot all spectra in the set """
        if ax is None:
            fig, ax = plt.subplots()
        if distinct_plot_args is None:
            distinct_plot_args = [{}]*self.N_specs
        for i in range(self.N_specs):
            mask_i = np.ones(self.N_pts[i], dtype=bool) if mask is None \
                     else mask[i]
            lines = ax.step(self.lam[i][mask_i], self.flux[i][mask_i], 
                            where="mid", **distinct_plot_args[i], 
                            **common_plot_args)
            if error_band:
                ax.fill_between(self.lam[i][mask_i], 
                                self.flux[i][mask_i] - self.error[i][mask_i], 
                                self.flux[i][mask_i] + self.error[i][mask_i], 
                                color=lines[0].get_color(), 
                                step='mid', alpha=error_band)
        return ax


def SpecSetfromList(list_of_dicts):
    """ 
    Create a SpecSet from a list of spectra, each stored as a dictionary
    """
    N_specs = len(list_of_dicts)
    N_pts = np.zeros(N_specs, dtype=int)
    lam_limits = np.zeros((N_specs, 2))
    galactic_coords = np.zeros((N_specs, 2))
    int_time = np.zeros(N_specs)
    inst_res = np.zeros(N_specs)
    v_rel = np.zeros(N_specs)
    D = np.zeros(N_specs)
    lam, flux, error = [], [], []
    for i, d in enumerate(list_of_dicts):
        N_pts[i] = len(d['sky'])
        print(N_pts[i])
        lam_limits[i] = [np.min(d['lam']), np.max(d['lam'])]
        galactic_coords[i] = [d['b'], d['l']]
        int_time[i] = d['int_time']
        inst_res[i] = d['res']
        v_rel[i] = d['v_rel']
        D[i] = d['D']
        lam.append(d['lam'])
        flux.append(d['sky'])
        error.append(d['error'])
    return SpecSet(N_specs=N_specs, flux=flux, error=error,
                   lam=lam, N_pts=N_pts, lam_limits=lam_limits,
                   galactic_coords=galactic_coords,
                   int_time=int_time, inst_res=inst_res,
                   v_rel=v_rel, D=D)

