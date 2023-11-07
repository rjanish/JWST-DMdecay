#!/usr/bin/env python3
""" top-level assumptions and parameters for JWST DM search """

import tomli

import numpy as np
import astropy.coordinates as coord 
import astropy.units as u

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
    galcen = coord.Galactocentric(
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
