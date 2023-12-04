#!/usr/bin/env python3
"""
Run DM analysis on sky spectra from GNz-11 observations, taking the
redder spectrum only in the region of overlap (as in v1 2310.15395).
"""


import sys
import os 
import time

import pandas as pd 

import DMdecay as dmd
import JWSTutils as jwst


def partition_gnz11(data, lam_min, lam_split): 
    """ 
    Truncate the blue GN-z11 spectrum at the start of the red one. 
    """
    for spec in data:
        if (spec["lam"][0] < lam_split) and (spec["name"] == "GN-z11"):
            select = (lam_min < spec["lam"]) & (spec["lam"] < lam_split)
            spec["lam"] = spec["lam"][select]
            spec["sky"] = spec["sky"][select]
            spec["error"] = spec["error"][select]
    return data 


if __name__ == "__main__":
    t0 = time.time()
    config_path = sys.argv[1]
    configs = dmd.prep.parse_configs(config_path)
    try:
        os.mkdir(configs["run"]["name"])
    except FileExistsError:
        pass

    # prep data and metadata 
    data = jwst.process_datafiles(configs["run"]["paths"], 
                                  configs["system"]["res_path"])
    data = partition_gnz11(data, configs["run"]["lambda_min"],
                           configs["run"]["lambda_split"])
    print()
    for spec in data:
        dmd.prep.doppler_correction(spec, configs["halo"])
        dmd.prep.add_Dfactor(spec, configs["halo"])
    exclude = ["sky", "lam", "error", "skycoord"]
    col_names = [k for k in data[0].keys() if k not in exclude]
    table = pd.DataFrame(data, columns=col_names)
    table_filename = configs["run"]["name"] + "/obv_table"
    table.to_csv(table_filename + ".dat", sep='\t')
    table.to_html(table_filename + ".html")

    test_lams = dmd.prep.get_mass_samples(data, configs)
    dmd.linesearch.run(data, configs, test_lams)
    print()
    dmd.fluxlimit.run(data, configs, test_lams)
    print()
    print(F"total: {(time.time() - t0)/60.0:0.2f} mins")