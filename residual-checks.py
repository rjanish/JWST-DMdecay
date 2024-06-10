"""
Refit split GNz-11 data, check stastics of the continuum fit residuals. 
"""

import sys
import os
import tomli
from copy import deepcopy
from multiprocessing import Pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import DMdecay as dmd
import JWSTutils as jwst
from gnz11_split import partition_gnz11  # gnz11-specific data preprocessing


def refit_data(data, configs, lam_test):
    fit_results = \
        dmd.linesearch.find_raw_limit(configs, data, lam_test)
    [[lmin, lmax], spec_list, knots, error_scale_factors, 
            [limit_rate, limit_knots], [best_rate, best_knots],
            delta_chisq, lam_list_msk, error_list_msk, lam0,
            sky_list, lam_list, error_list, fixed_list, 
            spec_list, res_list, mask_list, sky_list_msk] = fit_results
    if best_knots is None:
        print(F"Failed to fit {lam_test}")
        return []
    return dmd.linesearch.spline_residual(limit_knots, knots, lam_list_msk,
                                          sky_list_msk, error_list_msk)
    

if __name__ == "__main__":
    cur_config_path = sys.argv[1]
    with open(cur_config_path, "rb") as f: 
        cur_configs = tomli.load(f)
    run_dir, _ = os.path.split(cur_config_path)
    prev_config_path = F"{run_dir}/{cur_configs['previous']['config_path']}"
    prev_configs = dmd.prep.parse_configs(prev_config_path)
    run_name = prev_configs["run"]["name"]
    try:
        os.mkdir(run_name)
    except FileExistsError:
        pass

    # prep data and metadata as in existing run
    data = jwst.process_datafiles(prev_configs["run"]["paths"], 
                                  prev_configs["system"]["res_path"])
    data = partition_gnz11(data, 
                                       prev_configs["run"]["lambda_min"],
                                       prev_configs["run"]["lambda_split"])
    for spec in data:
        dmd.prep.doppler_correction(spec, prev_configs["halo"])
        dmd.prep.add_Dfactor(spec, prev_configs["halo"])
    
    # load existing results
    rawlimits = np.loadtxt(
        F"{run_name}/{prev_configs['run']['rawlimits_filename']}")
    valid = np.isfinite(rawlimits[:, 1])
    skipping = cur_configs["tests"]["skipping"]
    to_test = rawlimits[valid, :][::skipping, 0]

    print("Running re-fits") 
    wrapper = partial(refit_data, data, prev_configs)
    cores = cur_configs["tests"]["cores"]
    with Pool(processes=cores) as p:
        all_residuals = p.map(wrapper, to_test)

    # plot residuals
    fig, ax = plt.subplots()
    _, bins, _ = ax.hist(np.concatenate(all_residuals), bins=100, 
                         histtype="step", density=True)
    ax.plot(bins, norm.pdf(bins, loc=0, scale=1), color="red", linestyle="--")
    ax.axvline(0, color="black", linestyle="--")


            
    plt.show()

