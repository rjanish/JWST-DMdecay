"""
Run DM analysis on sky spectra from GNz-11 observations, taking the
redder spectrum only in the region of overlap (as in v1 2310.15395).
"""


import sys
import os 
import time
import tomli
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from scipy.stats import loguniform
from tqdm import tqdm

import DMdecay as dmd
import JWSTutils as jwst

import gnz11_split  # driver script for existing run


if __name__ == "__main__":
    t0 = time.time()
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
    data = gnz11_split.partition_gnz11(data, 
                                       prev_configs["run"]["lambda_min"],
                                       prev_configs["run"]["lambda_split"])
    for spec in data:
        dmd.prep.doppler_correction(spec, prev_configs["halo"])
        dmd.prep.add_Dfactor(spec, prev_configs["halo"])
    trial_data = deepcopy(data)

    # get results from existing run 
    # fluxlimits_path = F"{run_name}/{prev_configs['run']['fluxlimits_filename']}"
    # rawlimits_path = F"{run_name}/{prev_configs['run']['rawlimits_filename']}"
    # bestfits_path  = F"{run_name}/{prev_configs['run']['bestfits_filename']}"
    # pc_path  = F"{run_name}/{prev_configs['run']['pc_filename']}"
    # lineoutput_path  = F"{run_name}/{prev_configs['run']['lineoutput_filename']}"
    
    rawlimits = np.loadtxt(
        F"{run_name}/{prev_configs['run']['rawlimits_filename']}")
    print("Running injection test... ")
    Ntrials = cur_configs["injection"]["Ntrials"]
    valid = np.isfinite(rawlimits[:, 1])
    to_test = np.repeat(
        rawlimits[valid, :][::cur_configs["injection"]["skipping"], :],
        Ntrials, axis=0)
    test_results = np.empty((to_test.shape[0], 5))
    iteration = tqdm(enumerate(to_test), total=to_test.shape[0])
    for i, (m_test, decayrate_test, _) in iteration:
        lam_test = dmd.conversions.mass_to_wavelength(m_test)   # micron
        test_results[i, 0] = lam_test
        existing_limit = \
            dmd.conversions.invsec_to_fluxscale(decayrate_test) # code units
        test_results[i, 1] = existing_limit
        G = existing_limit*loguniform.rvs(0.33, 3) 
        test_results[i, 2] = G  # input signal
        for spec_index in range(len(trial_data)):
            # get signal sigma_lambda, combining instrumental and DM velocity 
            sigma_full = dmd.halo.sigma_from_fwhm(
                trial_data[spec_index]["res"], lam_test, 
                prev_configs["halo"]["sigma_v"]) 
            trial_data[spec_index]["sky"] = (
                data[spec_index]["sky"] +
                dmd.halo.MWDecayFlux(trial_data[spec_index]["lam"], lam_test,
                                     G, trial_data[spec_index]["D"], 
                                     sigma_full))
        new_rawlimits = \
            dmd.linesearch.find_raw_limit(prev_configs, trial_data, lam_test)
        test_results[i][3] = new_rawlimits[5][0] # new best fit
        test_results[i][4] = new_rawlimits[4][0] # new raw limit
    test_path = F"{run_name}/injection_results.dat"
    np.savetxt(test_path, test_results)

    # print stats
    discrepancy = (test_results[:, 4] - test_results[:, 2])/test_results[:, 1]
    num_negative = np.sum(discrepancy < 0)
    frac_negative = num_negative/discrepancy.size
    print(F"fraction of negative discrepancies: {100*frac_negative:0.4f}%")

    fig, ax = plt.subplots()
    ax.set_xlabel("relative discrepancy")
    ax.set_ylabel("count")
    ax.set_title("relative discrepancy between raw limit and injected flux")
    Nhist = 20
    positive_bins = np.linspace(0, discrepancy.max() + 1, Nhist)
    positive_hist = ax.hist(discrepancy[discrepancy >= 0],
                            bins=positive_bins, histtype='step', color='black')
    if num_negative > 0:
        bin_width = positive_hist[1][1] - positive_hist[1][0]
        negative_bins = -np.arange(0, -discrepancy.min(), bin_width)[::-1]
        ax.hist(discrepancy[discrepancy < 0], bins=negative_bins, 
                histtype='step', color='red')
    ax.axvline(0, color='gray', linestyle='--')
    ax.add_artist(
        AnchoredText(F"negative fraction = {100*frac_negative:0.2f}%", loc='upper right', prop=dict(color="firebrick"), frameon=False))
    fig.savefig(F"{run_name}/injection_discrepancy.png")
    plt.show()
