"""
Run DM analysis on sky spectra from GNz-11 observations, taking the
redder spectrum only in the region of overlap (as in v1 2310.15395).
"""


import sys
import os 
import time
import tomli
import multiprocessing as mlproc
from functools import partial
import pickle as pkl
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

import DMdecay as dmd
import JWSTutils as jwst

import gnz11_split  # driver script for existing run

def run_mock_set(data, configs, Ntrials, test_name,
                 input_factor, noise_factor, lam_test):
    """
    Fit a set of Ntrials mock specttra with the same injected signal of strength limit*input_factor [code units] located at lam_test [micron].
    """
    test_results = np.empty((Ntrials, 5))
        # cols: lam, true raw limit, injected flux, 
        #       mock best fit, mock raw limit      
    test_results[:, 0] = lam_test
    roll = np.random.default_rng()
    for trial in range(Ntrials):
        mock_data = deepcopy(data)
        # fit with extra noise and no injected signal
        for mock_spec in mock_data:
            valid = mock_spec["error"] > 0 
            scale = noise_factor*mock_spec["error"][valid]
            noise_draw = roll.normal(loc=0, scale=scale)
            mock_spec["sky"][valid] += noise_draw
            mock_spec["error"][valid] *= np.sqrt(1 + noise_factor**2)
        no_sig = dmd.linesearch.find_raw_limit(configs, mock_data, lam_test)
        test_results[trial, 1] = no_sig[4][0]              # raw limit 
        test_results[trial, 2] = input_factor*no_sig[4][0] # injected signal
        # inject signal and refit
        for mock_spec in mock_data:
            sigma_full = dmd.halo.sigma_from_fwhm(mock_spec["res"], lam_test,
                                                  configs["halo"]["sigma_v"]) 
            mock_spec["sky"] += \
                dmd.halo.MWDecayFlux(mock_spec["lam"], lam_test,
                                     test_results[trial, 2], 
                                     mock_spec["D"],
                                     sigma_full)
        injected = dmd.linesearch.find_raw_limit(configs, mock_data, lam_test)
        test_results[trial, 3] = injected[5][0] # new best fit
        test_results[trial, 4] = injected[4][0] # new raw limit
        # save for comparison 
        output = {"lam_test":lam_test, "data":mock_data, 
                  "G":test_results[trial, 2],
                  "new_bf":injected[5][0], 
                  "new_rawlimit":injected[4][0]}
        run_name = configs["run"]["name"]
        output_path = \
            F"{run_name}/injection/{test_name}-trial{trial}-{lam_test:0.6f}.pkl"
        with open(output_path, "wb") as jar:
            pkl.dump(output, jar)
    return test_results

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

    # load existing raw limits and setup injection test  
    rawlimits = np.loadtxt(
        F"{run_name}/{prev_configs['run']['rawlimits_filename']}")
    print("Running injection test... ")
    valid = np.isfinite(rawlimits[:, 1])
    skipping = cur_configs["injection"]["skipping"]
    mass_to_test = rawlimits[valid, :][::skipping, 0]
    lam_to_test = dmd.conversions.mass_to_wavelength(mass_to_test)
    Nlams = lam_to_test.shape[0]
    exisitng_limit_invsec = rawlimits[valid, :][::skipping, 1]
    decayrate_to_test = \
        dmd.conversions.invsec_to_fluxscale(exisitng_limit_invsec)
    Ntrials = cur_configs["injection"]["Ntrials"]
    test_results = np.empty((Nlams*Ntrials, 5))
        # col: lam, old raw limit, injected flux, new best fit, new raw limit

    # run injection test 
    test_name = cur_configs["injection"]["name"]
    input_factor = cur_configs["injection"]["input_factor"]
    noise_factor = cur_configs["injection"]["noise_factor"]
    wrapper = partial(run_mock_set, data, prev_configs, 
                      Ntrials, test_name, input_factor, noise_factor)
    cores = mlproc.cpu_count() - 1
    with mlproc.Pool(processes=cores) as pool:
        iteration = tqdm(pool.imap(wrapper, lam_to_test), 
                         total=Nlams, position=0)
        test_results = np.vstack(list(iteration))
    test_path = F"{run_name}/{test_name}-injection_results.dat"
    np.savetxt(test_path, test_results)

    # print stats
    noise_estimate = np.min([np.median(spec["error"]) for spec in data])
    discrepancy = (test_results[:, 4] - test_results[:, 2])/noise_estimate
    num_negative = np.sum(discrepancy < 0)
    frac_negative = num_negative/discrepancy.size
    expected_error = np.sqrt(1/discrepancy.size)
    neg_frac_str = (F"negative fraction = {100*frac_negative:0.1f}%"
                    F" +/- {100*expected_error:0.1f}%")
    print(neg_frac_str)

    fig, ax = plt.subplots()
    ax.set_xlabel("relative discrepancy")
    ax.set_ylabel("count")
    ax.set_title("relative discrepancy between raw limit and injected flux")
    Nhist = 50
    margin_factor = 1.1
    positive_disc = discrepancy >= 0
    max_right = discrepancy[positive_disc].max()*margin_factor
    positive_bins = np.linspace(0, max_right, Nhist)
    positive_hist = ax.hist(discrepancy[positive_disc],
                            bins=positive_bins, histtype='step', color='black')
    if num_negative > 0:
        bin_width = positive_hist[1][1] - positive_hist[1][0]
        negative_bins = -np.arange(0, -discrepancy.min(), bin_width)[::-1]
        ax.hist(discrepancy[~positive_disc], bins=negative_bins, 
                histtype='step', color='red')
    ax.axvline(0, color='gray', linestyle='--')
    ax.add_artist(
        AnchoredText(neg_frac_str, loc='upper right', 
                     prop=dict(color="firebrick"), frameon=False))
    fig.savefig(F"{run_name}/{test_name}-injection_discrepancy.png")
    plt.show()
