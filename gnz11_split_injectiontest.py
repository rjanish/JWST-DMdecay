"""
Run DM analysis on sky spectra from GNz-11 observations, taking the
redder spectrum only in the region of overlap (as in v1 2310.15395).
"""


import sys
import os 
import time
import tomli
from multiprocessing import Pool
from functools import partial

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText
from tqdm import tqdm

import DMdecay as dmd
import JWSTutils as jwst

import gnz11_split  # driver script for existing run

def run_mock_set(data, prev_configs, Ntrials, test):
    """
    Fit a set of Ntrials mock specttra with the same injected signal of strength test[1] [code units] located at test[0] [micron].
    """
    lam_test, input_strength = test
    test_results = np.empty((Ntrials, 5))
        # cols: lam, true raw limit, injected flux, 
        #       mock best fit, mock raw limit       
    test_results[:, 0] = lam_test
    old_output_redo = dmd.linesearch.find_raw_limit(prev_configs, data, 
                                                    lam_test)
    test_results[:, 1] = old_output_redo[4][0]
    # generate mock data list
    test_results[:, 2] = input_strength # injected signal
    spec_list = old_output_redo[1]
    lam_list_msk = old_output_redo[7]
    error_list_msk = old_output_redo[8]
    sky_list = old_output_redo[10]
    mask_list = old_output_redo[16]
    mock_data = []
    fake_signal = []
    for j, spec_i in enumerate(spec_list):
        mock_data.append({})
        mock_data[-1]["res"] = data[spec_i]["res"]
        mock_data[-1]["D"] = data[spec_i]["D"]
        mock_data[-1]["lam"] = lam_list_msk[j]
        mock_data[-1]["error"] = error_list_msk[j]#*np.sqrt(2)
        sigma_full = dmd.halo.sigma_from_fwhm(
            data[spec_i]["res"], lam_test, 
            prev_configs["halo"]["sigma_v"]) 
        fake_signal.append(
            dmd.halo.MWDecayFlux(lam_list_msk[j], lam_test, input_strength, 
                                 data[spec_i]["D"], sigma_full))
    # add noise and fit mocks
    for trial in range(Ntrials):
        for j, spec_i in enumerate(spec_list):
            # Npts = error_list_msk[j].size
            # roll = np.random.default_rng()
            # noise_draw = roll.normal(loc=0, 
            #                          scale=error_list_msk[j], size=Npts)
            noise_draw = 0
            mock_data[-1]["sky"] = \
                sky_list[j][~mask_list[j]] + noise_draw + fake_signal[j]
        # print(F"fitting trial {trial} of lam={lam_test:.2f} micron")
        try:
            new_rawlimits = dmd.linesearch.find_raw_limit(prev_configs, 
                                                        mock_data, lam_test)
            test_results[trial, 3] = new_rawlimits[5][0] # new best fit
            test_results[trial, 4] = new_rawlimits[4][0] # new raw limit
        except ValueError:
            print("value error: skipping {lam_test:.2f} micron")
            test_results[trial, 3] = np.nan
            test_results[trial, 4] = np.nan
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

    # load existing raw limits and run injection test  
    rawlimits = np.loadtxt(
        F"{run_name}/{prev_configs['run']['rawlimits_filename']}")
    print("Running injection test... ")
    Ntrials = cur_configs["injection"]["Ntrials"]
    # seed = cur_configs["injection"]["seed"]
    cores = cur_configs["injection"]["cores"]
    valid = np.isfinite(rawlimits[:, 1])
    skipping = cur_configs["injection"]["skipping"]
    mass_to_test = rawlimits[valid, :][::skipping, 0]
    lam_to_test = dmd.conversions.mass_to_wavelength(mass_to_test)
    Nlams = lam_to_test.shape[0]
    exisitng_limit_invsec = rawlimits[valid, :][::skipping, 1]
    input_scale_factor = 0.1
    decayrate_to_test = dmd.conversions.invsec_to_fluxscale(
        exisitng_limit_invsec*input_scale_factor)
    test_results = np.empty((Nlams*Ntrials, 5))
        # col: lam, old raw limit, injected flux, new best fit, new raw limit

    # for i in tqdm(range(Nlams), total=Nlams):
    #     test_results[i*Ntrials:(i+1)*Ntrials, :] = \
    #         run_mock_set(data, prev_configs, lam_to_test[i], 
    #                      decayrate_to_test[i], Ntrials, seed)
        
    wrapper = partial(run_mock_set, data, prev_configs, Ntrials)
    inputs = zip(lam_to_test, decayrate_to_test)
    with Pool(processes=cores) as pool:
        iteration = tqdm(pool.imap(wrapper, inputs), total=Nlams, position=0)
        test_results = np.vstack(list(iteration))
    test_path = F"{run_name}/injection_results.dat"
    np.savetxt(test_path, test_results)

    # print stats
    discrepancy = (test_results[:, 4] - test_results[:, 2])/test_results[:, 2]
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
        AnchoredText(neg_frac_str, loc='upper right', 
                     prop=dict(color="firebrick"), frameon=False))
    fig.savefig(F"{run_name}/injection_discrepancy.png")
    plt.show()
