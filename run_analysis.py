#!/usr/bin/env python3
""" Run JWST DM search """


import os
import sys
import time 

import numpy as np
import scipy.optimize as opt
import astropy.io as io
import astropy.table as table

import DMdecayJWST as assume 
import JWSTparsedatafiles as JWSTparse
import MWDMhalo as mw
import conversions as convert 


if __name__ == "__main__":
    run_name = sys.argv[1]    
    conservative_results_dir = "{}/conservative".format(run_name)
    continuum_results_dir = "{}/continuum".format(run_name)
    try:
        os.mkdir(run_name)
        os.mkdir(conservative_results_dir)
        os.mkdir(continuum_results_dir)
    except FileExistsError:
        pass

    data, targets = JWSTparse.process_target_list(assume.data_dir)

    # only gnz11
    # targets = targets[targets["name"] == "GN-z11"]
    # data = [run for run in data if run["name"] == "GN-z11"]

    # clone test
    # targets = targets[targets["name"] == "GN-z11"]
    # data_gnz11 = [run for run in data if run["name"] == "GN-z11"]
    # data = [] 
    # for copy in range(10**2):
    #     data += data_gnz11

    print(len(data))
    print(len(targets))

    targets.write("{}/targets.html".format(run_name), 
                  format="ascii.html", overwrite=True)
    targets.write("{}/targets.dat".format(run_name), 
                  format="ascii.csv", overwrite=True)

    chisq_threshold = 4
    frac_mass_step = 0.5e-3
    lam_min = np.min(targets["lambda_min"])
    lam_max = np.max(targets["lambda_max"])
    dlam = lam_max*frac_mass_step
    lam_test = np.arange(lam_min, lam_max+dlam, dlam)

    # find conservative limit 
    reg_factor = 10
    chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux, reg_factor)
    limit = np.ones(lam_test.shape)*np.nan
    Nsteps = limit.size
    Nstages = 10
    stage_size = int(np.ceil(Nsteps/Nstages))
    print("running analysis...\n" 
          "{} steps in {} stages of {} steps each"
          "".format(Nsteps, Nstages, stage_size))
    previous_stage = 0
    t0 = time.time()
    t00 = t0
    method = "interp"
    uppers = [1e-4, 1e-2, 1, 100]
    for index, lam_test_i in enumerate(lam_test):
        lower = 0.0
        for upper in uppers:
            if chisq_nb(upper, lam_test_i, chisq_threshold, method) > 0:
                sol = opt.root_scalar(chisq_nb, 
                                      args=(lam_test_i, chisq_threshold, method), 
                                      bracket=[lower, upper]) # arbitrary upper lim 
                limit[index] = sol.root
                break
            lower = upper
        stage = index//stage_size
        if stage > previous_stage:
            dt = time.time() - t0
            print("{}/{}: {:.2f} sec".format(previous_stage + 1, Nstages, dt))
            previous_stage = stage
            t0 = time.time()
    dt = time.time() - t0
    print("{}/{}: {:.2f} sec".format(previous_stage + 1, Nstages, dt))
    dt_total = (time.time() - t00)/60
    print("total time: {:0.1f} min".format(dt_total))
    # convert to physics units
    m = convert.wavelength_to_mass(lam_test)
    limit_decayrate = convert.fluxscale_to_invsec(limit, assume.rho_s, assume.r_s)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m)    
    # output 
    output_path = ("{}/JWST-NIRSPEC-limits.dat"
                   "".format(conservative_results_dir))
    header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(run_name))
    np.savetxt(output_path, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=header)

    # find continuum-modeled limit 
    # lam0 = 1.75
    # chisq_pl = mw.MWchisq_powerlaw(data, mw.MWDecayFlux)
    # sol = opt.shgo(chis_pl, args=(lam_test_i, chisq_threshold))

