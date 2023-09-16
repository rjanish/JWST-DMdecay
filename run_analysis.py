#!/usr/bin/env python3
""" Run JWST DM search """


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

    data, targets = JWSTparse.process_target_list(assume.data_dir)
    # read in resolutions
    max_res_table = io.ascii.read(assume.maxres_path)
    max_res = {line["grating"]:line["max_res"] for line in max_res_table}
    for row in data:
        row["max_res"] = max_res[row["grating"]]

    # find conservative limit 
    chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux)
    chisq_threshold = 4
    frac_step = 0.5e-3

    lam_min = np.min(targets["lambda_min"])
    lam_max = np.max(targets["lambda_max"])
    dlam = lam_max*frac_step
    lam_test = np.arange(lam_min, lam_max+dlam, dlam)
    
    limit = np.zeros(lam_test.shape)
    Nsteps = limit.size
    Nstages = 10
    stage_size = int(np.ceil(Nsteps/Nstages))
    print("running analysis...\n" 
          "{} steps in {} stages of {} steps each"
          "".format(Nsteps, Nstages, stage_size))
    previous_stage = 0
    t0 = time.time()
    t00 = t0
    for index, lam_test_i in enumerate(lam_test):
        sol = opt.root_scalar(chisq_nb, 
                              args=(lam_test_i, chisq_threshold), 
                              bracket=[0, 100])
        limit[index] = sol.root
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

    m = convert.wavelength_to_mass(lam_test)
    limit_decayrate = convert.fluxscale_to_invsec(limit, assume.rho_s, assume.r_s)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m)    
    
    output_filename = "JWST-NIRSPEC-limits-{}.dat".format(run_name)
    header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(run_name))
    np.savetxt(output_filename, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=header)



