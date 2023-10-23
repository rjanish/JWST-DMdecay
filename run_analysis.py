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


def sigma_from_fwhm(fwhm, lam0):
    sigma_inst = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.sqrt(sigma_inst**2 + (lam0*assume.sigma_v)**2)

def chisq_dm_totalflux(rate, lam0, fixed_list, x_list, 
             y_list, sigma_y_list, shift=0):
    total = 0.0
    for i in range(len(x_list)):
        model = mw.MWDecayFlux_old(x_list[i], fixed_list[i][0], rate, 
                                   fixed_list[i][1], fixed_list[i][2])
        resid = np.zeros(model.shape)
        dm_large = model > y_list[i]
        resid[dm_large] = ((model[dm_large] - y_list[i][dm_large]) / 
                           (sigma_y_list[i][dm_large])) 
            # use unscaled error - this doesn't matter much here, 
            # the result depends on the error at less-than-O(1)
            # level so long as error < flux
        total += np.sum(resid**2)
    return total - shift


if __name__ == "__main__":
    run_name = sys.argv[1]
    # targeting_path = sys.argv[2]
    # positives_mode = sys.argv[3]

    conservative_results_dir = "{}/conservative".format(run_name)
    continuum_results_dir = "{}/continuum".format(run_name)
    try:
        os.mkdir(run_name)
        os.mkdir(conservative_results_dir)
        os.mkdir(continuum_results_dir)
    except FileExistsError:
        pass
    with open("{}/{}.txt".format(run_name, run_name), 'w') as run_log:
        # run_log.write("{} {} {}\n".format(*sys.argv[1:4]))
        run_log.write("{}\n".format(sys.argv[1]))

    # data, targets = JWSTparse.process_target_list(assume.data_dir)
    # Nrows_full = len(data)
    # if targeting_path != "all":
    #     with open(targeting_path) as targeting_file:
    #         targeting = [line.strip() for line in targeting_file.readlines()]
    #         data = [row for row in data if row["name"] in targeting]
    #         select = np.zeros(Nrows_full, dtype=bool)
    #         for target in targeting:
    #             matches = (targets["name"] == target)
    #             select = select | (targets["name"] == target)
    #         targets = targets[select]

    # data, targets = assume.parse_gnz11()
    data, targets = assume.parse_sub(assume.gnz11_paths )

    targets.write("{}/targets.html".format(run_name), 
                  format="ascii.html", overwrite=True)
    targets.write("{}/targets.dat".format(run_name), 
                  format="ascii.csv", overwrite=True)
    Nrows = len(data)
    print("analyzing {} spectra".format(Nrows))

    chisq_threshold = 2.71 #4

    extract_list = lambda list_of_dicts, key: [d[key] for d in list_of_dicts]
    sky_list = extract_list(data, "sky")
    lam_list = extract_list(data, "lam")
    error_list = extract_list(data, "error")

    # frac_mass_step = 1e-3
    # lam_min = np.min([np.min(lam) for lam in lam_list])
    # lam_max = np.max([np.max(lam) for lam in lam_list])
    # dlam = lam_max*frac_mass_step
    # lam_test = np.arange(lam_min, lam_max+dlam, dlam)

    lstart = [spec["lam"][0] for spec in data]
    lend = [spec["lam"][-1] for spec in data]
    l_initial = np.min(lstart)
    l_final = np.max(lend)
    lam_test = [l_initial]
    while lam_test[-1] < l_final:
        dlam_i = 2*np.min([sigma_from_fwhm(spec["max_res"], lam_test[-1]) 
                           for spec in data]) #rescale for testing
        lam_test.append(lam_test[-1] + dlam_i)
    lam_test = np.asarray(lam_test[1:-1]) # the last one is always outside the range

    # find conservative limit 
    # chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux, positives_mode)
    # chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux_old, "any")

    limit = np.ones(lam_test.shape)*np.nan
    # chisq_itemized = np.ones((Nrows, lam_test.size))
    Nsteps = limit.size
    Nstages = 10
    stage_size = int(np.ceil(Nsteps/Nstages))
    print("running analysis...\n" 
          "{} steps in {} stages of {} steps each"
          "".format(Nsteps, Nstages, stage_size))
    previous_stage = 0
    t0 = time.time()
    t00 = t0
    uppers = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    for index, lam0 in enumerate(lam_test):
        fixed_list = [[lam0, spec["D"], sigma_from_fwhm(spec["max_res"], lam0)]
                      for spec in data]
        lower = 0.0
        for upper in uppers:
            chisq_test = chisq_dm_totalflux(upper, lam0, fixed_list, lam_list,
                                            sky_list, error_list,
                                            chisq_threshold)
            if chisq_test > 0:
                sol = opt.root_scalar(chisq_dm_totalflux, 
                                      args=(lam0, fixed_list, lam_list,
                                            sky_list, error_list, 
                                            chisq_threshold), 
                                      bracket=[lower, upper])
                limit[index] = sol.root
                # chisq_itemized[:, index] = chisq_nb(sol.root, lam0, 
                #                                     0, "itemized")
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
    limit_decayrate = convert.fluxscale_to_invsec(limit)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m)    
    # output 
    limits_path = ("{}/JWST-NIRSPEC-limits.dat"
                   "".format(conservative_results_dir))
    limits_header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(run_name))
    np.savetxt(limits_path, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=limits_header)
    # itemized_path = ("{}/JWST-NIRSPEC-chisq-itemized.dat"
    #                  "".format(conservative_results_dir))
    # itemized_header = ("DM decay chisq vs mass and spectra \n"
    #           "JWST NIRSPEC run {}\n"
    #           "values are the contribution to chisq of the given"
    #           "spectrum and DM mass, with the total chisq = 4\n"
    #           "row = spectrum\n"
    #           "col = 4pi/mass\n"
    #           "".format(run_name))
    # np.savetxt(itemized_path, chisq_itemized,
    #            header=itemized_header)


    # find conservative limit 
    # chisq_nb = mw.MWchisq_nobackground(data, mw.MWDecayFlux, positives_mode)
    # limit = np.ones(lam_test.shape)*np.nan
    # window_factor = 15
    # v_dm = 1e-3
    # # chisq_itemized = np.ones((Nrows, lam_test.size))
    # Nsteps = limit.size
    # Nstages = 10
    # stage_size = int(np.ceil(Nsteps/Nstages))
    # print("running analysis...\n" 
    #       "{} steps in {} stages of {} steps each"
    #       "".format(Nsteps, Nstages, stage_size))
    # previous_stage = 0
    # t0 = time.time()
    # t00 = t0
    # for index, lam0 in enumerate(lam_test):
    #     lam_start = lam0*(1 - window_factor*0.5*v_dm) 
    #     lam_end = lam0*(1 + window_factor*0.5*v_dm)
    #     select = (lam_start < data["lam"]) & (data["lam"] < lam_end)
    #     flux = data["sky"][select]
    #     lam = data["lam"][select]
    #     sigma_flux = data["error"][select]

        
    #     lower = 0.0
    #     for upper in uppers:
    #         if chisq_nb(upper, lam0, chisq_threshold, "total") > 0:
    #             sol = opt.root_scalar(chisq_nb, 
    #                                   args=(lam0, 
    #                                         chisq_threshold, 
    #                                         "total"), 
    #                                   bracket=[lower, upper])
    #             limit[index] = sol.root
    #             chisq_itemized[:, index] = chisq_nb(sol.root, lam0, 
    #                                                 0, "itemized")
    #             break
    #         lower = upper
    #     stage = index//stage_size
    #     if stage > previous_stage:
    #         dt = time.time() - t0
    #         print("{}/{}: {:.2f} sec".format(previous_stage + 1, Nstages, dt))
    #         previous_stage = stage
    #         t0 = time.time()
    # dt = time.time() - t0
    # print("{}/{}: {:.2f} sec".format(previous_stage + 1, Nstages, dt))
    # dt_total = (time.time() - t00)/60
    # print("total time: {:0.1f} min".format(dt_total))
    # # convert to physics units
    # m = convert.wavelength_to_mass(lam_test)
    # limit_decayrate = convert.fluxscale_to_invsec(limit)    
    # limit_g = convert.decayrate_to_axion_g(limit_decayrate, m)    
    # # output 
    # limits_path = ("{}/JWST-NIRSPEC-limits.dat"
    #                "".format(conservative_results_dir))
    # limits_header = ("DM decay limits vs mass \n"
    #           "JWST NIRSPEC run {}\n"
    #           "mass [ev]    lifetime [sec]    "
    #           "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
    #           "".format(run_name))
    # np.savetxt(limits_path, 
    #            np.column_stack((m, limit_decayrate, limit_g)),
    #            header=limits_header)
    # itemized_path = ("{}/JWST-NIRSPEC-chisq-itemized.dat"
    #                  "".format(conservative_results_dir))
    # itemized_header = ("DM decay chisq vs mass and spectra \n"
    #           "JWST NIRSPEC run {}\n"
    #           "values are the contribution to chisq of the given"
    #           "spectrum and DM mass, with the total chisq = 4\n"
    #           "row = spectrum\n"
    #           "col = 4pi/mass\n"
    #           "".format(run_name))
    # np.savetxt(itemized_path, chisq_itemized,
    #            header=itemized_header)


