""" 
Plot full spectrum with close-up of line model 
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

from tqdm import tqdm

import DMdecay as dmd
import JWSTutils as jwst
from gnz11_split import partition_gnz11

if __name__ == "__main__":
    config_filename = sys.argv[1]    
    lam0 = float(sys.argv[2])
    
    configs = dmd.prep.parse_configs(config_filename)
    # num_knots = configs["analysis"]["num_knots"]
    
    data = jwst.process_datafiles(configs["run"]["paths"], 
                                  configs["system"]["res_path"])
    data = partition_gnz11(data, configs["run"]["lambda_min"],
                           configs["run"]["lambda_split"])
    for spec in data:
        dmd.prep.doppler_correction(spec, configs["halo"])
        dmd.prep.add_Dfactor(spec, configs["halo"])
    num_specs = len(data)
    print()

    dofs = np.arange(5, 31)
    times = np.nan*np.ones(dofs.shape)
    limits = np.nan*np.ones(dofs.shape)
    resids = np.nan*np.ones(dofs.shape)
    
    fig, ax = plt.subplots()

    for j, num_knots in tqdm(enumerate(dofs), total=len(dofs)):
        configs["analysis"]["num_knots"] = num_knots
        t0 = time.time()
        raw_out = dmd.linesearch.find_raw_limit(configs, data, lam0)
        [[lmin, lmax], 
        spec_list, 
        knots, 
        error_scale_factors,
        [limit_rate, limit_knots], 
        [best_rate, best_knots],
        delta_chisq,
        lam_list_msk, 
        error_list_msk, 
        lam0_tmp,
        sky_list, 
        lam_list, 
        error_list, 
        fixed_list, 
        spec_list, 
        res_list, 
        mask_list, 
        sky_list_msk] = raw_out
        num_fitted_specs = len(spec_list)
        dt = time.time() - t0

        m0 = dmd.conversions.wavelength_to_mass(lam0)
        limit_decayrate = dmd.conversions.fluxscale_to_invsec(limit_rate)   
        limit_g = dmd.conversions.decayrate_to_axion_g(limit_decayrate, m0) 
        best_decayrate = dmd.conversions.fluxscale_to_invsec(best_rate)   
        best_g = dmd.conversions.decayrate_to_axion_g(best_decayrate, m0) 

        times[j] = dt
        limits[j] = limit_g
        
        residuals = []
        color=None
        for i in range(num_fitted_specs):
            start = i*num_knots
            end = (i + 1)*num_knots
            print(best_knots)
            bf_continuum_model = interp.CubicSpline(knots[start:end], 
                                                    best_knots[start:end])
            line = ax.plot(lam_list[i], 
                           bf_continuum_model(lam_list[i]), linestyle='-', 
                           marker='', alpha=0.2, color=color)
            color = line[0].get_color()
            ax.plot(knots, best_knots, linestyle='', 
                    marker='o', alpha=0.2, color=color)
            if j == 0:
                ax.step(lam_list[i], 
                           sky_list[i], linestyle='-', 
                           marker='', alpha=0.8, color='black')
            # residuals 
            residuals.append(
                (sky_list[i] - bf_continuum_model(lam_list[i]))**2)
        residual = np.mean(np.concatenate(residuals))
        resids[j] = residual


        print(F"fit line in {num_fitted_specs} spectra with {num_knots} dof spline")
        print(F"       lam0 = {lam0:0.4f} micron\n"
            F"       lmin = {lmin:0.4f} micron\n"
            F"       lmax = {lmax:0.4f} micron\n"
            F"  best_rate = {best_rate:0.2e} [fluxscale units]\n"
            F"delta_chisq = {delta_chisq:0.2f}\n"
            F" limit_rate = {limit_rate:0.2e} [fluxscale units]\n"
            F"         m0 = {m0:0.4f} eV\n"
            F"     best_g = {best_g:0.2e} GeV^{-1}\n"
            F"    limit_g = {limit_g:0.2e} GeV^{-1}")
        
        print(F"    mean square residual = {residual:0.2e}")

        # pc_input = [raw_out[9], raw_out[7], raw_out[8], 
        #             raw_out[2], raw_out[5][1], raw_out[1]]
        # lam0_tmt, pc_limit, all_uppers = dmd.linesearch.find_pc_limit(configs, 
        #                                                             data, 
        #                                                             pc_input)
        # pc_decayrate = dmd.conversions.fluxscale_to_invsec(pc_limit)   
        # pc_g = dmd.conversions.decayrate_to_axion_g(pc_decayrate, m0) 
        # print(F"       pc_g = {pc_g:0.2e} GeV^{-1}")

        print(F"\nelapsed time: {time.time() - t0:0.2f} seconds")

    fig, ax = plt.subplots()
    ax.plot(dofs, limits, 'o-')
    ax.set_ylabel("limit [GeV^-1]")

    fig, ax = plt.subplots()
    ax.plot(dofs, times, 'o-')
    ax.set_ylabel("time [s]")

    noise_estimate = np.min([np.median(spec["error"]) for spec in data])
    fig, ax = plt.subplots()
    ax.plot(dofs, np.sqrt(resids)/noise_estimate, 'o-')
    ax.set_ylabel("mean square residual")

    plt.show()