#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import DMdecay as dmd
import JWSTutils as jwst
from gnz11_split import partition_gnz11

if __name__ == "__main__":
    config_filename = sys.argv[1]    
    lam0 = float(sys.argv[2])
    configs = dmd.prep.parse_configs(config_filename)
    num_knots = configs["analysis"]["num_knots"]
    
    data = jwst.process_datafiles(configs["run"]["paths"], 
                                  configs["system"]["res_path"])
    data = partition_gnz11(data, configs["run"]["lambda_min"],
                           configs["run"]["lambda_split"])
    for spec in data:
        dmd.prep.doppler_correction(spec, configs["halo"])
        dmd.prep.add_Dfactor(spec, configs["halo"])
    num_specs = len(data)
    print()

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
     res_list] = raw_out
    num_fitted_specs = len(knots)//num_knots

    m0 = dmd.conversions.wavelength_to_mass(lam0)
    limit_decayrate = dmd.conversions.fluxscale_to_invsec(limit_rate)   
    limit_g = dmd.conversions.decayrate_to_axion_g(limit_decayrate, m0) 
    best_decayrate = dmd.conversions.fluxscale_to_invsec(best_rate)   
    best_g = dmd.conversions.decayrate_to_axion_g(best_decayrate, m0) 

    print(F"fit line in {num_fitted_specs} spectra")
    print(F"       lam0 = {lam0:0.2f} micron\n"
          F"       lmin = {lmin:0.2f} micron\n"
          F"       lmax = {lmax:0.2f} micron\n"
          F"  best_rate = {best_rate:0.2e} [fluxscale units]\n"
          F"delta_chisq = {delta_chisq:0.2f}\n"
          F" limit_rate = {limit_rate:0.2e} [fluxscale units]\n"
          F"         m0 = {m0:0.2f} eV\n"
          F"     best_g = {best_g:0.2e} GeV^{-1}\n"
          F"    limit_g = {limit_g:0.2e} GeV^{-1}")

    pc_input = [raw_out[9], raw_out[7], raw_out[8], 
                raw_out[2], raw_out[5][1], raw_out[1]]
    lam0_tmt, pc_limit = dmd.linesearch.find_pc_limit(configs, data, pc_input)
    pc_decayrate = dmd.conversions.fluxscale_to_invsec(pc_limit)   
    pc_g = dmd.conversions.decayrate_to_axion_g(pc_decayrate, m0) 
    print(F"       pc_g = {pc_g:0.2e} GeV^{-1}")



    plt.rcParams['text.usetex'] = True
    fig, [ax1, ax2] = plt.subplots(2, 1)

    # plot full spectral range
    red = np.argmax(np.asarray([np.min(spec["lam"]) for spec in data]))
    for index, spec in enumerate(data):
        if index == red:
            color = "firebrick"
            alpha = 0.8
        else:
            color = "mediumblue"
            alpha = 0.7
        ax1.step(spec["lam"], spec["sky"], color=color, alpha=alpha)

    # add line fit window
    for i in range(num_fitted_specs):    
        start = i*num_knots
        end = (i + 1)*num_knots
        # limiting model 
        limit_continuum_model = interp.CubicSpline(knots[start:end], 
                                                   limit_knots[start:end])
        limit_model = (limit_continuum_model(lam_list[i]) + 
                       dmd.linesearch.dm_line(lam_list[i], fixed_list[i], limit_rate))
        ax1.plot(lam_list[i], limit_model, 
                color='black', linestyle='-', marker='', 
                linewidth=2, alpha=1)
        ax1.axvline(lam0, linestyle='dotted', color='black', alpha=0.5)
        
    ax1.set_aspect(2)
    ax1.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                  fontsize=18)
    ax1.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                  fontsize=18)
    ax1.tick_params(axis='both', which='major', labelsize=16)

    # zoom in on fit line
    for i in range(num_fitted_specs):
        start = i*num_knots
        end = (i + 1)*num_knots
        ax2.fill_between(lam_list[i], 
                    sky_list[i] - error_list[i], 
                    sky_list[i] + error_list[i], step="mid",
                    color='black', alpha=0.3)
        ax2.step(lam_list[i], sky_list[i], where="mid",
                color='black', alpha=0.8)
        # limiting model 
        limit_continuum_model = interp.CubicSpline(knots[start:end], 
                                                   limit_knots[start:end])
        limit_model = (limit_continuum_model(lam_list[i]) + 
                       dmd.linesearch.dm_line(lam_list[i], fixed_list[i], limit_rate))
        ax2.plot(lam_list[i], limit_model, 
                color='firebrick', linestyle='-', marker='', 
                linewidth=2, alpha=1)
        ax2.axvline(lam0, linestyle='dotted', color='black', alpha=0.5)

    ax2.set_xlim(lam0 - 0.1, lam0 + 0.1)
    ax2.set_ylim(.08,.14)
    ax2.set_aspect(1)
    ax2.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                  fontsize=18)
    ax2.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                  fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    fig.tight_layout(pad=2)
    fig.set_size_inches(12, 8)
    output_path = "{}/combined-lam{}.pdf".format(configs["run"]["name"], lam0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")

