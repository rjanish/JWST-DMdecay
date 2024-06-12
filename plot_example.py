""" 
Plot full spectrum with close-up of line model 
"""

import sys
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp

import DMdecay as dmd
import JWSTutils as jwst
from gnz11_split import partition_gnz11

if __name__ == "__main__":
    config_filename = sys.argv[1]    
    lam0 = float(sys.argv[2])
    to_plot = sys.argv[3]
    scale = int(sys.argv[4]) # plot model with strength rate*scale
    try:
        plot_yrange = map(float, sys.argv[5:7])
    except:
        plot_yrange = None
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

    plt.rcParams['text.usetex'] = True
    fig = plt.figure()
    # fig, [ax_fullspec, ax_zoom, ax_resid] = plt.subplots(3, 1)
    # ax_resid.sharex(ax_zoom)

    gs_fullspec = fig.add_gridspec(7, 1, hspace=2, top=0.95)
    gs_zoom = fig.add_gridspec(7, 1, hspace=0, top=0.8)
    ax_fullspec = fig.add_subplot(gs_fullspec[0:3, :])
    ax_zoom = fig.add_subplot(gs_zoom[3:6, :])
    ax_resid = fig.add_subplot(gs_zoom[6, :])

    dofs = [5, 15]
    mcolors = ["firebrick", "darkgreen"]
    for j, (num_knots, m_color) in enumerate(zip(dofs, mcolors)):
        configs["analysis"]["num_knots"] = num_knots
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
        num_fitted_specs = len(knots)//num_knots

        m0 = dmd.conversions.wavelength_to_mass(lam0)
        limit_decayrate = dmd.conversions.fluxscale_to_invsec(limit_rate)   
        limit_g = dmd.conversions.decayrate_to_axion_g(limit_decayrate, m0) 
        best_decayrate = dmd.conversions.fluxscale_to_invsec(best_rate)   
        best_g = dmd.conversions.decayrate_to_axion_g(best_decayrate, m0) 

        print(F"fit line in {num_fitted_specs} spectra")
        print(F"       lam0 = {lam0:0.4f} micron\n"
            F"       lmin = {lmin:0.4f} micron\n"
            F"       lmax = {lmax:0.4f} micron\n"
            F"  best_rate = {best_rate:0.2e} [fluxscale units]\n"
            F"delta_chisq = {delta_chisq:0.2f}\n"
            F" limit_rate = {limit_rate:0.2e} [fluxscale units]\n"
            F"         m0 = {m0:0.4f} eV\n"
            F"     best_g = {best_g:0.2e} GeV^{-1}\n"
            F"    limit_g = {limit_g:0.2e} GeV^{-1}")

        pc_input = [raw_out[9], raw_out[7], raw_out[8], 
                    raw_out[2], raw_out[5][1], raw_out[1]]
        if to_plot == "pc":
            lam0_tmt, pc_limit, all_uppers = dmd.linesearch.find_pc_limit(configs, 
                                                                        data, 
                                                                        pc_input)
            pc_decayrate = dmd.conversions.fluxscale_to_invsec(pc_limit)   
            pc_g = dmd.conversions.decayrate_to_axion_g(pc_decayrate, m0) 
            print(F"       pc_g = {pc_g:0.2e} GeV^{-1}")

        if j == 0:
            # plot full spectral range
            red = np.argmax(np.asarray([np.min(spec["lam"]) for spec in data]))
            for index, spec in enumerate(data):
                if index == red:
                    color = "firebrick"
                    alpha = 0.8
                else:
                    color = "mediumblue"
                    alpha = 0.7
                ax_fullspec.step(spec["lam"], spec["sky"], color=color, alpha=alpha)

            # add line fit window
            for i in range(num_fitted_specs):    
                start = i*num_knots
                end = (i + 1)*num_knots
                # limiting model 
                limit_continuum_model = interp.CubicSpline(knots[start:end], 
                                                        limit_knots[start:end])
                limit_model = (limit_continuum_model(lam_list[i]) + 
                            dmd.linesearch.dm_line(lam_list[i], fixed_list[i], limit_rate))
                ax_fullspec.plot(lam_list[i], limit_model, 
                        color='black', linestyle='-', marker='', 
                        linewidth=2, alpha=1)
                ax_fullspec.axvline(lam0, linestyle='dotted', color='black', alpha=0.5)
                
            # ax_fullspec.set_aspect(2)
            ax_fullspec.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                        fontsize=18)
            ax_fullspec.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                        fontsize=18)
            ax_fullspec.tick_params(axis='both', which='major', labelsize=18)

        # zoom in on fit line
        for i in range(num_fitted_specs):
            start = i*num_knots
            end = (i + 1)*num_knots
            if j == 0:
                ax_zoom.fill_between(lam_list[i], 
                            sky_list[i] - error_list[i], 
                            sky_list[i] + error_list[i], step="mid",
                            color='black', alpha=0.3)
                ax_zoom.step(lam_list[i], sky_list[i], where="mid",
                        color='black', alpha=0.8)
                ax_zoom.axvline(lam0, linestyle='dotted', color='black', alpha=0.5)
                ax_resid.fill_between(lam_list[i], -error_list[i], error_list[i],
                                    step="mid", color='black', alpha=0.3) 
                ax_resid.axhline(0, linestyle='solid', linewidth=0.5, 
                                color='black', alpha=0.6)
                ax_resid.axvline(lam0, linestyle='dotted', color='black', alpha=0.5)
            # limiting model 
            limit_continuum_model = interp.CubicSpline(knots[start:end], 
                                                    limit_knots[start:end])
            # if to_plot == "limit":
            #     rate = limit_rate
            #     print(F"plotting zoom-in limiting line with rate {rate:3f}")
            # elif to_plot == "bestfit":
            #     rate = best_rate
            #     print(F"plotting zoom-in bestfit line with rate {rate:3f}")
            model = (limit_continuum_model(lam_list[i]) + 
                    dmd.linesearch.dm_line(lam_list[i], fixed_list[i], limit_rate))
            ax_zoom.plot(lam_list[i], model, 
                    color=m_color, linestyle='-', marker='', 
                    linewidth=2, alpha=1)

            # residuals 
            ax_resid.step(lam_list[i], sky_list[i] - model, 
                    color=m_color, linestyle='-', marker='', 
                    linewidth=2, alpha=1)

    margin = 0.001
    ax_zoom.set_xlim(lam_list[0][0]*(1-margin), lam_list[0][-1]*(1+margin))
    ax_resid.set_xlim(lam_list[0][0]*(1-margin), lam_list[0][-1]*(1+margin))
    if plot_yrange is not None:
        ax_zoom.set_ylim(*plot_yrange)
    # ax_zoom.set_aspect(1)
    ax_resid.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                fontsize=18)
    ax_resid.set_ylabel(r"$\displaystyle {\rm residual}$", fontsize=18)
    ax_zoom.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                fontsize=18)
    ax_zoom.set_yticks([0.09, 0.11, 0.13])
    ax_zoom.tick_params(axis='both', which='major', labelsize=16)
    ax_zoom.set_ylim(0.08, 0.14)
    ax_resid.set_yticks([-0.01, 0, 0.01])
    ax_resid.tick_params(axis='both', which='major', labelsize=16)
    ax_resid.set_ylim(-0.018, 0.018)
    fig.tight_layout(pad=1)
    fig.set_size_inches(12, 8)
    output_path = "{}/combined-lam{}.pdf".format(configs["run"]["name"], lam0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

