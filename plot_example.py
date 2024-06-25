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
    to_plot = sys.argv[2]
    lams = np.asarray(sys.argv[3:], dtype=float)
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

    configs["analysis"]["width_factor"] = 75
    dofs = [5]
    styles = ["solid", "dotted"]
    colors = ["firebrick", "mediumblue"]
    for lam0, color in zip(lams, colors):
        print(F"fitting {lam0} micron in {color}")
        for j, (num_knots, style) in enumerate(zip(dofs, styles)):
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

            print("-"*33 + "\n" + F"fit line in {num_fitted_specs} spectra")
            print(F"     lam0 = {lam0:0.4f} micron\n"
                F"         m0 = {m0:0.4f} eV\n"
                F"       lmin = {lmin:0.4f} micron\n"
                F"       lmax = {lmax:0.4f} micron\n"
                F"  num_knots = {num_knots}\n"
                F"  best_rate = {best_rate:0.2e} [fluxscale units]\n"
                F"delta_chisq = {delta_chisq:0.2f}\n"
                F" limit_rate = {limit_rate:0.2e} [fluxscale units]\n"
                F"     best_g = {best_g:0.2e} GeV^{-1}\n"
                F"    limit_g = {limit_g:0.2e} GeV^{-1}\n"
                F"noise_rescale = {error_scale_factors}")
            dof = sum([np.sum(~mask) for mask in mask_list])
            print(F"         dof = {dof}")

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
                        c1 = "firebrick"
                        alpha = 0.8
                    else:
                        c1 = "mediumblue"
                        alpha = 0.7
                    ax_fullspec.step(spec["lam"], spec["sky"], color=c1, alpha=alpha)

                # add line fit window
                for i in range(num_fitted_specs):    
                    start = i*num_knots
                    end = (i + 1)*num_knots
                    # limiting model 
                    limit_continuum_model = \
                        interp.CubicSpline(knots[start:end],
                                           limit_knots[start:end])(lam_list[i])  
                    limit_line_model = \
                        dmd.linesearch.dm_line(lam_list[i], 
                                               fixed_list[i], limit_rate)
                    ax_fullspec.plot(lam_list[i], limit_continuum_model,
                                     color='black', linestyle='-', marker='',
                                     linewidth=2, alpha=1)
                    ax_fullspec.axvline(lam0, linestyle='dotted', 
                                        color='black', alpha=0.5)
                    
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
                print(F"plotting {color}")
                limit_continuum_model = interp.CubicSpline(
                    knots[start:end], limit_knots[start:end])
                linewidth = \
                    lam_list[i][np.abs(lam_list[i] - lam0) < lam0*0.002]
                limit_line_model = \
                        dmd.linesearch.dm_line(linewidth, 
                                               fixed_list[i], limit_rate)
                ax_zoom.plot(lam_list[i], limit_continuum_model(lam_list[i]), 
                        color=color, linestyle=style, marker='', 
                        linewidth=2, alpha=0.8)
                total_model = (limit_continuum_model(linewidth) + 
                               1.5*(5/2)*limit_line_model)
                ax_zoom.plot(linewidth, total_model, 
                             color="darkgreen", linestyle=style, marker='', 
                             linewidth=2, alpha=0.8)

                # residuals 
                ax_resid.step(lam_list[i], 
                              sky_list[i] - limit_continuum_model(lam_list[i]), 
                        color=color, linestyle=style, marker='', 
                        linewidth=2, alpha=0.8)
                ax_resid.step(linewidth, limit_line_model, 
                        color="darkgreen", linestyle=style, marker='', 
                        linewidth=2, alpha=0.8)

    margin = 0.001
    ax_zoom.set_xlim(lam_list[0][0]*(1-margin), lam_list[0][-1]*(1+margin))
    ax_resid.set_xlim(lam_list[0][0]*(1-margin), lam_list[0][-1]*(1+margin))
    lower = np.min(sky_list[i] - np.median(error_list[i]))
    upper = np.max(sky_list[i] + np.median(error_list[i]))
    height = upper - lower
    gap = 0.5*height
    ax_zoom.set_ylim(lower - gap, upper + gap)
    # ax_zoom.set_aspect(1)
    ax_resid.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                fontsize=18)
    ax_resid.set_ylabel(r"$\displaystyle {\rm residual}$", fontsize=18)
    ax_zoom.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                fontsize=18)
    # ax_zoom.set_yticks([0.09, 0.11, 0.13])
    ax_zoom.tick_params(axis='both', which='major', labelsize=16)
    # ax_zoom.set_ylim(0.08, 0.14)
    ax_resid.set_yticks([-0.01, 0, 0.01])
    ax_resid.tick_params(axis='both', which='major', labelsize=16)
    ax_resid.set_ylim(-0.018, 0.018)
    fig.tight_layout(pad=1)
    fig.set_size_inches(12, 8)
    output_path = "{}/combined-lam{}.pdf".format(configs["run"]["name"], lam0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.show()

