#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt

import DMdecayJWST as assume  
import conversions as convert


decay_only = {#"HST COB", 
          "MUSE":["MUSE", 3e0, 7e24, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 7e24, "0.4"]}

filenames = {"Globular Clusters":"GlobularClusters.txt", 
             # "HST COB":"HST.txt",
             "LeoT Heating":"LeoT.txt",
             "MUSE":"Telescopes_MUSE.txt", 
             "VIMOS":"Telescopes_VIMOS.txt",
             "CAST1":"CAST-CAPP.txt",
             "CAST2":"CAST_highm.txt",
             "CAST3":"CAST.txt"}

colors = {"Globular Clusters":"0.4", 
          # "HST COB":"0.75",
          "LeoT Heating":"0.6",
          "MUSE":"0.5", 
          "VIMOS":"0.65",
          "CAST1":"0.5",
          "CAST2":"0.5",
          "CAST3":"0.5"}

# name, x, y, color
labels = {"Globular Clusters":["Stellar Evolution", 4.6e-1, 5.5e-11, "0.4"],
          # "HST COB":["HST", 8e0, 3e-11, "0.5"],
          "MUSE":["MUSE", 3.1e0, 2e-11, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 2e-11, "0.4"],
          "CAST":["CAST", 5.5e-1, 4e-10, "0.8"]}

if __name__ == "__main__":
    run_name = sys.argv[1]    
    conservtive_limit_path = ("{}/conservative/JWST-NIRSPEC-limits.dat"
                              "".format(run_name))
    conservative_limit = np.loadtxt(conservtive_limit_path)
    line_limit_path = ("{}/continuum/JWST-NIRSPEC-limits.dat"
                              "".format(run_name))
    line_limit = np.loadtxt(line_limit_path)
    plt.rcParams['text.usetex'] = True

    limit_data = {}
    for name in filenames:
        path = os.path.join(assume.AxionLimits_dir, filenames[name])
        try:
            limit_data[name] = np.loadtxt(path)
        except:
            limit_data[name] = np.loadtxt(path, delimiter=",")

    lower_edge = 1e-13
    upper_edge = 1e-9
    left_edge = 4.5e-1
    right_edge = 6e0
    fig, [ax_t, ax_g] = plt.subplots(2, 1)

    for name in limit_data:
        if name == "Globular Clusters":
            muse_left = limit_data["MUSE"][1, 0]
            m_to_plot = np.array([left_edge, muse_left])
            limit_to_plot = np.ones(2)*limit_data[name][0, 1]
            ax_g.plot(m_to_plot, limit_to_plot, 
                    color=colors[name], marker='', 
                    linestyle='dotted', linewidth=1.5,
                    alpha=0.6)
        else:
            ax_g.fill_between(limit_data[name][:,0], limit_data[name][:,1], 
                            upper_edge, color=colors[name], linewidth=0)
    for name in labels:
        ax_g.text(labels[name][1], labels[name][2], 
                labels[name][0], color=labels[name][3], size=12)
    for name in decay_only:
        ax_t.text(decay_only[name][1], decay_only[name][2], 
                  decay_only[name][0], color=decay_only[name][3], size=12)

    ax_g.fill_between(line_limit[:,0], line_limit[:,2], 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.9, edgecolor=None)

    ax_g.vlines(conservative_limit[[0, -1], 0],
              conservative_limit[[0, -1], 2],
              [upper_edge, upper_edge],
              color="darkred", linewidth=0.75)

    ax_g.plot(conservative_limit[:,0], conservative_limit[:,2], 
            color="black", linewidth=2)

    # scale estimate 
    time = np.asarray([2, 15])*(3e7) #sec
    style= ['dashed', 'dotted']
    t0 = 2e3 #sec
    efficency = 0.01
    width = 150
    print("estimate for g:")
    for t, s in zip(time, style):
        scaled_limit = line_limit[:, 2]*(t*efficency/t0)**(-0.25)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_g.plot(line_limit[:, 0], smoothed_limit,
            color="blue", linewidth=1, linestyle=s,
            marker='', alpha=0.6) 
        print("    {:d} yr:  {:0.2e} - {:0.2e}"
              "".format(int(t/3e7), 
                        np.min(smoothed_limit), 
                        np.max(smoothed_limit)))
    print()

    ax_g.set_ylim([lower_edge, upper_edge])
    ax_g.set_xlim([left_edge, right_edge])
    ax_g.set_yscale('log')
    ax_g.set_xscale('log')

    ax_g.text(1e0, 13e-11, 
            "Total Flux", 
            color="black",
            fontsize=13, rotation=0)

    ax_g.text(5e-1, 1e-11, 
        "Continuum Model", 
        color="firebrick",
        fontsize=14, rotation=0)
    # ax_g.text(6.35e-1, 6e-12, 
    #     "Model", 
    #     color="firebrick",
    #     fontsize=13, rotation=0)


    ax_g.text(2.15e0, 4.7e-12, 
        "Current", 
        color="darkblue",
        fontsize=12, rotation=0)

    ax_g.text(8e-1, 1.5e-12, 
        "15 year", 
        color="darkblue",
        fontsize=13, rotation=0)


    ax_g.set_xlabel(r"$\displaystyle m_a\; [{\rm \tiny eV }]$", 
                  fontsize=16)
    ax_g.set_ylabel(r"$\displaystyle g_{a\gamma\gamma}\; [{\rm \tiny GeV }^{-1}]$", 
                  fontsize=16)


    lifetime_upper_edge = 4e29
    lifetime_lower_edge = 1e24

    for name in decay_only:
        rate_limit = convert.axion_g_to_decayrate(limit_data[name][:,1], 
                                                  limit_data[name][:,0])
        ax_t.fill_between(limit_data[name][:,0], lifetime_lower_edge,
                        rate_limit**-1, color=colors[name], 
                        linewidth=0, alpha=1)


    ax_t.fill_between(line_limit[:,0], line_limit[:,1]**-1, 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.85, edgecolor=None)

    ax_t.vlines(conservative_limit[[0, -1], 0],
              conservative_limit[[0, -1], 1]**-1,
              [upper_edge, upper_edge],
              color="firebrick", linewidth=0.75)

    ax_t.plot(conservative_limit[:,0], conservative_limit[:,1]**-1, 
            color="Black", linewidth=2)

    # scale estimate 
    print("estimate for tau:")
    for t, s in zip(time, style):
        scaled_limit = (line_limit[:, 1]**-1)*(t*efficency/t0)**(0.5)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_t.plot(line_limit[:, 0], smoothed_limit,
            color="blue", linewidth=1, linestyle=s,
            marker='', alpha=0.6) 
        print("    {:d} yr:  {:0.2e} - {:0.2e}"
              "".format(int(t/3e7), 
                        np.min(smoothed_limit), 
                        np.max(smoothed_limit)))

    ax_t.text(9e-1, 0.25e25, 
            "Total Flux", 
            color="black",
            fontsize=14, rotation=0)

    ax_t.text(5.4e-1, 0.25e27, 
        "Continuum", 
        color="firebrick",
        fontsize=14, rotation=0)   

    ax_t.text(6.25e-1, 1e26, 
        "Model", 
        color="firebrick",
        fontsize=14, rotation=0)

    ax_t.text(6.1e-1, 0.25e28, 
        "Current", 
        color="darkblue",
        fontsize=14, rotation=0)

    ax_t.text(2e0, 0.5e28, 
        "15 year", 
        color="darkblue",
        fontsize=14, rotation=0)

    ax_t.set_ylim([lifetime_lower_edge, lifetime_upper_edge])
    ax_t.set_xlim([left_edge, right_edge])
    ax_t.set_yscale('log')
    ax_t.set_xscale('log')
    # lifetime_constraint_path = "{}/lifetime_constraints.pdf".format(run_name)
    # fig.savefig(lifetime_constraint_path, dpi=300, bbox_inches="tight")

    # ax_t.set_xlabel(r"$\displaystyle m_{\rm DM}\; [{\rm \tiny eV }]$", 
    #               fontsize=16)
    ax_t.set_ylabel(r"$\displaystyle \tau \; [{\rm \tiny sec }]$", 
                  fontsize=16)

    ax_t.tick_params(axis='both', which='major', labelsize=13)
    ax_g.tick_params(axis='both', which='major', labelsize=13)
    ticks = [.5, .7, 1, 3, 6]
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels(ticks)
    ax_g.set_xticks(ticks)
    ax_g.set_xticklabels(ticks)
    constraint_path = "{}/constraints.pdf".format(run_name)
    fig.set_size_inches(8, 6)
    fig.tight_layout(pad=1)
    fig.savefig(constraint_path, dpi=300, bbox_inches="tight")