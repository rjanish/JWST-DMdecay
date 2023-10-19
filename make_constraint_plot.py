#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt

import DMdecayJWST as assume  
import conversions as convert


decay_only = ["HST COB", "MUSE", "VIMOS"]

filenames = {"Globular Clusters":"GlobularClusters.txt", 
             "HST COB":"HST.txt",
             "LeoT Heating":"LeoT.txt",
             "MUSE":"Telescopes_MUSE.txt", 
             "VIMOS":"Telescopes_VIMOS.txt",
             "CAST1":"CAST-CAPP.txt",
             "CAST2":"CAST_highm.txt",
             "CAST3":"CAST.txt"}

colors = {"Globular Clusters":"0.4", 
          "HST COB":"0.75",
          "LeoT Heating":"0.6",
          "MUSE":"0.5", 
          "VIMOS":"0.65",
          "CAST1":"0.5",
          "CAST2":"0.5",
          "CAST3":"0.5"}

# name, x, y, color
labels = {"Globular Clusters":["Stellar Evolution", 1.1e-1, 5.5e-11, "0.4"],
          "HST COB":["HST", 8e0, 3e-11, "0.5"],
          "MUSE":["MUSE", 2.8e0, 4e-12, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 1e-11, "0.4"],
          "CAST":["CAST", 1.1e-1, 3e-10, "0.8"]}

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
    left_edge = 5e-1
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
                labels[name][0], color=labels[name][3], size=10)

    ax_g.fill_between(line_limit[:,0], line_limit[:,2], 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.85, edgecolor=None)

    ax_g.vlines(conservative_limit[[0, -1], 0],
              conservative_limit[[0, -1], 2],
              [upper_edge, upper_edge],
              color="darkred", linewidth=0.75)

    ax_g.plot(conservative_limit[:,0], conservative_limit[:,2], 
            color="darkgreen", linewidth=1)

    # scale estimate 
    time = np.asarray([2, 15])*(3e7) #sec
    style= ['dashed', 'dotted']
    t0 = 2e3 #sec
    efficency = 0.1
    width = 150
    for t, s in zip(time, style):
        scaled_limit = line_limit[:, 2]*(t*efficency/t0)**(-0.25)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_g.plot(line_limit[:, 0], smoothed_limit,
            color="blue", linewidth=1, linestyle=s,
            marker='', alpha=0.6) 

    ax_g.set_ylim([lower_edge, upper_edge])
    ax_g.set_xlim([left_edge, right_edge])
    ax_g.set_yscale('log')
    ax_g.set_xscale('log')

    ax_g.text(4e-1, 2.8e-11, 
            "Total Flux", 
            color="darkgreen",
            fontsize=10, rotation=0)

    ax_g.text(2.75e-1, 6e-12, 
        "Continuum Model", 
        color="firebrick",
        fontsize=10, rotation=0)

    ax_g.text(5.5e-1, 1.7e-12, 
        "2 year", 
        color="darkblue",
        fontsize=10, rotation=0)

    ax_g.text(5.1e-1, 1e-12, 
        "15 year", 
        color="darkblue",
        fontsize=10, rotation=0)


    ax_g.set_xlabel(r"$\displaystyle m_a\; [{\rm \tiny eV }]$", 
                  fontsize=14)
    ax_g.set_ylabel(r"$\displaystyle g_{a\gamma\gamma}\; [{\rm \tiny GeV }^{-1}]$", 
                  fontsize=14)


    lifetime_upper_edge = 1e30
    lifetime_lower_edge = 1e23

    for name in decay_only:
        rate_limit = convert.axion_g_to_decayrate(limit_data[name][:,1], 
                                                  limit_data[name][:,0])
        ax_t.fill_between(limit_data[name][:,0], lifetime_lower_edge,
                        rate_limit**-1, color=colors[name], linewidth=0)


    ax_t.fill_between(line_limit[:,0], line_limit[:,1]**-1, 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.85, edgecolor=None)

    ax_t.vlines(conservative_limit[[0, -1], 0],
              conservative_limit[[0, -1], 1]**-1,
              [upper_edge, upper_edge],
              color="darkred", linewidth=0.75)

    ax_t.plot(conservative_limit[:,0], conservative_limit[:,1]**-1, 
            color="darkgreen", linewidth=1)

    # scale estimate 
    for t, s in zip(time, style):
        scaled_limit = (line_limit[:, 1]**-1)*(t*efficency/t0)**(0.5)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_t.plot(line_limit[:, 0], smoothed_limit,
            color="blue", linewidth=1, linestyle=s,
            marker='', alpha=0.6) 

    ax_t.set_ylim([lifetime_lower_edge, lifetime_upper_edge])
    ax_t.set_xlim([left_edge, right_edge])
    ax_t.set_yscale('log')
    ax_t.set_xscale('log')
    # lifetime_constraint_path = "{}/lifetime_constraints.pdf".format(run_name)
    # fig.savefig(lifetime_constraint_path, dpi=300, bbox_inches="tight")

    ax_t.set_xlabel(r"$\displaystyle m_{\rm DM}\; [{\rm \tiny eV }]$", 
                  fontsize=14)
    ax_t.set_ylabel(r"$\displaystyle \tau \; [{\rm \tiny sec }]$", 
                  fontsize=14)

    constraint_path = "{}/constraints.pdf".format(run_name)
    fig.set_size_inches(8, 12)
    fig.savefig(constraint_path, dpi=300, bbox_inches="tight")