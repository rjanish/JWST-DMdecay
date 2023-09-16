#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import matplotlib.pyplot as plt

AxionLimits_dir = "/home/rjanish/physics/AxionLimits/limit_data/AxionPhoton"

filenames = {"Globular Clusters":"GlobularClusters.txt", 
             "HST COB":"HST.txt",
             "LeoT Heating":"LeoT.txt",
             "MUSE":"Telescopes_MUSE.txt", 
             "VIMOS":"Telescopes_VIMOS.txt",
             "CAST1":"CAST-CAPP.txt",
             "CAST2":"CAST_highm.txt",
             "CAST3":"CAST.txt"}

colors = {"Globular Clusters":"0.8", 
          "HST COB":"0.75",
          "LeoT Heating":"0.6",
          "MUSE":"0.5", 
          "VIMOS":"0.65",
          "CAST1":"0.6",
          "CAST2":"0.6",
          "CAST3":"0.6"}

if __name__ == "__main__":
    new_limit_path = sys.argv[1]    
    new_limit = np.loadtxt(new_limit_path)
    # plt.rcParams['text.usetex'] = True

    limit_data = {}
    for name in filenames:
        path = os.path.join(AxionLimits_dir, filenames[name])
        try:
            limit_data[name] = np.loadtxt(path)
        except:
            limit_data[name] = np.loadtxt(path, delimiter=",")

    lower_edge = 1e-14
    upper_edge = 1e-8
    left_edge = 5e-2
    right_edge = 50
    fig, ax = plt.subplots()
    for name in limit_data:
        ax.fill_between(limit_data[name][:,0], limit_data[name][:,1], 
                        upper_edge, color=colors[name], linewidth=0)

    ax.fill_between(new_limit[:,0], new_limit[:,2], 
                    upper_edge, facecolor="indianred", linewidth=1,
                    alpha=0.8, edgecolor="darkred")

    ax.set_ylim([lower_edge, upper_edge])
    ax.set_xlim([left_edge, right_edge])
    ax.set_yscale('log')
    ax.set_xscale('log')


    # ax.set_xlabel(r"$\displaystyle m_a\; [{\rm \tiny eV }]$", 
    #               fontsize=14)
    # ax.text(1.5e-7, 8e-12, r"$\displaystyle g_{a\gamma\gamma}$", 
    #         fontsize=20, rotation=0)

    plt.show()

