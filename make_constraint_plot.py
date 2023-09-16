#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import matplotlib.pyplot as plt

import DMdecayJWST as assume  


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

# name, x, y, color
labels = {"Globular Clusters":["Globular Clusters", 1.1e-1, 7e-11, "0.4"],
          "HST COB":["HST", 8e0, 3e-11, "0.5"],
          "MUSE":["MUSE", 2.9e0, 1e-10, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 1e-11, "0.4"],
          "CAST":["CAST", 1.1e-1, 3e-10, "0.8"]}

if __name__ == "__main__":
    new_limit_path = sys.argv[1]    
    new_limit_name = os.path.splitext(new_limit_path)[0]
    new_limit = np.loadtxt(new_limit_path)
    plt.rcParams['text.usetex'] = True

    limit_data = {}
    for name in filenames:
        path = os.path.join(assume.AxionLimits_dir, filenames[name])
        try:
            limit_data[name] = np.loadtxt(path)
        except:
            limit_data[name] = np.loadtxt(path, delimiter=",")

    lower_edge = 1e-14
    upper_edge = 1e-8
    left_edge = 1e-1
    right_edge = 1.1e1
    fig, ax = plt.subplots()
    for name in limit_data:
        ax.fill_between(limit_data[name][:,0], limit_data[name][:,1], 
                        upper_edge, color=colors[name], linewidth=0)
    for name in labels:
        ax.text(labels[name][1], labels[name][2], 
                labels[name][0], color=labels[name][3], size=10)

    ax.fill_between(new_limit[:,0], new_limit[:,2], 
                    upper_edge, facecolor="indianred", linewidth=1,
                    alpha=0.8, edgecolor="darkred")

    ax.set_ylim([lower_edge, upper_edge])
    ax.set_xlim([left_edge, right_edge])
    ax.set_yscale('log')
    ax.set_xscale('log')


    ax.set_xlabel(r"$\displaystyle m_a\; [{\rm \tiny eV }]$", 
                  fontsize=14)
    ax.text(3.75e-2, 2e-11, 
            r"$\displaystyle g_{a\gamma\gamma}$", 
            fontsize=20, rotation=0)
    ax.text(3.5e-2, 6e-12, 
            r"$\displaystyle [{\rm \tiny GeV }^{-1}]$", 
            fontsize=13, rotation=0)

    constraint_path = "{}-CONSTRAINTS.pdf".format(new_limit_name)
    fig.savefig(constraint_path, dpi=300, bbox_inches="tight")



