#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['text.usetex'] = True

import DMdecayJWST as setup  

if __name__ == "__main__":
    data, targets = setup.parse_gnz11()
    fig, ax = plt.subplots()
    # find spectrum with largest starting wavelength
    red = np.argmax(np.asarray([np.min(spec["lam"]) for spec in data]))
    for index, spec in enumerate(data):
        if index == red:
            color = "firebrick"
            alpha = 1.0
        else:
            color = "mediumblue"
            alpha = 0.8
        ax.step(spec["lam"], spec["sky"], color=color, alpha=alpha)
    
    ax.set_xlabel(r"$\displaystyle \lambda \; [\mu{\rm \tiny m}]$", 
                  fontsize=14)
    ax.set_ylabel(r"$\displaystyle \Phi \; [{\rm MJy/sr}]$", 
                  fontsize=14)
    fig.savefig("full-spectra.pdf", dpi=300, bbox_inches="tight")

