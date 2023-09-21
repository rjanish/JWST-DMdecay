#!/usr/bin/env python3
""" Download data from JWST archives """

import sys
import os 

import numpy as np
import matplotlib.pyplot as plt
import astropy.table as table

import DMdecayJWST as assume  


if __name__ == "__main__":
    run_name = sys.argv[1]    
    conservtive_itemized_path = (
        "{}/conservative/JWST-NIRSPEC-chisq-itemized.dat"
        "".format(run_name))
    itemized = np.loadtxt(conservtive_itemized_path)
    
    fraction_per_obv = np.sum(itemized, axis=1)/np.sum(itemized)
    contributing = fraction_per_obv > 1e-12
    sorted_obvs = np.argsort(fraction_per_obv[contributing])[::-1]
    targets = table.Table.read("{}/targets.dat".format(run_name), 
                               format="ascii.csv")
    leading = targets["name", "filter", "grating"][contributing][sorted_obvs]
    leading.add_column(fraction_per_obv[contributing][sorted_obvs],
                       name="chisq fraction")
    leading.write("{}/leading-targets.dat".format(run_name), 
                  format="ascii.csv", overwrite=True)