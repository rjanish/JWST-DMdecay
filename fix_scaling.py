#!/usr/bin/env python3
# coding: utf-8

import numpy as np 
import conversions as convert 

old_path = "gnz11_final/continuum/wrong_scaling/JWST-NIRSPEC-pc.dat"
output = np.loadtxt(old_path)
	# columns: (test_lams, m, limits, full_pc_limits, pc_hit)

m = output[:, 1]
limits = output[:, 2]
full_pc_limits = output[:, 3]

final_limits = np.max([limits, full_pc_limits], axis=0)

# physical conversion 
limit_decayrate = convert.fluxscale_to_invsec(final_limits)    
limit_g = convert.decayrate_to_axion_g(limit_decayrate, m) 


# write output
new_path = "gnz11_final/continuum/JWST-NIRSPEC-limits.dat"
limits_header = ("DM decay limits vs mass \n"
          "JWST NIRSPEC run gnz11_final (RESCALED)\n"
          "mass [ev]    lifetime [sec]    "
          "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)")
np.savetxt(new_path, 
           np.column_stack((m, limit_decayrate, limit_g)),
           header=limits_header)