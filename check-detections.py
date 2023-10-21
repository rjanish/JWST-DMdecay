#!/usr/bin/env python3
# coding: utf-8

import numpy as np 
from scipy.special import erfc, erfcinv

p = lambda chisq: 0.5*erfc(np.sqrt(0.5*chisq))

bestfits_path = "gnz11_final/continuum/JWST-NIRSPEC-bestfits.dat"

bestfits = np.loadtxt(bestfits_path)
chisqs = bestfits[:, 4]
num_trials = chisqs.size 
# find 5-sigma local detections
Ns = 5
detections = chisqs > Ns**2
Ndetect = np.sum(detections)
print("num detections: {}".format(Ndetect))
for detection in bestfits[detections, :]:
	lam = detection[0]
	chisq = detection[4]
	best_g = detection[3]
	print("\nlam = {:0.4f}, d(chisq) = {:0.1f}"
		  "".format(lam, chisq))
	# global significance
	p_local = p(chisq)
	p_global = p_local*num_trials
	Z_global = np.sqrt(2)*erfcinv(2*p_global)
	print("    sigma_local  = {:0.1f}\n" 
		  "    sigma_global = {:0.1f}\n"
		  "    best_rate    = {:0.2e}"
		  "".format(np.sqrt(chisq), Z_global, best_g))



# # limits diagonostics 
# limits_path = "gnz11_final/continuum/JWST-NIRSPEC-limits.dat"
# limits = np.loadtxt(limits_path)
# strongest_m = bestfits[np.argsort(limits[:, 2]), 0]
# strongest_g = limits[np.argsort(limits[:, 2]), 2]
# for l, g in zip(strongest_m[:10], strongest_g[:10]):
# 	print(l, g)