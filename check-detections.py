#!/usr/bin/env python3
# coding: utf-8

import numpy as np 

bestfits_path = "gnz11_final/continuum/JWST-NIRSPEC-bestfits.dat"

bestfits = np.loadtxt(bestfits_path)
chisqs = bestfits[:, 4]
detection = chisqs > (6.18)**2
print("num detections: {}".format(np.sum(detection)))

bestfits_detection = bestfits[detection, :]

line_lams = bestfits_detection[:, 0]
print("detection wavelengths:")
print(np.unique(np.round(line_lams, 3)))
print("rough detection levels")
print(np.unique(np.round(bestfits_detection[:, 4], 3)))
