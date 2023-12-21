"""
Print summary stastics from line search 
"""


import sys 

import numpy as np 
from scipy.special import erfc, erfcinv

import DMdecay as dmd


p_value = lambda chisq: 0.5*erfc(np.sqrt(0.5*chisq))

if __name__ == "__main__":
    config_filename = sys.argv[1]    
    configs = dmd.prep.parse_configs(config_filename)    
    run_name = configs["run"]["name"]
    bestfits_path = F"{run_name}/{configs['run']['bestfits_filename']}"

    bestfits = np.loadtxt(bestfits_path)
    bestfits = bestfits[np.isfinite(bestfits[:, 2]), :] # omit nans 
    chisqs = bestfits[:, 4]
    num_trials = chisqs.size 
    # find 5-sigma local detentions
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
        p_local = p_value(chisq)
        p_global = p_local*num_trials
        Z_global = np.sqrt(2)*erfcinv(2*p_global)
        print("    sigma_local  = {:0.1f}\n" 
              "    sigma_global = {:0.1f}\n"
              "    best_rate    = {:0.2e}"
              "".format(np.sqrt(chisq), Z_global, best_g))

    # limits diagnostics 
    limits_path = F"{run_name}/{configs['run']['pc_filename']}"
    limits = np.loadtxt(limits_path)
    rawlimits_path = F"{run_name}/{configs['run']['rawlimits_filename']}"
    rawlimits = np.loadtxt(rawlimits_path)
    # replace nan's from power constraint with raw result
    infs = ~np.isfinite(limits[:, 2])
    limits[infs, 1:3] = rawlimits[infs, 1:3]
    limits[infs, 3] = 0
    # if raw result is also nan, just omit those datapoints 
    limits = limits[np.isfinite(limits[:, 2]), :]

    print("\nlimits summary:\n")
    largest_lifetime = np.argmin(limits[:, 1])  # limits is decay rate
    print("largest constrainted lifetime:\n"
          "      m = {:0.3f}\n"
          "    tau = {:0.2e}\n"
          "".format(limits[largest_lifetime, 0], 1.0/limits[largest_lifetime, 1]))

    smallest_g = np.argmin(limits[:, 2])
    print("smallest constrainted g:\n"
          "      m = {:0.3f}\n"
          "      g = {:0.2e}\n"
          "".format(limits[smallest_g, 0], limits[smallest_g, 2]))

    closest_to_1eV = np.argmin(np.absolute(limits[:, 0] - 1.0))
    m_1eV = limits[closest_to_1eV, 0]
    lambda_1eV = dmd.conversions.mass_to_wavelength(m_1eV)
    print("at m = {:0.4f}\n"
          "  lambda0 = {:0.4f}\n"
          "      tau = {:0.2e}\n"
          "        g = {:0.2e}\n"
          "".format(m_1eV, lambda_1eV,
                    1.0/limits[closest_to_1eV, 1],
                    limits[closest_to_1eV, 2]))

    print("NIRSpec mass coverage (eV)")
    print(dmd.conversions.wavelength_to_mass(np.asarray([0.6, 5])))