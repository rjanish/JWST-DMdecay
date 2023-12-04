""" DM limits from total flux  """


import time 

import numpy as np
import scipy.optimize as opt

from . import halo
from . import conversions as convert 


def chisq_dm_totalflux(rate, fixed_list, data, shift=0):
    total = 0.0
    for i, spec in enumerate(data):
        model = halo.MWDecayFlux(spec["lam"], fixed_list[i][0], rate, 
                                 fixed_list[i][1], fixed_list[i][2])
        resid = np.zeros(model.shape)
        dm_large = model > spec["sky"]
        resid[dm_large] = ((model[dm_large] - spec["sky"][dm_large]) / 
                           (spec["error"][dm_large])) 
            # use unscaled error - this doesn't matter much here, 
            # the result depends on the error at less-than-O(1)
            # level so long as error < flux
        total += np.sum(resid**2)
    return total - shift


def run(data, configs, test_lams):
    limit = np.ones(test_lams.shape)*np.nan
    print("scanning {} mass trials for total flux limits...".format(len(test_lams)))
    t0 = time.time()
    uppers = [1e-8, 1e-6, 1e-4, 1e-2, 1, 100]
    for index, lam0 in enumerate(test_lams):
        fixed_list = [[lam0, spec["D"], 
                       halo.sigma_from_fwhm(spec["res"], lam0, 
                                            configs["halo"]["sigma_v"])]
                      for spec in data]
        lower = 0.0
        for upper in uppers:
            chisq_test = chisq_dm_totalflux(upper, fixed_list, data,
                                            configs["analysis"]["chisq_step"])
            if chisq_test > 0:
                sol = opt.root_scalar(chisq_dm_totalflux, 
                                      args=(fixed_list, data, 
                                            configs["analysis"]["chisq_step"]), 
                                      bracket=[lower, upper])
                limit[index] = sol.root
                # chisq_itemized[:, index] = chisq_nb(sol.root, lam0, 
                #                                     0, "itemized")
                break
            lower = upper
    print("elapsed: {:0.2f} sec".format(time.time() - t0))
    # convert to physics units
    m = convert.wavelength_to_mass(test_lams)
    limit_decayrate = convert.fluxscale_to_invsec(limit)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m)    
    # output 
    limits_path = ("{}/flux-limits.dat"
                   "".format(configs["run"]["name"]))
    limits_header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(configs["run"]["name"]))
    np.savetxt(limits_path, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=limits_header)
