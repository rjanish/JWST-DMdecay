"""
Search for DM line on smooth background
"""

import time  
import copy
import multiprocessing as mltproc  
import functools
import json  

import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as opt
import scipy.interpolate as interp
import scipy.integrate as integ 

from . import conversions as convert 
from . import halo
from nestedtolist import nestedtolist


def spline_residual(knot_values, knots, x, y, sigma_y):
    num_specs = len(x)
    num_knots = int(knots.size/num_specs)
    resiudal_list = []
    for i in range(num_specs):
        start = i*num_knots
        end = (i + 1)*num_knots
        spline = interp.CubicSpline(knots[start:end], 
                                    knot_values[start:end])
        resiudal_list.append((y[i] - spline(x[i]))/sigma_y[i])
    return np.concatenate(resiudal_list)

def spline_plus_line_residual(params, knots, fixed, x, y, sigma_y):
    num_specs = len(x)
    num_knots = int(knots.size/num_specs)
    knot_values = params[:-1]
    rate = params[-1]
    resiudal_list = []
    for i in range(num_specs):
        start = i*num_knots
        end = (i + 1)*num_knots
        sub_knots = knots[start:end]
        sub_knot_values = knot_values[start:end]
        spline = interp.CubicSpline(sub_knots, sub_knot_values)
        model = spline(x[i]) + dm_line(x[i], fixed[i], rate)
        resiudal_list.append((y[i] - model)/sigma_y[i])
    return np.concatenate(resiudal_list)

def line_chisq(decay_rate, knot_values, knots, fixed, 
               x, y, sigma_y, shift=0.0):
    resids = spline_plus_line_residual(
        np.concatenate((knot_values, [decay_rate])),
        knots, fixed, x, y, sigma_y)
    total_chisq = np.sum(resids**2) 
    return total_chisq - shift

def dm_line(lam, fixed, rate):
    return halo.MWDecayFlux(lam, fixed[0], rate, fixed[1], fixed[2])

def rate_func(knot_values, best_rate, knots, fixed_list, 
              lam_list, sky_list, error_list, 
              threshold, upper_bracket): 
    current_withbest = line_chisq(best_rate, knot_values, knots, 
                                  fixed_list, 
                                  lam_list, sky_list, 
                                  error_list, threshold)
    if current_withbest < 0:
        sol = opt.root_scalar(line_chisq, 
                              bracket=[best_rate, upper_bracket],
                              args=(knot_values, knots, fixed_list, 
                                    lam_list, sky_list, 
                                    error_list, threshold))
        return -sol.root
    else:
        return -best_rate

def find_raw_limit(configs, data, lam0):
    padding = configs["analysis"]["padding"]
    num_knots = configs["analysis"]["num_knots"]
    max_clip_iters = configs["analysis"]["max_clip_iters"]
    clipping_factor = configs["analysis"]["clipping_factor"]
    chisq_step = configs["analysis"]["chisq_step"]
    limit_guess = configs["analysis"]["limit_guess"] 
    window = configs["analysis"]["width_factor"]*configs["halo"]["sigma_v"]
    # extract line search region 
    sky_list = []
    lam_list = []
    raw_error_list = []
    fixed_list = []
    spec_list = []
    res_list = []
    lstart = [spec["lam"][0] for spec in data]
    lend = [spec["lam"][-1] for spec in data]
    lmin = lam0*(1.0 - 0.5*window)
    lmax = lam0*(1.0 + 0.5*window)
    for i, spec in enumerate(data):
        if (lam0 < lstart[i]) or (lam0 > lend[i]):
            continue
        if lmin < lstart[i]:
            l_left = lstart[i]
            l_right = l_left*(1 + window)
        elif lend[i] < lmax:
            l_right = lend[i]
            l_left = l_right*(1 - window)
        else:
            l_left = lmin
            l_right = lmax
        select = (l_left < spec["lam"]) & (spec["lam"] < l_right)
        spec_list.append(i)
        sky_list.append(spec["sky"][select])
        lam_list.append(spec["lam"][select])
        raw_error_list.append(spec["error"][select])
        fixed_list.append([lam0, spec["D"], 
                           halo.sigma_from_fwhm(spec["res"], lam0, 
                                                configs["halo"]["sigma_v"])])    
        res_list.append(spec["res"])
    num_specs = len(sky_list)
    if num_specs == 0:
        return [[lmin, lmax], spec_list, [None], [None], 
                [np.nan, None], [np.nan, None], np.nan, None, 
                None, lam0, None, None, None, None, None, None]

    # fit continuum
    knots = np.zeros(num_specs*num_knots)
    initial_knot_values = np.zeros(num_specs*num_knots)
    for i in range(num_specs):
        start = i*num_knots
        end = (i + 1)*num_knots
        knots[start:end] = np.linspace(lam_list[i][0]*(1 + padding), 
                                     lam_list[i][-1]*(1 - padding), 
                                     num_knots)
        initial_knot_values[start:end] = (
            interp.interp1d(lam_list[i], sky_list[i])(knots[start:end]))
    spline_fit = opt.least_squares(
        spline_residual, initial_knot_values,
        args=(knots, lam_list, sky_list, raw_error_list))
    best_knot_values = spline_fit["x"]

    # scale errors 
    raw_weighted_residual = spline_residual(best_knot_values, knots, 
                                            lam_list, sky_list, 
                                            raw_error_list)
    error_scale_factors = np.ones(num_specs) # initial value
    prev_num_masked = [0]*num_specs 
    for dummy in range(max_clip_iters):
        marker = 0
        num_masked = []
        mask_list = []
        for i in range(num_specs):
            spec_length = lam_list[i].size
            sub_wr = raw_weighted_residual[marker:marker+spec_length]
            mask = np.zeros(lam_list[i].shape, dtype=bool)
            for left in lam_list[i]:
                right = left + 1.5*res_list[i]
                if right > lam_list[i][-1]:
                    break
                res_window = ((left < lam_list[i]) & 
                              (lam_list[i] < right))
                threshold = clipping_factor*error_scale_factors[i]
                if (np.all(sub_wr[res_window] > threshold) or 
                    np.all(sub_wr[res_window] < -threshold)):
                    mask[res_window] = True 
            error_scale_factors[i] = np.std(sub_wr[~mask])
            num_masked.append(int(np.sum(mask)))
            mask_list.append(mask)
        if all([n_new == n_prev for n_new, n_prev 
               in zip(num_masked, prev_num_masked)]):
            break
        else:
            prev_num_masked = copy.deepcopy(num_masked)
    error_list = [sf*raw_errors 
                  for sf, raw_errors 
                  in zip(error_scale_factors, raw_error_list)]

    # apply masks
    ldm_left = lam0*(1 - configs["halo"]["sigma_v"])
    ldm_right = lam0*(1 + configs["halo"]["sigma_v"])
    sky_list_msk = []  # erasing these, replace with masked data
    lam_list_msk = []
    error_list_msk = []
    mask_list = []
    for i in range(num_specs):
        dm_region = (ldm_left < lam_list[i]) & (lam_list[i] < ldm_right)
        to_mask = mask & (~dm_region)
        mask_list.append(to_mask)
        sky_list_msk.append(sky_list[i][~to_mask])
        lam_list_msk.append(lam_list[i][~to_mask])
        error_list_msk.append(error_list[i][~to_mask])

    # re-fit continuum
    spline_fit = opt.least_squares(
        spline_residual, best_knot_values,
        args=(knots, lam_list_msk, sky_list_msk, error_list_msk))
    refit_best_knot_values = spline_fit["x"]

        
    # fit continuum(s) + line
    guess = np.concatenate((refit_best_knot_values, [limit_guess[0]]))
    upper_bound = np.inf*np.ones(guess.size)    
    lower_bound = -np.inf*np.ones(guess.size)
    lower_bound[-num_specs:] = 0.0
    line_fit = opt.least_squares(
            spline_plus_line_residual, guess,
            args=(knots, fixed_list, lam_list_msk, 
                  sky_list_msk, error_list_msk),
            bounds = (lower_bound, upper_bound))
    best_knots = line_fit["x"][:-1]
    best_rate = line_fit["x"][-1] 

    # compare chisq 
    best_fit_chisq = line_chisq(best_rate, best_knots, knots, 
                                fixed_list, lam_list_msk, 
                                sky_list_msk, error_list_msk)
    no_fit_chisq = line_chisq(0.0, refit_best_knot_values, knots, 
                              fixed_list, lam_list_msk, 
                              sky_list_msk, error_list_msk)
    delta_chisq = no_fit_chisq - best_fit_chisq

    # marginalize over continuum params 
    threshold = chisq_step + best_fit_chisq
    knot_bounds = np.zeros((knots.size, 2))
    for i in range(num_specs):
        error = np.median(error_list[i])
        for j in range(num_knots):
            knot_index = i*num_knots + j 
            knot_bounds[knot_index, 0] = (
                best_knots[knot_index] - 2*error)    
            knot_bounds[knot_index, 1] = (
                best_knots[knot_index] + 2*error)  
    maximize_rate_sol = opt.minimize(rate_func, best_knots,
                                     bounds=knot_bounds,
                                     args=(best_rate, knots, fixed_list, 
                                           lam_list_msk, sky_list_msk, 
                                           error_list_msk, threshold,
                                           limit_guess[1]))

    limit_rate = - maximize_rate_sol.fun
    limit_knots = maximize_rate_sol.x
    return [[lmin, lmax], spec_list, knots, error_scale_factors, 
            [limit_rate, limit_knots], [best_rate, best_knots],
            delta_chisq, lam_list_msk, error_list_msk, lam0,
            sky_list, lam_list, error_list, fixed_list, 
            spec_list, res_list, mask_list]
     

def find_pc_limit(configs, data, fit_region):
    Ntrials = configs["analysis"]["Ntrials"]
    Nbins = configs["analysis"]["Nbins"]
    power_threshold = configs["analysis"]["power_threshold"]
    [lam0, lam_list_msk, error_list_msk, 
     knots, best_knots, spec_list] = fit_region
    num_knots = configs["analysis"]["num_knots"]
    # generate mock data    
    upper_limits = np.nan*np.ones(Ntrials)
    for trial in range(Ntrials):
        mock_data = []
        for i, spec_i in enumerate(spec_list):
            mock_data.append({})
            mock_data[-1]["res"] = data[spec_i]["res"]
            mock_data[-1]["D"] = data[spec_i]["D"]
            mock_data[-1]["lam"] = lam_list_msk[i]
            mock_data[-1]["error"] = error_list_msk[i]
            start = i*num_knots
            end = (i + 1)*num_knots
            Npts = error_list_msk[i].size
            draw = np.random.normal(size=Npts)*error_list_msk[i]
            continuum = interp.CubicSpline(knots[start:end], 
                                           best_knots[start:end])
            mock_data[-1]["sky"] = continuum(lam_list_msk[i]) + draw
        out = find_raw_limit(configs, mock_data, lam0)
        upper_limits[trial] = out[4][0]

    # integrate histogram
    hist = np.histogram(upper_limits, bins=Nbins)
    midpoints = 0.5*(hist[1][1:] +  hist[1][:-1])
    pdf  = interp.interp1d(midpoints, hist[0], 
                           bounds_error=False, fill_value=(0, 0))
    rate_max = np.max(midpoints)
    norm = integ.quad(pdf, 0, rate_max,
                      epsrel=1e-4, epsabs=1e-4, limit=500)[0]
    cdf_threshold = lambda gamma, threshold: (
        integ.quad(pdf, 0, gamma, epsrel=1e-4, epsabs=1e-4, 
                   limit=500)[0]/norm - threshold)
    sol = opt.root_scalar(cdf_threshold, args=(power_threshold,),
                          bracket=[0, rate_max])
    return [lam0, sol.root, upper_limits]

def find_full_limit(configs, data, lam0):
    raw_results = find_raw_limit(configs, data, lam0)
    try:
        fit_region = [raw_results[9], raw_results[7], raw_results[8], 
                      raw_results[2], raw_results[5][1], raw_results[1]]
        pc_results = find_pc_limit(configs, data, fit_region)
    except:
        pc_results = [np.nan, np.nan, None] 
    return raw_results + [pc_results[1]]

def run(data, configs, test_lams, line_output_path):
    """
    Find bestfit line and chi-sq limit on line strength
    """
    raw_limits_func = functools.partial(find_full_limit, configs, data)
    print(F"scanning {len(test_lams)} mass trials for line limits...")
    t0 = time.time()
    with open(line_output_path, "w") as wf:
        with mltproc.Pool(configs["analysis"]["Nthreads"]) as pool:
            raw_output = [] 
            try:
                for result in pool.imap(raw_limits_func, test_lams):
                    raw_output.append(result)
                    if len(raw_output) % configs["analysis"]["checkpoint"] == 0:
                        print(F"check: {len(raw_output)}")
                        json.dump(nestedtolist(copy.deepcopy(raw_output)), wf, indent=4)
            except KeyboardInterrupt:
                pass    
            json.dump(nestedtolist(copy.deepcopy(raw_output)), wf, indent=4)
    dt_raw = time.time() - t0
    print(F"completed {len(raw_output)}/{len(test_lams)}")
    print("elapsed: {:0.2f} sec".format(time.time() - t0))
    return raw_output
 
def parse_and_save(test_lams, line_output, run_name, 
                   limits_path, bestfits_path, pc_path):
    """
    Convert output to physical quantities and write to disk  
    """
    print("writing raw results...")
    raw_limits = np.asarray([out[4][0] for out in line_output])
    bestfit = np.asarray([out[5][0] for out in line_output])
    delta_chisqs = np.asarray([out[6] for out in line_output])
    error_scale_factors = np.asarray([out[3] for out in line_output])
    pc_limit = np.asarray([out[16] for out in line_output])
    # physical conversion 
    m = convert.wavelength_to_mass(test_lams)
    limit_decayrate = convert.fluxscale_to_invsec(raw_limits)
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m) 
    bestfit_decayrate = convert.fluxscale_to_invsec(bestfit)    
    bestfit_g = convert.decayrate_to_axion_g(bestfit_decayrate, m) 
    # write raw_limits output 
    limits_header = (F"DM decay limits (not power constrained) vs mass \n"
                      "JWST NIRSPEC run {run_name}\n"
                      "mass [ev]    lifetime [sec]    "
                      "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)")
    np.savetxt(limits_path, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=limits_header)
    # write bestfit output
    bestfits_header = (F"DM decay best fits vs mass \n"
                        "JWST NIRSPEC run {run_name}\n"
                        "lambda0 [micron]    mass [ev]    lifetime [sec]    "
                        "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)    "
                        "d(chisq)")
    np.savetxt(bestfits_path, 
               np.column_stack((test_lams, m, 
                                bestfit_decayrate, bestfit_g,
                                delta_chisqs)),
               header=bestfits_header)

    final_limits = np.max([raw_limits, pc_limit], axis=0)
    pc_hit = np.argmax([raw_limits, pc_limit], axis=0)
    print("power constrained fraction: {:0.2f}".format(np.sum(pc_hit)/pc_hit.size))
    # physical conversion 
    finallimit_decayrate = convert.fluxscale_to_invsec(final_limits)    
    finallimit_g = convert.decayrate_to_axion_g(finallimit_decayrate, m) 
    # write output 
    pc_header = (F"DM decay power constrained results vs mass \n"
                  "JWST NIRSPEC run {run_name}\n"
                  "lambda0 [micron]    "
                  "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)    "
                  "pc used")
    np.savetxt(pc_path, 
               np.column_stack((m, finallimit_decayrate, 
                                finallimit_g, pc_hit)),
               header=pc_header)
    return
