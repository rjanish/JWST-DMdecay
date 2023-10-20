#!/usr/bin/env python3
# coding: utf-8

import time  
import copy
import multiprocessing as mltproc  
import functools

import matplotlib.pyplot as plt
import numpy as np 
import scipy.optimize as opt
import scipy.interpolate as interp

import DMdecayJWST as assume
import JWSTparsedatafiles as JWSTparse 
import MWDMhalo as mw
import conversions as convert 


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

def sigma_from_fwhm(fwhm, lam0):
    sigma_inst = fwhm/(2*np.sqrt(2*np.log(2)))
    return np.sqrt(sigma_inst**2 + (lam0*assume.sigma_v)**2)
    
def dm_line(lam, fixed, rate):
    return mw.MWDecayFlux_old(lam, fixed[0], rate, fixed[1], fixed[2])

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

def find_raw_limit(setup_params, data, lam0):
    window = setup_params["window"]
    padding = setup_params["padding"]
    num_knots = setup_params["num_knots"]
    max_clip_iters = setup_params["max_clip_iters"]
    clipping_factor = setup_params["clipping_factor"]
    chisq_step = setup_params["chisq_step"]
    limit_guess = setup_params["limit_guess"] 
    # extract line search region 
    sky_list = []
    lam_list = []
    raw_error_list = []
    fixed_list = []
    spec_list = []
    res_list = []
    lmin = lam0*(1.0 - 0.5*window)
    lmax = lam0*(1.0 + 0.5*window)
    print(lmin, lam0, lmax)
    for i, spec in enumerate(data):
        print(lstart[i], lend[i])
        if (lam0 < lstart[i]) or (lam0 > lend[i]):
            continue
        if lmin < lstart[i]:
            lmin = lstart[i]
            lmax = lmin*(1 + window)
        elif lend[i] < lmax:
            lmax = lend[i]
            lmin = lmax*(1 - window)
        select = (lmin < spec["lam"]) & (spec["lam"] < lmax)
        spec_list.append(i)
        sky_list.append(spec["sky"][select])
        lam_list.append(spec["lam"][select])
        raw_error_list.append(spec["error"][select])
        fixed_list.append([lam0, spec["D"], 
                           sigma_from_fwhm(spec["max_res"], lam0)])    
        res_list.append(spec["max_res"])
    num_specs = len(sky_list)
    print(num_specs)
    print(lmin, lam0, lmax)
    print()
    if num_specs == 0:
        return [[lmin, lmax], spec_list, [None], [None], 
                [np.nan, None], [np.nan, None], np.nan]
                ## why is this executing ???

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
    ldm_left = lam0*(1 - v_dm)
    ldm_right = lam0*(1 + v_dm)
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

        
    # fit continuum(s) + line
    guess = np.concatenate((best_knot_values, [limit_guess[0]]))
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
    no_fit_chisq = line_chisq(0.0, best_knot_values, knots, 
                              fixed_list, lam_list_msk, 
                              sky_list_msk, error_list_msk)
    delta_chisq = no_fit_chisq - best_fit_chisq

    # marganilize over continuum params 
    best_chisq = line_chisq(best_rate, best_knots, knots, 
                            fixed_list, lam_list_msk, sky_list_msk, 
                            error_list_msk)
    threshold = chisq_step + best_chisq
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
            delta_chisq, lam_list_msk, error_list_msk, lam0]
     

def find_pc_limit(Ntrials, Nbins, power_threshold, 
                 setup_params, data, fit_region):
    [lam0, lam_list_msk, error_list_msk, 
     knots, best_knots] = fit_region
    num_knots = setup_params["num_knots"]
    guess = np.concatenate((best_knots, [0.0])) # fit starting

    # generate mock data    
    upper_limits = np.nan*np.ones(Ntrials)
    for trial in range(Ntrials):
        mock_data = []
        trial_sky_list_msk = []
        for i in range(len(data)):
            mock_data.append({})
            mock_data[-1]["max_res"] = data[i]["max_res"]
            mock_data[-1]["D"] = data[i]["D"]
            mock_data[-1]["lam"] = lam_list_msk[i]
            mock_data[-1]["error"] = error_list_msk[i]
            start = i*num_knots
            end = (i + 1)*num_knots
            Npts = error_list_msk[i].size
            draw = np.random.normal(size=Npts)*error_list_msk[i]
            continuum = interp.CubicSpline(knots[start:end], 
                                           best_knots[start:end])
            mock_data[-1]["sky"] = continuum(lam_list_msk[i]) + draw
        out = find_raw_limit(setup_params, mock_data, lam0)
        upper_limits[trial] = out[4][0]

    # integrate histogram
    print(upper_limits)
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
    return [lam0, sol.root]


if __name__ == "__main__":

    data, target = assume.parse_sub(assume.gnz11_paths)
    
    # raw limit params 
    setup_params = {}
    width_factor = 150
    v_dm = 7e-4
    setup_params["window"] = width_factor*v_dm
    setup_params["padding"] = 1e-8
    setup_params["num_knots"] = 5
    setup_params["chisq_step"] = 2.71 #4
    setup_params["limit_guess"] = [1e-5, 1e-3]
    setup_params["max_clip_iters"] = 100
    setup_params["clipping_factor"] = 2
    # pc limit params 
    pc_step_factor = 6 # subsampling factor for computing pc limit
    Ntrials = 10**2
    Nbins = 15
    power_threshold = 0.1587
    # run params
    Nthreads = 1
    inflate = 50   # set > 1 to undersample mass range for testing

    # generate mass sampling
    lstart = [spec["lam"][0] for spec in data]
    lend = [spec["lam"][-1] for spec in data]
    l_initial = np.min(lstart)
    l_final = np.max(lend)
    test_lams = [l_initial]
    while test_lams[-1] < l_final:
        dlam_i = 2*np.min([sigma_from_fwhm(spec["max_res"], test_lams[-1]) 
                           for spec in data])*inflate 
        test_lams.append(test_lams[-1] + dlam_i)
    test_lams = np.asarray(test_lams[1:-1]) # strip to stay inside data bounds

    # get raw, non-pc limited bounds
    raw_limits_func = functools.partial(find_raw_limit, setup_params, data)
    print("scanning {} mass trials for raw limits...".format(len(test_lams)))
    t0 = time.time()
    with mltproc.Pool(Nthreads) as pool:
        raw_output = pool.map(raw_limits_func, test_lams)    
    limits = np.asarray([out[4][0] for out in raw_output])
    bestfit = np.asarray([out[5][0] for out in raw_output])
    delta_chisqs = np.asarray([out[6] for out in raw_output])
    dt = time.time() - t0
    print("{:0.2f} sec".format(dt))

    # compute pc limits 
    pc_inputs = [[out[9], out[7], out[8], out[2], out[5][1]] 
                 for out in raw_output[::pc_step_factor]]
    pc_limits_func = functools.partial(find_pc_limit, Ntrials, Nbins, 
                                       power_threshold, setup_params, data)
    print("scanning {} mass trials for pc bounds...".format(len(pc_inputs)))
    t0 = time.time()
    with mltproc.Pool(Nthreads) as pool:
        pc_output = pool.map(pc_limits_func, pc_inputs)    
    dt = time.time() - t0
    pc_limits = np.asarray(pc_output)
    print("{:0.2f} sec".format(dt))
    for l, g in pc_limits:
        print("{:0.2f}    {:0.2e}".foramt(l , g))

    # physical conversion 
    m = convert.wavelength_to_mass(test_lams)
    limit_decayrate = convert.fluxscale_to_invsec(
        limits, assume.rho_s, assume.r_s)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m) 
    bestfit_decayrate = convert.fluxscale_to_invsec(
        bestfit, assume.rho_s, assume.r_s)    
    bestfit_g = convert.decayrate_to_axion_g(bestfit_decayrate, m) 

    # write output 
    run_name = "gnz11_ngc6552_final"
    line_results_dir = "{}/continuum".format(run_name)

    limits_path = ("{}/JWST-NIRSPEC-limits.dat"
                   "".format(line_results_dir))
    limits_header = ("DM decay limits vs mass \n"
              "JWST NIRSPEC run {}\n"
              "mass [ev]    lifetime [sec]    "
              "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)"
              "".format(run_name))
    np.savetxt(limits_path, 
               np.column_stack((m, limit_decayrate, limit_g)),
               header=limits_header)

    bestfits_path = ("{}/JWST-NIRSPEC-bestfits.dat"
                     "".format(line_results_dir))
    bestfits_header = ("DM decay best fits vs mass \n"
                       "JWST NIRSPEC run {}\n"
                       "lambda0 [micron]    mass [ev]    lifetime [sec]    "
                       "g_a\\gamma\\gamma [GeV^-1] (for vanilla axion)    "
                       "d(chisq)"
                       "".format(run_name))
    np.savetxt(bestfits_path, 
               np.column_stack((test_lams, m, 
                                bestfit_decayrate, bestfit_g,
                                delta_chisqs)),
               header=bestfits_header)

