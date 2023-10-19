#!/usr/bin/env python3
# coding: utf-8


import time  
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
    
# fixed = [lam0, D factor, line width (sigma via resolution)]
dm_line = lambda lam, fixed, rate: (
    mw.MWDecayFlux_old(lam, fixed[0], rate, fixed[1], fixed[2]))

def replace_centers(central_knots, best_line_knots, centers):
    knot_values = best_line_knots.copy()
    knot_values[centers] = central_knots 
    return knot_values

def find_limits(window, padding, num_knots, 
                chisq_step, limit_guess, data, lam0):
    # extract line search region 
    sky_list = []
    lam_list = []
    raw_error_list = []
    fixed_list = []
    spec_list = []
    lmin = lam0*(1.0 - 0.5*window)
    lmax = lam0*(1.0 + 0.5*window)
    for i, spec in enumerate(data):
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
    num_specs = len(sky_list)
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
    weighted_residual = spline_residual(best_knot_values, knots, 
                                        lam_list, sky_list, 
                                        raw_error_list)

    # scale errors
    error_list = []
    error_scaling = []
    marker = 0
    for i in range(num_specs):
        spec_length = lam_list[i].size
        sub_wr = weighted_residual[marker:marker+spec_length]
        sub_num_dof = spec_length - num_knots
        raw_chisq_dof = np.sum(sub_wr**2)/sub_num_dof 
        error_list.append(raw_error_list[i]*np.sqrt(raw_chisq_dof))
        error_scaling.append(np.sqrt(raw_chisq_dof))
        
    # fit continuum(s) + line
    guess = np.concatenate((best_knot_values, [limit_guess]))
    upper_bound = np.inf*np.ones(guess.size)    
    lower_bound = -np.inf*np.ones(guess.size)
    lower_bound[-num_specs:] = 0.0
    line_fit = opt.least_squares(
            spline_plus_line_residual, guess,
            args=(knots, fixed_list, lam_list, sky_list, error_list),
            bounds = (lower_bound, upper_bound))
    best_line_knots = line_fit["x"][:-1]
    best_rate = line_fit["x"][-1] 

    # compare chisq 
    best_fit_chisq = line_chisq(best_rate, best_line_knots, knots, 
                                fixed_list, lam_list, 
                                sky_list, error_list)
    no_fit_chisq = line_chisq(0.0, best_knot_values, knots, 
                              fixed_list, lam_list, 
                              sky_list, error_list)
    delta_chisq = no_fit_chisq - best_fit_chisq

    # marganilize over continuum params 
    best_chisq = line_chisq(best_rate, best_line_knots, knots, 
                            fixed_list, lam_list, sky_list, error_list)
    threshold = chisq_step + best_chisq
    centers = np.arange(knots.size)
    center_bounds = np.zeros((centers.size, 2))
    for i in range(num_specs):
        error = np.median(error_list[i])
        for j in range(num_knots):
            knot_index = i*num_knots + j 
            center_bounds[knot_index, 0] = (
                best_line_knots[knot_index] - 2*error)    
            center_bounds[knot_index, 1] = (
                best_line_knots[knot_index] + 2*error)  
        
    def rate_func(central_knots): 
        knot_values = replace_centers(central_knots,
                                      best_line_knots, 
                                      centers)
        chisq_bestrate = line_chisq(best_rate, knot_values, knots, 
                                    fixed_list, lam_list, sky_list, 
                                    error_list, threshold)
        if chisq_bestrate < -padding:
            upper_brackets = 10.0**np.arange(-4, 1)
            for upper_bracket in upper_brackets:
                chisq_upper = line_chisq(upper_bracket, knot_values, knots, 
                                         fixed_list, lam_list, sky_list, 
                                         error_list, threshold)
                if chisq_upper > padding:
                    sol = opt.root_scalar(line_chisq, 
                        args=(knot_values, knots, fixed_list, lam_list, 
                              sky_list, error_list, threshold),
                        bracket=[best_rate, upper_bracket])
                    return -sol.root
            print("failed to find upper lim\n"
                  "lam0 = {}\n"
                  "chisq(best_rate) = {}\n"
                  "chise({}) = {}\n".format(lam0, chisq_bestrate, 
                                            upper_bracket, chisq_upper))
            return -best_rate
        else:
            return -best_rate

    best_centers = [best_line_knots[i] for i in centers]
    maximize_rate_sol = opt.minimize(rate_func, best_centers,
                                     bounds=center_bounds)
    limit = - maximize_rate_sol.fun
    limit_knots = best_line_knots.copy()
    limit_knots[centers] = maximize_rate_sol.x
    return [[lmin, lmax], spec_list, knots, error_scaling, 
            [limit, limit_knots], [best_rate, best_line_knots],
            delta_chisq]

if __name__ == "__main__":

    data, target = assume.parse_sub(assume.all_paths)
    
    width_factor = 150
    v_dm = 7e-4
    window = width_factor*v_dm
    padding = 1e-8
    num_knots = 5
    chisq_step = 4
    limit_guess = 1e-4

    lstart = [spec["lam"][0] for spec in data]
    lend = [spec["lam"][-1] for spec in data]
    l_initial = np.min(lstart)
    l_final = np.max(lend)

    # dlam = l_initial*v_dm*0.5*50  # run subsample for testing 
    # test_lams = np.arange(l_initial + 0.5*dlam,
    #                       l_final - 0.5*dlam, dlam)

    test_lams = [l_initial]
    while test_lams[-1] < l_final:
        dlam_i = 2*np.min([sigma_from_fwhm(spec["max_res"], test_lams[-1]) 
                           for spec in data]) #rescale for testing
        test_lams.append(test_lams[-1] + dlam_i)
    test_lams = np.asarray(test_lams[1:-1]) # the last one is always outside the range

    loop_only = functools.partial(find_limits, 
                                  window, padding, num_knots,
                                  chisq_step, limit_guess, data)
    print("scanning {} mass trials...".format(len(test_lams)))
    t0 = time.time()
    with mltproc.Pool() as pool:
        output = pool.map(loop_only, test_lams)    
            #   returns [[lmin, lmax], 
            #            spec_list, 
            #            knots, 
            #            error_scaling, 
            #            [limit, limit_knots], 
            #            [best_rate, best_line_knots]]
    dt = time.time() - t0
    print("{} sec".format(dt))

    limits = np.asarray([out[4][0] for out in output])
    m = convert.wavelength_to_mass(test_lams)
    limit_decayrate = convert.fluxscale_to_invsec(
        limits, assume.rho_s, assume.r_s)    
    limit_g = convert.decayrate_to_axion_g(limit_decayrate, m) 

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

    bestfit = np.asarray([out[5][0] for out in output])
    bestfit_decayrate = convert.fluxscale_to_invsec(
        bestfit, assume.rho_s, assume.r_s)    
    bestfit_g = convert.decayrate_to_axion_g(bestfit_decayrate, m) 
    delta_chisqs = np.asarray([out[6] for out in output])

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

