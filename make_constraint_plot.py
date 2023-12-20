""" 
Generate constraint plot from line and flux search results. 
"""

import sys
import os 

import numpy as np
import scipy as sp 
import matplotlib.pyplot as plt
import matplotlib.ticker as tick

import DMdecay as dmd

decay_only = {#"HST COB", 
          "MUSE":["MUSE", 3e0, 7e24, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 7e24, "0.4"]}

filenames = {"Globular Clusters":"GlobularClusters.txt", 
             # "HST COB":"HST.txt",
             "LeoT Heating":"LeoT.txt",
             "MUSE":"Telescopes_MUSE.txt", 
             "VIMOS":"Telescopes_VIMOS.txt",
             "CAST1":"CAST-CAPP.txt",
             "CAST2":"CAST_highm.txt",
             "CAST3":"CAST.txt"}

colors = {"Globular Clusters":"0.4", 
          # "HST COB":"0.75",
          "LeoT Heating":"0.6",
          "MUSE":"0.5", 
          "VIMOS":"0.65",
          "CAST1":"0.5",
          "CAST2":"0.5",
          "CAST3":"0.5"}

# name, x, y, color
labels = {"Globular Clusters":["Stellar\nEvolution", 6.1e-1, 1.2e-11, "0.4"],
          # "HST COB":["HST", 8e0, 3e-11, "0.5"],
          "MUSE":["MUSE", 3.1e0, 2e-11, "0.8"],
          "VIMOS":["VIMOS", 4.7e0, 2e-11, "0.4"],
          "CAST":["CAST", 6.3e-1, 4e-10, "0.8"]}

def qcd_axion(c_agamma, m_ev):
    return c_agamma*m_ev*(2e-10) # GeV^-1

axion_models = {"KSVZ":1.92, "DFSZI":0.75}

if __name__ == "__main__":
    config_filename = sys.argv[1]    
    configs = dmd.prep.parse_configs(config_filename)

    run_name = configs["run"]["name"]
    flux_limit_path = F"{run_name}/{configs['run']['fluxlimits_filename']}"
    flux_limit = np.loadtxt(flux_limit_path)
    line_limit_path = F"{run_name}/{configs['run']['pc_filename']}"
    line_limit = np.loadtxt(line_limit_path)
    raw_limit_path = F"{run_name}/{configs['run']['rawlimits_filename']}"
    raw_limit = np.loadtxt(raw_limit_path)
    plt.rcParams['text.usetex'] = True

    # replace nan's from power constraint with raw result
    combined_limit = line_limit[:, 0:3]
    infs = ~np.isfinite(combined_limit)
    combined_limit[infs] = raw_limit[:, 0:3][infs]
    # if raw result is also nan, just omit those datapoints 
    combined_limit = combined_limit[np.isfinite(combined_limit[:, 2]), :]

    limit_data = {}
    for name in filenames:
        path = os.path.join(configs["system"]["AxionLimits_dir"], 
                            filenames[name])
        try:
            limit_data[name] = np.loadtxt(path)
        except:
            limit_data[name] = np.loadtxt(path, delimiter=",")

    lower_edge = 1e-13
    upper_edge = 1e-9
    left_edge = 6e-1
    right_edge = 6e0
    fig, [ax_t, ax_g] = plt.subplots(2, 1)


    m_sample = np.linspace(left_edge, right_edge, 10**4)
    ax_g.fill_between(m_sample, 
                      qcd_axion(axion_models["DFSZI"]*3.0, m_sample),                      
                      qcd_axion(axion_models["KSVZ"]/3.0, m_sample),
                      color='goldenrod', alpha=0.7, linewidth=0)
    ax_g.text(0.61, 9.5e-11, 
            "QCD Axion", 
            color="darkgoldenrod",
            fontsize=10, rotation=5)


    for name in limit_data:
        if name == "Globular Clusters":
            muse_left = limit_data["MUSE"][1, 0]
            m_to_plot = np.array([left_edge, muse_left])
            limit_to_plot = np.ones(2)*limit_data[name][0, 1]
            ax_g.plot(m_to_plot, limit_to_plot, 
                    color=colors[name], marker='', 
                    linestyle='dotted', linewidth=1.5,
                    alpha=0.6)
        else:
            ax_g.fill_between(limit_data[name][:,0], limit_data[name][:,1], 
                            upper_edge, color=colors[name], linewidth=0)
    for name in labels:
        ax_g.text(labels[name][1], labels[name][2], 
                labels[name][0], color=labels[name][3], size=12)
    for name in decay_only:
        ax_t.text(decay_only[name][1], decay_only[name][2], 
                  decay_only[name][0], color=decay_only[name][3], size=12)

    ax_g.fill_between(combined_limit[:,0], combined_limit[:,2], 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.9, edgecolor=None)


    # tame flux limit 
    flux_limit = flux_limit[np.isfinite(flux_limit[:, 2]), :]
    flux_limit = flux_limit[4::, :]

    ax_g.plot(flux_limit[:,0], flux_limit[:,2], 
            color="black", linewidth=2)



    # scale estimate 
    time = np.asarray([2, 15])*(3e7) #sec
    style= ['dashed', 'dotted']
    line_colors = ['darkgreen', 'darkblue']
    t0 = 2e3 #sec
    efficency = 0.01
    width = 100
    print("estimate for g:")
    for t, s, c in zip(time, style, line_colors):
        scaled_limit = combined_limit[:, 2]*(t*efficency/t0)**(-0.25)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_g.plot(combined_limit[:, 0], smoothed_limit,
            color=c, linewidth=1.2, linestyle=s,
            marker='', alpha=0.6) 
        print("    {:d} yr:  {:0.2e} - {:0.2e}"
              "".format(int(t/3e7), 
                        np.min(smoothed_limit), 
                        np.max(smoothed_limit)))
    print()

    ax_g.set_ylim([lower_edge, upper_edge])
    ax_g.set_xlim([left_edge, right_edge])
    ax_g.set_yscale('log')
    ax_g.set_xscale('log')

    ax_g.text(0.85e0, 16e-11, 
            "Total Flux", 
            color="black",
            fontsize=12, rotation=0)

    ax_g.text(1.4, 3e-10, 
        "Continuum Model", 
        color="lightcoral",
        fontsize=15, rotation=0)
    # ax_g.text(6.35e-1, 6e-12, 
    #     "Model", 
    #     color="firebrick",
    #     fontsize=13, rotation=0)


    ax_g.text(8e-1, 9e-12, 
        "Current", rotation=-4,
        color="darkgreen",
        fontsize=12)

    ax_g.text(8e-1, 2e-12, 
        "15 year", rotation=-3,
        color="darkblue",
        fontsize=13)


    ax_g.set_xlabel(r"$\displaystyle m_a\; [{\rm \tiny eV }]$", 
                  fontsize=16)
    ax_g.set_ylabel(
        r"$\displaystyle g_{a\gamma\gamma}\; [{\rm \tiny GeV }^{-1}]$", 
        fontsize=16)


    lifetime_upper_edge = 3e29
    lifetime_lower_edge = 1e23

    for name in decay_only:
        rate_limit = dmd.conversions.axion_g_to_decayrate(
            limit_data[name][:,1], 
            limit_data[name][:,0])
        ax_t.fill_between(limit_data[name][:,0], lifetime_lower_edge,
                        rate_limit**-1, color=colors[name], 
                        linewidth=0, alpha=1)


    ax_t.fill_between(combined_limit[:,0], combined_limit[:,1]**-1, 
                    upper_edge, facecolor="firebrick", 
                    linewidth=1, alpha=0.85, edgecolor=None)

    ax_t.vlines(flux_limit[[0, -1], 0],
              flux_limit[[0, -1], 1]**-1,
              [upper_edge, upper_edge],
              color="firebrick", linewidth=0.75)

    ax_t.plot(flux_limit[:,0], flux_limit[:,1]**-1, 
            color="Black", linewidth=2)

    # scale estimate 
    print("estimate for tau:")
    for t, s, c in zip(time, style, line_colors):
        scaled_limit = (combined_limit[:, 1]**-1)*(t*efficency/t0)**(0.5)
        smoothed_limit = np.exp(
            sp.ndimage.gaussian_filter(np.log(scaled_limit), width))
        ax_t.plot(combined_limit[:, 0], smoothed_limit,
            color=c, linewidth=1.2, linestyle=s,
            marker='', alpha=0.6) 
        print("    {:d} yr:  {:0.2e} - {:0.2e}"
              "".format(int(t/3e7), 
                        np.min(smoothed_limit), 
                        np.max(smoothed_limit)))

    ax_t.text(1.7, 4.5e23, 
            "Total Flux", 
            color="black",
            fontsize=12, rotation=-6)

    ax_t.text(8.2e-1, 3e23, 
        "Continuum Model", 
        color="lightcoral",
        fontsize=15, rotation=0)   

    ax_t.text(8e-1, 9e26, 
        "Current", 
        color="darkgreen",
        fontsize=12, rotation=-2)

    ax_t.text(2e0, 2.2e27, 
        "15 year", 
        color="darkblue",
        fontsize=12, rotation=-4)


    ax_t.set_ylim([lifetime_lower_edge, lifetime_upper_edge])
    ax_t.set_xlim([left_edge, right_edge])
    ax_t.set_yscale('log')
    ax_t.set_xscale('log')
    ax_t.set_ylabel(r"$\displaystyle \tau \; [{\rm \tiny sec }]$", 
                  fontsize=16)

    # locmin = tick.LogLocator(base=10.0,
    #                          subs=(0.2,0.4,0.6,0.8),
    #                          numticks=12)
    # ax_t.xaxis.set_minor_locator(locmin)
    # ax_t.xaxis.set_minor_formatter(tick.NullFormatter())

    ax_t.tick_params(axis='both', which='major', labelsize=13)
    ax_g.tick_params(axis='both', which='major', labelsize=13)
    ticks = [0.6, .8, 1, 2, 3, 4, 5, 6]
    ax_t.set_xticks(ticks)
    ax_t.set_xticklabels(ticks)
    ax_g.set_xticks(ticks)
    ax_g.set_xticklabels(ticks)
    ax_g.yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    ax_t.set_yticks([1e23, 1e24, 1e25, 1e26, 1e27, 1e28, 1e29])
    ax_t.yaxis.get_minor_locator().set_params(numticks=99, subs=[.2, .4, .6, .8])
    [label.set_visible(False) for label in ax_t.yaxis.get_ticklabels()[1::2]]

    constraint_path = "{}/constraints.pdf".format(configs["run"]["name"])
    fig.set_size_inches(8, 6)
    fig.tight_layout(pad=1)
    fig.savefig(constraint_path, dpi=300, bbox_inches="tight")