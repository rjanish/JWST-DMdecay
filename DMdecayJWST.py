# #!/usr/bin/env python3
# """ top-level assumptions and parameters for JWST DM search """

# import tomli

# import numpy as np
# import astropy.coordinates as coord 
# import astropy.units as u

# import JWSTparsedatafiles as JWSTparse 

# def parse_configs(config_filenames):
#     assume = {}
#     for filename in config_filenames:
#         with open(filename, "rb") as f: 
#             print(F"parsing {filename}")
#             assume.update(tomli.load(f))
#     # compute DM halo data
#     assume["mw_halo"]["v_sun"] = np.asarray(assume["mw_halo"]["v_sun"])
#     assume["mw_halo"]["sigma_v"] = (
#         assume["mw_halo"]["sigma_v_kms"]/assume["mw_halo"]["c_kms"])
#     assume["mw_halo"]["galcen"] = coord.Galactocentric(
#         galcen_coord=coord.SkyCoord(ra=assume["mw_halo"]["ra_gc"]*u.deg, 
#                                     dec=assume["mw_halo"]["dec_gc"]*u.deg),
#         galcen_distance=assume["mw_halo"]["r_sun"]*u.kpc,
#         galcen_v_sun=assume["mw_halo"]["v_sun"]*u.km/u.s,
#         z_sun = assume["mw_halo"]["z_sun"]*u.pc)
#     assume["mw_halo"]["vec_sun"] = np.asarray(
#         [-assume["mw_halo"]["r_sun"], 
#          0.0,
#          assume["mw_halo"]["z_sun"]]) 
#          # kpc, cartesain position of sun in galcen fame 
#     # construct run paths 
#     assume["run_data"]["paths"] = [
#         "{}/{}".format(assume["paths"]["data_dir"], f) 
#         for f in assume["run_data"]["filenames"]]
#     return assume

# def parse_sub(assume): 
#     """ Truncate the blue GN-z11 spectrum at the start of the red one """
#     data, targets = JWSTparse.process_target_list(assume)
#     gnz11_split = assume["run_setup"]["gnz11_split"] # specific to gnz-11
#     gnz11_min = assume["run_setup"]["gnz11_min"] # specific to gnz-11
#     for spec in data:
#         if (spec["lam"][0] < gnz11_split) and (spec["name"] == "GN-z11"):
#             select = (gnz11_min < spec["lam"]) & (spec["lam"] < gnz11_split)
#             spec["lam"] = spec["lam"][select]
#             spec["sky"] = spec["sky"][select]
#             spec["error"] = spec["error"][select]
#     return data, targets 