#!/usr/bin/env python3
""" Download data from JWST archives """


import os 
import time  

from astroquery.mast import Observations as obv 
import astropy.table as table

import JWSTparsedatafiles as parse


if __name__ == "__main__":
    # download NIRSPEC IFU observations 
    allobv_table  = obv.query_criteria(
        obs_collection="JWST", 
        instrument_name=["NIRSPEC/IFU","NIRSPEC/MSA"])
        # target_name=["GNZ11", "NGC-7319"])
    target_obvs = table.Table(dtype=allobv_table.dtype)
    Nobvs = len(allobv_table)
    print("found {} canditate observations\n"
          "searching for calibrated 1D spectra... ".format(Nobvs))
    Nstages = 10
    stage_size = Nobvs//Nstages
    prev_stage = 0 
    target_products = table.Table(
        dtype=obv.get_product_list(allobv_table[0]).dtype)
    t00 = time.time()
    t0 = time.time()
    for index, run in enumerate(allobv_table):
        product_list = obv.get_product_list(run)
        # select only those observation that have available 
        # fully calibrated (level 3) 1D extracted spectrum 
        mask = ((product_list["productSubGroupDescription"] == "X1D") & 
                (product_list["calib_level"] == 3))
        if mask.sum() > 0:
            target_obvs.add_row(run)
            target_products = table.vstack(
                (target_products, product_list[mask]))
        stage = index//stage_size
        if stage > prev_stage:
            dt = time.time() - t0 
            t0 = time.time()
            print("{}/{}: {:0.2f} sec".format(prev_stage + 1, Nstages, dt))
            prev_stage = stage
    dt_total = time.time() - t00
    print("total: {:0.2f} min".format(dt_total/60.0))
    print("found {} calibrated 1D spectra\n"
          "downloading spectra... ".format(len(target_products)))
    obv.download_products(target_products, download_dir=assume["paths"]["download_dir"])
    for name, table_to_save in zip(
        ["selected_observations", "selected_data_products"],
        [target_obvs, target_products]):
        table_to_save.write("{}/{}.csv".format(assume["paths"]["data_dir"], name),
                          format='ascii.csv', overwrite=True)
        table_to_save.write("{}/{}.html".format(assume["paths"]["data_dir"], name),
                          format='ascii.html', overwrite=True)