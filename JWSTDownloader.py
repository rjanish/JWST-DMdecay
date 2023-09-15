#!/usr/bin/env python3
""" Download data from JWST archives """


import os 

from astroquery.mast import Observations as obv 
import astropy.table as table


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"
download_dir = os.path.join(project_dir, "data")

if __name__ == "__main__":
    if os.getcwd() != project_dir:
        print("ERROR: only run from {}".format(project_dir))
        exit()

    # download NIRSPEC IFU observations 
    allobv_table  = obv.query_criteria(obs_collection="JWST", 
                                    instrument_name="NIRSPEC/IFU",
                                    target_name=["GNZ11", "NGC-7319"])
    target_obvs = table.Table(dtype=allobv_table.dtype)
    target_products = table.Table(
        dtype=obv.get_product_list(allobv_table[0]).dtype)
    for run in allobv_table:
        product_list = obv.get_product_list(run)
        mask = ((product_list["productSubGroupDescription"] == "X1D") & 
                (product_list["calib_level"] == 3))
        if mask.sum() > 0:
            target_obvs.add_row(run)
            target_products = table.vstack(
                (target_products, product_list[mask]))
    for filename_base, table_to_save in zip(
        ["selected_observations", "selected_data_products"],
        [target_obvs, target_products]):
        table_to_save.write(os.path.join(download_dir, 
                                       "{}.csv".format(filename_base)),
                          format='ascii.csv', overwrite=True)
        table_to_save.write(os.path.join(download_dir, 
                                       "{}.html".format(filename_base)),
                          format='ascii.html', overwrite=True)
    obv.download_products(target_products, download_dir=download_dir)