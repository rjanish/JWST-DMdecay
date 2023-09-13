#!/usr/bin/env python3
""" Download data from JWST archives """


import os 

from astroquery.mast import Observations as obv 


project_dir = "/home/rjanish/physics/optical-ir-axion-decay"

if __name__ == "__main__":
    if os.getcwd() != project_dir:
        print("ERROR: only run from {}".format(project_dir))
        exit()

    # download GNZ11 NIRSPEC IFU observations 
    obv_table  = obv.query_criteria(obs_collection="JWST", 
                                    instrument_name="NIRSPEC/IFU") #,
                                    # target_name="GNZ11")
    
    for run in obv_table:
        product_list = obv.get_product_list(run)
        mask = ((product_list["productSubGroupDescription"] == "X1D") & 
                (product_list["calib_level"] == 3))
        if mask.sum() > 0:
            print()
            print(run["target_name"])
            print(product_list[mask])
            # obv.download_products(product_list[mask],
            #                       download_dir="{}/data".format(project_dir))