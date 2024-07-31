import importlib
import itertools
import math
import matplotlib.pyplot as plt
import multiprocessing as mp
import numpy as np
import os, os.path
import pandas as pd
import scipy.optimize as sco
import setup_analysis as sa
import time
import ri_water_model as wm



###########################################################
###                                                     ###
###    PLACEHOLDER FOR GENERATING LHS, FUTURES, ETC.    ###
###                                                     ###
###########################################################

# check to ensure current module is "__main__"
if __name__ == "__main__":

    if sa.generate_experimental_design_q:

        print("Building model input database and primary key...")

        # read in strategies and futures (normally, futures would be defined using an experimental design/model input database script; we skip that step here)
        df_futures = pd.read_csv(sa.fp_csv_futures)
        df_strategies = pd.read_csv(sa.fp_csv_strategies)

        # set up attribute of primary key
        df_attribute_primary_id = pd.DataFrame([tuple(x) for x in itertools.product(df_strategies["strategy_id"].unique(), df_futures["future_id"].unique())])
        df_attribute_primary_id.rename(columns = {0: "strategy_id", 1: "future_id"}, inplace = True)
        fields_nonprimary_ids = list(df_attribute_primary_id.columns)
        df_attribute_primary_id[sa.field_primary_key] = range(len(df_attribute_primary_id))
        df_attribute_primary_id = df_attribute_primary_id[[sa.field_primary_key] + cols]

        # build input database
        df_model_input_database = df_attribute_primary_id.copy()
        df_model_input_database = pd.merge(df_model_input_database, df_futures, how = "inner")
        df_model_input_database = pd.merge(df_model_input_database, df_strategies, how = "inner")
        df_model_input_database = df_model_input_database[[x for x in df_model_input_database.columns if (x not in fields_nonprimary_ids)]]
        df_model_input_database.sort_values(by = [sa.field_primary_key, "year", "month"]).reset_index(drop = True)

        # export files - default set of futures to run, model input database
        print("Exporting files...")
        df_attribute_primary_id.to_csv(sa.fp_csv_attribute_primary_id)
        df_attribute_primary_id[[sa.field_primary_key]].to_csv(sa.fp_csv_experiment_primary_ids, index = None, encoding = "UTF-8")
        df_model_input_database.to_csv(sa.fp_csv_model_input_database, index = None, encoding = "UTF-8")

        print("Done.")


    #
    #    READ IN PRIMARIES TO RUN AS WELL AS MODEL INPUT DATABASE
    #

    df_attribute_primary_id = pd.read_csv(sa.fp_csv_attribute_primary_id)
    df_experiment_primary_ids = pd.read_csv(sa.fp_csv_experiment_primary_ids)
    df_model_input_database = pd.read_csv(sa.fp_csv_model_input_database)
    primaries_to_run = list(set(df_experiment_primary_ids[sa.field_primary_key]))

    # reduce for this runset
    df_model_input_database = df_model_input_database[df_model_input_database["year"].isin(range(sa.dict_init["model_year_0"], sa.dict_init["model_year_1"] + 1))]
    df_model_input_database = df_model_input_database[df_model_input_database[sa.field_primary_key].isin(df_experiment_primary_ids[sa.field_primary_key])]
    df_model_input_database = df_model_input_database.sort_values(by = [sa.field_primary_key, "year", "month"]).reset_index(drop = True)



    ##########################################################
    ###                                                    ###
    ###    RUN MODEL USING ASYNCHRONOUS PARALLELIZATION    ###
    ###                                                    ###
    ##########################################################


    
    t0_par_async = time.time()
    
    # initialize output vector/array and set up dummy functions to get results
    vec_collect_run_status = []
    def get_result(result):
        global vec_collect_run_status
        # update
        vec_collect_run_status.append(result)

    # set the number of CPUs
    n_cpus = mp.cpu_count()
    # start the MP pool for asynchronous parallelization
    pool = mp.Pool(n_cpus)
    # for ease, create a list to store output file paths in
    fps_out = []
    pk_run = primaries_to_run
    
    #notify
    print(f"\n***\tStarting paralellization on {n_cpus} cores for {len(pk_run)} scenarios\t***\n")

    # apply the function; note: if the function only takes one argument (e.g., f(x)), make sure the args is args = (x, ) - that extra comma is important
    for p in pk_run:
        # set the output file path
        fp_out_cur = sa.fpt_csv_model_output_database_parcomponent%(p)
        fps_out.append(fp_out_cur)
        
        df_run = df_model_input_database[df_model_input_database[sa.field_primary_key] == p]
        pool.apply_async(
            # target function
            wm.ri_water_resources_model,
            # function arguments 
            args = (df_run, wm.md_dict_initial_states, wm.md_dict_parameters, wm.md_dict_default_levers, {"primary_id": p}, {"writecsv": fp_out_cur}),
            callback = get_result
        )

    pool.close()
    pool.join()
    # print timing
    t1_par_async = time.time()
    t_elapse_par_async = t1_par_async - t0_par_async
    print("Time elapsed to in parallelization: %s seconds."%(t_elapse_par_async))

    
    ##  collect parallel files, write to single file

    init_q = True
    for fp in fps_out:
        #create new file
        df_tmp = pd.read_csv(fp)
        #write to new file?
        if init_q:
            # check for primary key and order accordingly
            fields_ord = [sa.field_primary_key, "year", "month"]
            fields_ord = [x for x in fields_ord if x in df_tmp.columns]
            fields_dat = [x for x in df_tmp.columns if (x not in fields_ord)]
            fields_dat.sort()
            # set header and reorganize
            header_write = fields_ord + fields_dat
            df_tmp = df_tmp[header_write]
            # write to output; explicit output options here 
            df_tmp.to_csv(sa.fpt_csv_model_output_database, index = None, encoding = "UTF-8", mode = "w", header = True)

            # turn off initialization
            init_q = False
        else:
            # use mode = "a" to append; note that the header is stored in header_write
            df_tmp[header_write].to_csv(sa.fpt_csv_model_output_database, index = None, encoding = "UTF-8", mode = "a", header = False)

        # remove the file after successful re-write
        os.remove(fp)
    
    # notify timing
    t2_par_async = time.time()
    t_elapse_par_async_wreagg = t2_par_async - t0_par_async
    print("Total time elapsed, including reaggregation of parallel files: %s seconds."%(t_elapse_par_async_wreagg))


    ################################################################
    #  CHECK CLOUD INFO, CALL CLOUD CLOSEOUT SCRIPT IF NECESSARY   #
    ################################################################

    if sa.dict_init["cloud_run_q"]:
        print(f"Calling 'cloud_complete' to copy files and terminate instance id '{sa.dict_init["instance_id"]}' (num {sa.dict_init["instance_num"]})")
        import cloud_complete
    







