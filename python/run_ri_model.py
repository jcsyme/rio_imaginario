import data_functions as dat
import itertools
import math
from metric_functions import *
import multiprocessing as mp
import numpy as np
import os, os.path
import pandas as pd
import pyDOE2 as pyd
import ri_water_model as rm
import scipy.optimize as sco
import setup_analysis as sa
import time
from typing import *



#####################################
###                               ###
###    INITIALIZE KEY FUNCIONS    ###
###                               ###
#####################################


def build_futures(
    df_climate_deltas_annual: pd.DataFrame,
    df_model_data: pd.DataFrame,
):
    """
    Build LHS table and all futures.

    Function Arguments
    ------------------
    - df_climate_deltas_annual: data frame of climate deltas
    - df_model_data: baseline trajectories to modify
    """
    # build lhs samples
    dict_f0_vals, dict_ranges = get_lhs_ranges_and_base_values(df_climate_deltas_annual)
    df_lhs = dat.generate_lhs_samples(sa.n_lhs, dict_ranges, dict_f0_vals, sa.field_key_future)

    # get climate deltas
    dict_climate_factor_delta_field_map = {"flow_m3s": "flow_m3s", "precipitation_mm": "precipitation_mm"}
   
    df_climate_deltas_by_future = dat.get_climate_factor_deltas(
        df_model_data,
        df_lhs,
        dict_climate_factor_delta_field_map,
        sa.range_delta_base,
        sa.range_delta_fut,
        max(sa.model_historical_years),
        field_future_id = sa.field_key_future,
    )

    # get time periods for variation and apply other deltas
    t0 = max(
        df_model_data[
            df_model_data[sa.field_year] == min(sa.model_projection_years) - 1
        ][sa.field_time_period]
    )
    t1 = max(
        df_model_data[sa.field_time_period]
    )

    df_other_deltas_by_future = dat.get_linear_delta_trajectories_by_future(
        df_model_data,
        df_lhs[[x for x in df_lhs.columns if x not in dict_climate_factor_delta_field_map.keys()]],
        t0,
        t1,
        field_future_id = sa.field_key_future,
    )

    # merge back in some data
    df_other_deltas_by_future = pd.merge(
        df_other_deltas_by_future,
        df_model_data[[sa.field_time_period, sa.field_year, sa.field_month]]
    )

    # build final futures table
    df_futures = pd.merge(df_climate_deltas_by_future, df_other_deltas_by_future)
    fields_ind = [sa.field_key_future, sa.field_time_period, sa.field_year, sa.field_month]
    fields_dat = sorted([x for x in df_futures.columns if x not in fields_ind])
    df_futures = df_futures[fields_ind + fields_dat]

    return df_futures, df_lhs



def build_primary_attribute(
    df_fut: pd.DataFrame,
    df_strat: pd.DataFrame,
    field_key_future: str = sa.field_key_future,
    field_key_primary: str = sa.field_key_primary,
    field_key_strategy: str = sa.field_key_strategy,
) -> pd.DataFrame:

    # create a primary key
    fields_index = [field_key_strategy, field_key_future]
    field_primary_key = field_key_primary
    df_attribute_primary = pd.DataFrame(
        list(itertools.product(
            list(df_strat[field_key_strategy]),
            list(df_fut[field_key_future])
        )),
        columns = fields_index
    )
    df_attribute_primary[field_key_primary] = range(len(df_attribute_primary))
    df_attribute_primary = df_attribute_primary[[field_key_primary] + fields_index]

    return df_attribute_primary




def get_lhs_ranges_and_base_values(
    df_climate_deltas_annual: pd.DataFrame,
) -> tuple:
    """
        Get the lhs ranges and base values used to build futures

        - df_climate_deltas_annual: data frame with climate deltas used to set ranges for sampling around climate variables
    """
    #
    #  NOTE: this function would be modified to read these data from a table. For now, we specify the dictionary here
    #

    # setup ranges for lhs
    dict_ranges = {
        "flow_m3s": [
            0.95*min(df_climate_deltas_annual["delta_q_2055_annual"]),
            1.05*max(df_climate_deltas_annual["delta_q_2055_annual"])
        ],
        "precipitation_mm": [
            0.95*min(df_climate_deltas_annual["delta_p_2055_annual"]),
            1.05*max(df_climate_deltas_annual["delta_p_2055_annual"])
        ],
        "population": [0.8, 1.3],
        "demand_municipal_m3p": [0.8, 1.2],
        "demand_agricultural_m3km2": [0.9, 1.1],
        "area_ag_km2": [0.8, 1.5]
    }
    # set future 0 values - different because we'll apply deltas differently
    dict_f0_vals = {
        "flow_m3s": 0,
        "precipitation_mm": 0,
        "population": 1,
        "demand_municipal_m3p": 1,
        "demand_agricultural_m3km2": 1,
        "area_ag_km2": 1
    }

    return dict_f0_vals, dict_ranges



##  use to collect and clean results after running in parallel
def get_metrics_from_node_return(
    result: tuple,
) -> pd.DataFrame:

    id_primary, df_ret = result

    df_ret[sa.field_key_primary] = id_primary

    df_metric_1 = get_mean_reservoir(
        df_ret,
        sa.field_key_primary,
        sa.field_year,
        10
    )

    df_metric_2 = get_mean_groundwater(
        df_ret,
        sa.field_key_primary,
        sa.field_year,
        10
    )

    df_year_unacceptable, df_metric_3 = get_unacceptable_unmet_demand(
        df_ret,
        field_key_primary = sa.field_key_primary,
        field_measure = "u_2_proportion",
        field_metric_exceed = "exceed_threshes",
        field_metric_prop = "proportion_unacceptable_unmet_demand",
        field_month = sa.field_month,
        field_year = sa.field_year,
    )

    df_metrics_summary = pd.merge(df_metric_1, df_metric_2)
    df_metrics_summary = pd.merge(df_metrics_summary, df_metric_3)

    return df_metrics_summary



def get_metric_df_out(
    vec_df_out_ri: list,
    fp_template_csv_out: str = None,
    save_complete: bool = False,
) -> Tuple[pd.DataFrame, Union[pd.DataFrame, None]]:
    """
    Collect metrics from parallelized return list and form output dataframe.
        Returns tuple of form

        (df_metrics, df_complete)

        where df_complete is None if not save_complete

    Function Arguments
    ------------------
    - vec_df_out_ri: list of raw outputs from pool.async()
    
    Keyword Arguments
    -----------------
    - fp_template_csv_out: template to use ("%s"%()) to save individual files
    - save_complete: save the complete dat frame of outputs?
    """
    vec_df_metrics = []
    vec_df_complete = []

    for i in range(len(vec_df_out_ri)):

        # write out the results for the primary id
        if fp_template_csv_out is not None:
            fp_out = fp_template_csv_out%(vec_df_out_ri[i][0])
            vec_df_out_ri[i][1].to_csv(fp_out, index = None, encoding = "UTF-8")

        df_cur = get_metrics_from_node_return(vec_df_out_ri[i])

        # add metric outputs to list? spawns list if otherwise empty
        if len(vec_df_metrics) == 0:
            vec_df_metrics = [df_cur for x in vec_df_out_ri]
        else:
            vec_df_metrics[i] = df_cur[vec_df_metrics[0].columns]

        # add full outputs to list? spawns list if otherwise empty
        if save_complete:
            id_primary, df_ret = vec_df_out_ri[i]
            df_ret[sa.field_key_primary] = id_primary

            if len(vec_df_complete) == 0:
                vec_df_complete = [df_ret for x in vec_df_out_ri]
            else:
                vec_df_complete[i] = df_ret[vec_df_complete[0].columns]

    
    ##  CONVERT TO FULL OUTPUTS

    df_metrics = (
        pd.concat(
            vec_df_metrics, 
            axis = 0
        )
        .reset_index(drop = True)
    )

    df_complete = (
        (
            pd.concat(
                vec_df_complete, 
                axis = 0
            )
            .reset_index(drop = True)
        )
        if len(vec_df_complete) > 0
        else None
    )

    out = (df_metrics, df_complete)

    return out



def get_model_data_from_primary_key(
    id_primary: int,
    df_attribute_primary: pd.DataFrame,
    df_futures: pd.DataFrame,
    df_strategies: pd.DataFrame,
    field_primary_key: str = "primary_id",
    field_future: str = "future_id",
    field_strategy: str = "strategy_id",
) -> pd.DataFrame:
    """
    Get model input data for a single primary key
    """
    row_scenario = df_attribute_primary[df_attribute_primary[field_primary_key] == id_primary]
    # get ids
    id_future = int(row_scenario[field_future])
    id_primary = int(row_scenario[field_primary_key])
    id_strategy = int(row_scenario[field_strategy])

    # get input data
    df_future = df_futures[df_futures[field_future] == id_future].copy()
    df_future.drop([field_future], axis = 1, inplace = True)
    df_strategy = df_strategies[df_strategies[field_strategy] == id_strategy].copy()
    df_strategy.drop([field_strategy], axis = 1, inplace = True)
    df_input_data = pd.merge(df_future, df_strategy)

    return df_input_data



def load_data(
    fp_climate_deltas: str = sa.fp_csv_climate_deltas_annual,
    fp_model_data: str = sa.fp_csv_baseline_trajectory_model_input_data,
    fp_stratey_inputs: str = sa.fp_xlsx_strategy_inputs,
    field_key_strategy: str = sa.field_key_strategy,
) -> Tuple:

    """
    Load input data for the ri water model
    """
    # load baseline model data
    try:
        df_model_data = pd.read_csv(fp_model_data)
    except:
        raise ValueError(f"Error: model data input file {fp_model_data} not found.")

    # load climate deltas
    try:
        df_climate_deltas_annual = pd.read_csv(fp_climate_deltas)
    except:
        raise ValueError(f"Error: annual climate delta input file {df_climate_deltas_annual} not found.")

    # load strategies
    try:
        df_attr_strategy, df_strategies = dat.get_strategy_table(
            fp_stratey_inputs,
            field_strategy_id = field_key_strategy
        )
    except:
        raise ValueError(f"Error in get_strategy_table: check the file at path {fp_stratey_inputs}.")

    return df_attr_strategy, df_climate_deltas_annual, df_model_data, df_strategies



def write_output_csvs(
    dir_output: str,
    dict_write: dict = {},
    makedirs: bool = True
) -> None:
    """
    Use a dictionary to map a file name to a dataframe out

    Function Arguments
    ------------------
    - dir_output: output directory for files
    - dict_write: dictionary of form {fn_out: df_out, ...}
    - makedirs: make the directory dir_output if it does not exist
    """

    if not os.path.exists(dir_output):
        if not makedirs:
            raise ValueError(f"Error in write_output_csvs: output directory {dir_output} not found. Set makedirs = True to make the directory.")

        os.makedirs(dir_output, exist_ok = True)


    for fn in dict_write.keys():
        fp_out = os.path.join(dir_output, fn)
        dict_write[fn].to_csv(fp_out, index = None, encoding = "UTF-8")

    return True




#######################################
###                                 ###
###    SETUP MAIN MODEL FUNCTION    ###
###                                 ###
#######################################

def main():

    ##  SETUP INPUT DATA

    # read in input data
    df_attr_strategy, df_climate_deltas_annual, df_model_data, df_strategies = load_data()

    # sample LHS and build futures
    df_futures, df_lhs = build_futures(df_climate_deltas_annual, df_model_data)

    # built the attribute table and get primary ids to run
    df_attribute_primary = build_primary_attribute(df_lhs, df_attr_strategy)
    all_primaries = list(df_attribute_primary[
        df_attribute_primary[sa.field_key_strategy].isin(sa.strats_run)
    ][sa.field_key_primary])


    ##  INITIAlIZE MODEL
    model = rm.RIWaterResourcesModel()


    ##  RUN MODEL IN PARALLEL USING DATA

    # start the MP pool for asynchronous parallelization
    t0_par_async = time.time()
    print("Starting pool.async()...")

    # initialize callback function - note vec_df_out_ri is specified OUTSIDE of main()
    def get_result(result):
        global vec_df_out_ri
        vec_df_out_ri.append(result)

    #with mp.Pool() as pool:

    pool = mp.Pool()

    for id_primary in all_primaries:

        # get data for this primary key
        df_input_data = get_model_data_from_primary_key(
            id_primary,
            df_attribute_primary,
            df_futures,
            df_strategies,
            sa.field_key_primary,
            sa.field_key_future,
            sa.field_key_strategy
        )

        # apply to pool
        pool.apply_async(
            model.project,
            args = (
                df_input_data,
            ),
            kwds = {
                "id_primary": id_primary,
            },
            callback = get_result,
        )

    pool.close()
    pool.join()

    # print some info
    t1_par_async = time.time()
    t_delta = np.round(t1_par_async - t0_par_async, 2)
    print(f"Pool.async() done in {t_delta} seconds.\n")

    # collect output
    df_metrics, df_all_output = get_metric_df_out(
        vec_df_out_ri,
        save_complete = True,
    )

    # export data
    dir_return = os.path.join(sa.dir_out, sa.analysis_name)
    dict_out = {
        sa.fn_csv_attribute_future_id: df_lhs,
        sa.fn_csv_attribute_strategy_id: df_attr_strategy,
        sa.fn_csv_strategies: df_strategies,
        sa.fn_csv_attribute_primary_id: df_attribute_primary,
        sa.fn_csv_futures: df_futures,
        sa.fn_csv_metrics: df_metrics
    }
    (
        dict_out.update({sa.fn_csv_all_output: df_all_output})
        if df_all_output is not None
        else None
    )
    write_output_csvs(dir_return, dict_out)

    print("Done.")



# run
if __name__ == "__main__":

    # initialize callback vector in global context (outside of main())
    vec_df_out_ri = []

    # call main
    main()
