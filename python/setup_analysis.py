import datetime as dt
import os, os.path
import pandas as pd
import support_functions as sf
from typing import *

    
    
#############################
#    SOME INITIALIZATION    #
#############################

# some pieces that might be good in a configuration file
# export images?
export_image_q = False
# save individual CSVs
#save_raw_output_q = True

# setup some shared fields
field_key_future = "future_id"
field_key_strategy = "strategy_id"
field_key_primary = "primary_id"
field_month = "month"
field_time_period = "time_period"
field_year = "year"

# model years
model_historical_years = list(range(2011, 2021))
model_projection_years = list(range(2021, 2056))

# number of lhs samples (good for a configuration file)?
n_lhs = 100

# set some climate info
range_delta_base = list(range(2011, 2021))
range_delta_fut = list(range(2046, 2056))

# strategies to run
strats_run = [0, 1, 2, 3, 5, 6]
#print(f"Running strategies {strats_run}")


# set a name for this run (good for a configuration file) 
time_stamp = sf.get_time_stamp()
analysis_name = f"crdm_project_{time_stamp}_{n_lhs}fut"


##  INIT DIRECTORIES

##  some directory stuff
dir_py = os.path.dirname(os.path.realpath(__file__)) # this finds the current directory and identifies dir_py as the python
dir_proj = os.path.dirname(dir_py)
dir_ed = os.path.join(dir_proj, "experimental_design")
dir_out = os.path.join(dir_proj, "out")
dir_ref = os.path.join(dir_proj, "ref")

# make directories if they don't exist
if not os.path.exists(dir_ed):
    os.makedirs(dir_ed, exist_ok = True)
    
if not os.path.exists(dir_out):
    os.makedirs(dir_out, exist_ok = True)
    
if not os.path.exists(dir_ref):
    print(f"WARNING: path {dir_ref} not found.")
    

##  SET SOME FILE PATHS

# names
fn_csv_all_output = "model_output.csv"
fn_csv_attribute_future_id = f"attribute_{field_key_future}.csv"
fn_csv_attribute_primary_id = f"attribute_{field_key_primary}.csv"
fn_csv_attribute_strategy_id = f"attribute_{field_key_strategy}.csv"
fn_csv_futures = "futures.csv"
fn_csv_metrics = "metrics_and_futures.csv"
fn_csv_strategies = "strategies.csv"

# paths
fp_csv_attribute_climate_id = os.path.join(dir_ref, "ri_attribute_climate_id.csv")
fp_csv_baseline_trajectory_model_input_data = os.path.join(dir_ref, "ri_baseline_trajectory_model_input_data.csv")
fp_csv_baseline_strategy_values = os.path.join(dir_out, "strategy_table_base.csv")
fp_csv_climate_deltas = os.path.join(dir_ref, "ri_climate_deltas.csv")
fp_csv_climate_deltas_annual = os.path.join(dir_ref, "ri_climate_deltas_annual.csv")
fp_csv_template_save_scenario = (
    os.path.join(dir_out, f"ri_model_output_{field_key_primary}-%s.csv")
    if True
    else None
)

fp_xlsx_strategy_inputs = os.path.join(dir_ref, "strategy_table_inputs.xlsx")



