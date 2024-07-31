import pandas as pd
import numpy as np
from typing import *

# mean groundwater metric
def get_mean_groundwater(
    df_cur: pd.DataFrame,
    field_key_primary: str = "primary_id",
    field_year: str = "year",
    last_n_years: int = 10
) -> pd.DataFrame:

    out = get_mean_value_over_period(
        df_cur,
        field_key_primary,
        "mean_groundwater_storage_last_ten_years_m3",
        "groundwater_storage_m3",
        field_year,
        last_n_years
    )
    
    return out




def get_mean_reservoir(
    df_cur: pd.DataFrame,
    field_key_primary: str = "primary_id",
    field_year: str = "year",
    last_n_years: int = 10
) -> pd.DataFrame:
    """
    Get the mean reservoir level
    """
    
    out = get_mean_value_over_period(
        df_cur,
        field_key_primary,
        "mean_reservoir_storage_last_ten_years_m3",
        "reservoir_storage_m3",
        field_year,
        last_n_years
    )
    
    return out



def get_mean_value_over_period(
    df_cur: pd.DataFrame,
    field_key_primary: str = "primary_id",
    field_metric: str = "mean_reservoir_storage_last_ten_years_m3",
    field_storage: str = "reservoir_storage_m3",
    field_year: str = "year",
    last_n_years: int = 10
) -> pd.DataFrame:
    """
    Get the mean reservoir or groundwater metric
    """
    y1 = max(df_cur[field_year])
    y0 = y1 - last_n_years + 1
    years_keep = range(y0, y1 + 1)

    mean_out = np.mean(df_cur[df_cur[field_year].isin(years_keep)][field_storage])
    df_metric = pd.DataFrame({field_key_primary: [int(df_cur[field_key_primary].loc[0])], field_metric: [mean_out]})

    return df_metric



def get_unacceptable_unmet_demand(
    df_cur: pd.DataFrame,
    field_key_primary: str = "primary_id",
    field_measure: str = "u_2_proportion",
    field_metric_exceed: str = "exceed_threshes",
    field_metric_prop: str = "proportion_unacceptable_unmet_demand",
    field_month: str = "month",
    field_year: str = "year",
    thresh_count: int = 4,
    thresh_demand: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    unacceptable unmet demand
    """
    
    vec_exceed_thresh_demand = [int(x > thresh_demand) for x in list(df_cur[field_measure])]
    
    field_pass_through = "exceed_thresh_demand"
    df_return = df_cur[[field_year, field_month, field_measure]].copy()
    df_return[field_pass_through] = vec_exceed_thresh_demand
    
    df_return = df_return.groupby([field_year]).agg({field_year: "first", field_pass_through: "sum"})
    df_return.reset_index(drop = True, inplace = True)
    
    vec_flag = [int(x >= thresh_count) for x in list(df_return[field_pass_through])]
    df_return[field_metric_exceed] = vec_flag
    
    metric_frac_vuln = np.sum(df_return[field_metric_exceed])/len(df_return)
    df_metric = pd.DataFrame({field_key_primary: [int(df_cur[field_key_primary].loc[0])], field_metric_prop: [metric_frac_vuln]})
    
    out = df_return[[field_year, field_metric_exceed]], df_metric

    return out
