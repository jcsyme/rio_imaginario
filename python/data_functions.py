import pandas as pd
import numpy as np
import pyDOE2 as pyd
from typing import *



def apply_delta_factor(
    df_climate_projection: pd.DataFrame, 
    delta: float,
    range_years_delta_base: set, 
    range_years_delta_fut: set, 
    year_cur: int,
    field_apply: str,
    field_delta_scale: str = "delta_scale",
    field_year: str = "year",
    scale_delta_for_inflection: bool = False
) -> pd.DataFrame:
    """
    Support function for get_climate_factor_deltas(). Apply the deltas to a
        baseline trajectory set. 
    """

    ##  AGGREGATE BY YEAR
    
    # setup aggregation for annual totals
    dict_agg = {field_year: "first", field_apply: "sum"}
    flds_agg = list(dict_agg.keys())
    # get annual totals
    df_annual_totals_delta_base = df_climate_projection[
        df_climate_projection[field_year].isin(range_years_delta_base)
    ][[field_year, field_apply]].copy().groupby([field_year]).aggregate(dict_agg).reset_index(drop = True)
    df_annual_totals_delta_mid = df_climate_projection[
        df_climate_projection[field_year].isin(range_years_delta_fut)
    ][[field_year, field_apply]].copy().groupby([field_year]).aggregate(dict_agg).reset_index(drop = True)
    
    
    ##  SETUP DELTAS
    
    # get current observed difference
    mean_base = np.mean(np.array(df_annual_totals_delta_base[field_apply]))
    mean_fut = np.mean(np.array(df_annual_totals_delta_mid[field_apply]))
    
    # get appropriate annual increases to applly
    annual_increase = get_annual_increase(
        delta, 
        mean_base, 
        mean_fut, 
        year_cur,
        range_years_delta_fut,
        df_annual_totals_delta_mid, 
        field_apply
    )
    
    yr_delta_0 = int(np.floor(np.mean(np.array(range_years_delta_base))))
    yr_delta_1 = int(np.floor(np.mean(np.array(range_years_delta_fut))))
    delta_apply = annual_increase*(yr_delta_1 - year_cur)

    #add shifts from deltas
    df_out = df_climate_projection.copy()
    df_out[field_delta_scale] = [
        1 + int(x > year_cur)*(x - year_cur)*delta_apply/(yr_delta_1 - year_cur) 
        for x in list(df_out[field_year])
    ]
    df_out[f"{field_apply}_w_delta"] = np.array(df_out[field_apply])*np.array(df_out[field_delta_scale])

    return df_out



def generate_lhs_samples(
    n: int, 
    dict_ranges: dict,
    dict_values_future_0: dict = None,
    field_future_id: str = "future_id",
    random_seed: Union[int, None] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate dataframe of all LHS trials that are adjusted. Returns a tuple of
        the form

        (df_lhs, df_lhs_transformed)

    Function Arguments
    ------------------
    - n: number of trials
    - dict_ranges: dictionary of form {variable: [min, max], ...} where variable 
        is the variable and min/max are the lower/upper bound scalars
    - dict_values_future_0: dictionary of form {variable: val, ...} where 
        variable is the variable and val gives the value to specify for future 
        0. If default of None is specifed, will specify 1 for all values.
    - field_future_id: field storing the future id, or LHS trial index
    - random_seed: random seed to use in generating LHC trials
    """
    
    all_variables = sorted(list(dict_ranges.keys()))
    dict_values_future_0 = dict(zip(all_variables, np.ones(len(all_variables)))) if (dict_values_future_0 is None) else dict_values_future_0 
    k = len(all_variables)
    mat_lhs = pyd.lhs(k, n, random_state = random_seed, )
    
    # transform the lhs trials to ranges using a min/max vector *in the same columnar order as all_variables
    vec_min = np.array([dict_ranges[x][0] for x in all_variables])
    vec_max = np.array([dict_ranges[x][1] for x in all_variables])
    mat_lhs_transformed = vec_min + mat_lhs*(vec_max - vec_min)
    base_values_f0 = np.array([[dict_values_future_0[x] for x in all_variables]])
    mat_lhs_transformed = np.concatenate([base_values_f0, mat_lhs_transformed], axis = 0)

    # now, create a data frame associated with each LHS trial and add a future id
    df_lhs = pd.DataFrame(mat_lhs, columns = all_variables)
    df_lhs[field_future_id] = range(1, n + 1)
    df_lhs = df_lhs[[field_future_id] + all_variables]

    df_lhs_transformed = pd.DataFrame(mat_lhs_transformed, columns = all_variables)
    df_lhs_transformed[field_future_id] = range(0, n + 1)
    df_lhs_transformed = df_lhs_transformed[[field_future_id] + all_variables]
    
    out = (df_lhs, df_lhs_transformed)

    return out



def get_annual_increase(
    delta: float, 
    mean_base: float, 
    mean_fut: float, 
    yr_base: int, 
    range_delta_fut: list,
    df_fut: pd.DataFrame,
    field_apply: str,
    field_year: str = "year"
) -> np.ndarray:
    """
    Calculate an annual increase to apply based on a delta factor
    
    Function Arguments
    ------------------
    - delta: the climate delta to apply (factor)
    - mean_base:  mean value of input time series during base time range used to 
        estimate delta
    - mean_fut:  mean value of input time series during future time range used 
        to estimate delta
    - range_delta_fut: list of years defining the time period for future delta
    - df_fut: data frame giving input time series
    - field_apply: field in df_fut to apply delta to
    - field_year: field in df_fut denoting the year
    """
    
    n = len(range_delta_fut)
    num = n*(mean_base*(1 + delta) - mean_fut)
    den = np.dot((np.array(df_fut[field_year]) - yr_base), np.array(df_fut[field_apply]))
    
    return num/den



def get_climate_factor_deltas(
    df_base_climate_trajectory: pd.DataFrame,
    df_climate_deltas_annual: pd.DataFrame,
    dict_field_traj_to_field_delta: dict,
    years_delta_base: list,# sa.range_delta_base, 
    years_delta_fut: list,#       sa.range_delta_fut, 
    year_base_uncertainty: int, #max(sa.model_historical_years)
    drop_climate_delta_duplicate_keys: bool = True,
    field_future_id: str = "future_id",
    fields_date: list = ["year", "month"],
    field_append_w_delta: str = "w_delta",
) -> pd.DataFrame:
    """
    Apply climate deltas to a base trajectory. Returns a data frame of 
        trajectories modified to reflect climate deltas.

    - df_base_climate_trajectory: data frame with base climate trajectories to 
        modify with deltas
    - df_climate_deltas_annual: data frame with deltas to apply to base climate 
        trajectories, indexed by key 'field_future_id'
    - dict_field_traj_to_field_delta: dictionary of form 
        {field_traj: field_delta, ...} where field_traj is a field in 
        df_base_climate_trajectory to be modified and field_delta is a field in 
        df_annual_deltas to use to find the delta 
    - field_future_id: key value in df_annual_deltas to use to loop to apply 
        deltas
    - years_delta_base: list of years (integers) that the delta changes from
    - years_delta_fut: list of years (integers) used to calcualte the delta 
        target as change from base
    - year_base: base year, or last year before uncertainty begins
    - drop_climate_delta_duplicate_keys: if the climate data frame contains 
        multiple instances of the key, drop duplicate rows? If True, keeps first 
        instance by default. 
    - field_future_id: default 'future_id'. Must be contained in 
        df_climate_deltas_annual. Used to determine scenario key values to loop 
        over.
    - fields_date: default ["year", "month"]. Fields necessary to define dates.
         Must include year.
    """

    # initialiez some important pieces
    all_key_values = set(df_climate_deltas_annual[field_future_id])
    if (len(df_climate_deltas_annual) != len(all_key_values)) & drop_climate_delta_duplicate_keys:
        df_climate_deltas_annual.drop_duplicates(subset = [field_future_id], keep = "first")
    all_key_values = sorted(list(all_key_values))
    fields_traj = list(dict_field_traj_to_field_delta.keys())
    
    # intiialize the output database and add climate id columns
    df_tmp = df_base_climate_trajectory[fields_date + fields_traj].copy()
    df_tmp[field_future_id] = 0
    df_tmp = df_tmp[[field_future_id] + fields_date + fields_traj]
    df_out = [df_tmp for x in range(len(all_key_values))]

    for future_id in enumerate(all_key_values):

        ind, future_id = future_id

        df_delta_cur = [df_base_climate_trajectory[fields_date]]

        for field in fields_traj:
            field_delta = dict_field_traj_to_field_delta[field]
            delta_climate = float(df_climate_deltas_annual[df_climate_deltas_annual[field_future_id] == future_id][field_delta])
    
            # calculate delta 
            df_delta = apply_delta_factor(
                df_base_climate_trajectory[fields_date + [field]], 
                delta_climate,
                years_delta_base, 
                years_delta_fut, 
                year_base_uncertainty,
                field
            )
            
            df_delta = df_delta[fields_date + [f"{field}_{field_append_w_delta}"]]
            df_delta.rename(columns = {f"{field}_{field_append_w_delta}": field}, inplace = True)
            df_delta_cur.append(df_delta[[field]])
            
        # no need to merge, so we'll do a horizontal concatenation
        df_delta_cur = pd.concat(df_delta_cur, axis = 1).reset_index(drop = True)
        df_delta_cur[field_future_id] = future_id
        df_out[ind] = df_delta_cur[df_out[0].columns]

    df_out = pd.concat(df_out, axis = 0).reset_index(drop = True)
    
    return df_out



def get_linear_delta_trajectories_by_future(
    df_base_trajectory: pd.DataFrame,
    df_lhs: pd.DataFrame,
    t0_vary: int, 
    t1_vary: int,
    field_future_id: str = "future_id",
    field_time_period: str = "time_period",
) -> pd.DataFrame:
    """
    Apply deltas to a base trajectory. Returns a data frame of trajectories 
        modified to reflect climate deltas.

    Function Arguments
    ------------------
    - df_base_trajectory: data frame with base trajectories to modify with 
        deltas
    - df_lhs: data frame with scalars to apply to base trajectories by last 
        time period, indexed by key 'field_future_id'
    - t0_vary: last time period without uncertainty
    - t1_vary: final time period of uncertainty

    Keyword Arguments
    -----------------
    - field_future_id: default 'future_id'. Must be contained in df_lhs. Used to 
        determine scenario key values to loop over.
    - field_time_period: default is 'time_period'. Used for determining scaling 
        of deltas
    """
    
    # initilize key variables
    all_futures = sorted(list(set(df_lhs[field_future_id])))
    all_time_periods = sorted(df_base_trajectory[field_time_period])
    all_variables = sorted(list(set(df_lhs.columns) & set(df_base_trajectory.columns)))
    mat_nominal_traj = np.array(df_base_trajectory[all_variables])
    k = len(all_variables)
    n = len(all_futures)
    T = len(all_time_periods)
    vec_nom = mat_nominal_traj[-1]

    # for this example, inter-annual deltas will be applied linearly, scaling from y0_vary to y1_vary
    vec_del_template = np.array([max((t - t0_vary)/(t1_vary - t0_vary), 0) for t in all_time_periods])

    # initialize output database and component vectors
    df_db_out = []
    df_db = [mat_nominal_traj for x in range(n)]
    vec_fut = np.concatenate([x*np.ones(len(all_time_periods)) for x in range(0, n)])
    vec_time_periods = np.concatenate([all_time_periods for x in range(0, n)])
    
    # initialize
    mat_tmp = np.array(df_lhs.sort_values(by = [field_future_id])[all_variables]) - 1
    
    # use the outer product to build all potential deltas with the del_template ramp in place 
    mat_expand_traj = np.outer(mat_tmp, vec_del_template)
    mat_expand_traj = mat_expand_traj.reshape(n, k, T)
    
    # there is a faster way, but loop over each futre to build adjustments
    for f in enumerate(all_futures):
        ind, f = f
        
        # get the lhs-transformed component to multiply by the nominal value in the last year
        mat_add = mat_expand_traj[ind].transpose() * vec_nom
        mat_new = mat_add + mat_nominal_traj
        
        # we know that the futures align because both df_lhs and all_variabels are sorted by future id
        df_db[ind] = mat_new

    # finally, convert to a data frame, add columns, and reorder
    df_db = pd.DataFrame(np.concatenate(df_db), columns = all_variables)
    df_db[field_future_id] = vec_fut.astype(int)
    df_db[field_time_period] = vec_time_periods.astype(int)

    return df_db



def get_strategy_table(
    fp_xlsx_strategies: str,
    field_strategy_id: str = "strategy_id",
    fields_sort_additional: list = ["time_period"],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read an XLSX file containing strategies in each sheet. If the field 
        field_strategy_id is not specified in each sheet, it will be inferred.
    
    Function Arguments
    ------------------
    - fp_xlsx_strategies: file path to Excel workbooks

    Keyword Arguments
    -----------------
    - field_strategy_id: field specifying the strategy id
    - fields_sort_additional: additional fields to sort by after 
        field_strategy_id
    """
    
    # read in data from file
    dfs_read = pd.read_excel(fp_xlsx_strategies, sheet_name = None)
    
    # check for strategy field and determine if it needs to be inferred
    defined_strats = set({})
    infer_strategies = False
    init_header = True
    
    ##  RUN SOME CHECKS
    
    # loop over all specified dataframes
    for df in dfs_read.keys():
        
        # check strategies
        if field_strategy_id not in dfs_read[df].columns:
            infer_strategies = True
        elif len(set(dfs_read[df][field_strategy_id])) != 1:
            infer_strategies = True
        else:
            defined_strats = defined_strats | set(dfs_read[df][field_strategy_id])
            
        # get universe of headers and turn off
        header = list(dfs_read[df].columns) if init_header else [x for x in dfs_read[df].columns if x in header]
        init_header = False if init_header else init_header
        
    # if there are repeat strategies, we will infer
    if len(defined_strats) != len(dfs_read):
        infer_strategies = True
        
        
    ##  NOW BUILD STRATEGY ATTRIBTUE TABLE AND STRATEGY DATASET
    
    df_strats = []
    vec_strategy_id = []
    vec_strategy_name = []
    strat_names = sorted(list(dfs_read.keys()))
    if field_strategy_id not in header:
        header.prepend(field_strategy_id)
    
    for i in enumerate(strat_names):
        i, strat_name = i
        df_cur = dfs_read[strat_name]
        
        # get strategy information and update for attribute table
        strategy_id = i if infer_strategies else int(df_cur[field_strategy_id].loc[0])
        df_cur[field_strategy_id] = strategy_id
        vec_strategy_id.append(strategy_id)
        vec_strategy_name.append(strat_name)
        
        # add to out put
        if len(df_strats) == 0:
            df_strats = [df_cur[header] for x in strat_names]
        else:
            df_strats[i] = df_cur[header]
    
    # collapse df_strats and convert vecs to dataframe
    fields_sort_additional = [field_strategy_id] + [x for x in fields_sort_additional if x in header]
    df_strats = pd.concat(df_strats, axis = 0).sort_values(by = fields_sort_additional).reset_index(drop = True)
    df_attribute = pd.DataFrame({field_strategy_id: vec_strategy_id, "strategy_name": vec_strategy_name}).sort_values(by = [field_strategy_id]).reset_index(drop = True)
    
    return df_attribute, df_strats