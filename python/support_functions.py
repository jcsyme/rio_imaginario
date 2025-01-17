import datetime as dt
import math
import numpy as np
import pandas as pd
from typing import *



def build_dict(
    df_in: pd.DataFrame,
) -> Union[dict, None]:
    """
    Build a dictionary from two columns
    """
    
    # verify input type
    if not isinstance(df_in, pd.DataFrame):
        return None
    
    
    out = dict(
        x for x in zip(df_in.iloc[:, 0], df_in.iloc[:, 1])
    )
    
    return out



def date_shift(
    ym_tup: Tuple[int, int], 
    n_months: int,
) -> Tuple[int, int]:
    """
    From an input tuple (y, m), shift the date by n_months to return a new date
        tuple.
    """
    y = ym_tup[0]
    m = ym_tup[1]
    
    y_0 = y + (m - 1)/12
    y_frac = n_months/12
    y_1 = math.floor(y_0 + y_frac)
    
    m_1 = round((y_frac + y_0 - y_1)*12) + 1

    out = (y_1, m_1)

    return out



def get_time_stamp(
    time_now: Union[dt.datetime, None] = None,
) -> str:
    """
    Based on the current `time` (datetime obj), create a time stamp
        for use in analytical names.
    """
    
    # check time
    time_now = (
        time_now
        if isinstance(time_now, dt.datetime)
        else dt.datetime.now()
    )
    
    # convert to string and clean
    str_out = time_now.isoformat()
    str_out = (
        str_out
        .replace(":", "")
        .replace("-", "")
        .replace(".", "")
    )
    
    return str_out



def num_days_per_month(
    ym_tup: Tuple,
) -> Union[int, None]:
    """
    From a tuple of ym_tuple = (y, m), return the number of days in that month.
    """
    year = ym_tup[0]
    month = ym_tup[1]
    
    dict_base = {
        1: 31,
        2: 28,
        3: 31,
        4: 30,
        5: 31, 
        6: 30, 
        7: 31, 
        8: 31,
        9: 30,
        10: 31, 
        11: 30, 
        12: 31
    }
    
    (
        dict_base.update({2: 29})
        if year%4 == 0
        else None
    )
    
    out = dict_base.get(month)

    return out



def subset_df(
    df: pd.DataFrame,
    dict_as_exclusionary: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """
    Subset a dataframe using values associated with fields, passed in a 
        filtering dictionary

    Function Arguments
    ------------------
    - df: data frame to reduce

    Keyword Arguments
    -----------------
    - dict_as_exclusionary: set to True to *exclude* values passed in the 
        dictionary
    - **kwargs: specify keywords used to reduce df that takes the following 
        form:

        
        field_a = [v_a1, v_a2, v_a3, ... v_an],
        field_b = v_b,
        .
        .
        .

        where `field_a` and `field_b` are fields in the data frame and

            [v_a1, v_a2, v_a3, ... v_an]

        is a list of acceptable values to filter on, and

            v_b

        is a single acceptable value for field_b.
    """

    for k, v in kwargs.items():
        if k not in df.columns:
            continue
    
        val = [v] if not isinstance(v, list) else v
        df = (
            df[df[k].isin(val)]
            if not dict_as_exclusionary
            else df[~df[k].isin(val)]
        )

    df.reset_index(drop = True, inplace = True)

    return df
