import math
import numpy as np
import pandas as pd
import scipy.optimize as sco
import support_functions as sf
import time
from typing import *


##  SET SOME DEFAULTS

# dictionary of default initial states
md_dict_initial_states = {
    "reservoir_storage_million_m3": 150, 
    "groundwater_storage_million_m3": 14000
}

# dictionary of default parameter values
md_dict_parameters = {
    "area_catchment_km2": 4000,
    "costs_unmet_demand": [5, 1000, 1000],
    "groundwater_inflow_m3s": 5,
    "maximum_gw_discharge_m3s": 12.5,
    "maximum_reservoir_outflow_m3s": 25,
    "proportion_gw_discharge": 0.015,
    "proportion_precip_runoff": 0.30,
    "proportion_precip_infiltration": 0.70,
    "proportion_agricultural_water_runoff": 0.5
}

# dictionary of default lever states
md_dict_default_levers = {
    "capacity_reservoir_million_m3": 300,
    "increase_ag_efficiency_rate": 0,
    "increase_mun_efficiency_rate": 0,
    "transmission_gw_ag_m3s": np.nan,
    "transmission_gw_mun_m3s": 2.5,
    "transmission_res_ag_m3s": 10,
    "transmission_res_mun_m3s": 10,
    "recylcing_proportion_mun": 0.2,
    "wastewater_treatment_capacity_m3s": 0.5
}




class RIWaterResourcesModel:
    """
    Create a water resources model for use in projecting different systemic 
        components. 
        
    Optional Arguments
    ------------------
    - dict_default_levers: dictionary of default lever parameter values. 
        Values specifie the value of the lever in 2050. Must include the 
        following keys:
        * "capacity_reservoir_million_m3": 300,
        * "increase_ag_efficiency_rate": 0,
        * "increase_mun_efficiency_rate": 0,
        * "transmission_gw_ag_m3s": np.nan,
        * "transmission_gw_mun_m3s": 2.5,
        * "transmission_res_ag_m3s": 10,
        * "transmission_res_mun_m3s": 10,
        * "recylcing_proportion_mun": 0.2,
        * "wastewater_treatment_capacity_m3s": 0.5
    - dict_initial_states: dictionary of initial states for the Rio Imaginario
        reservoir and groundwater storage. Must contain the following keys:
        * "reservoir_storage_million_m3"
        * "groundwater_storage_million_m3"
    - md_dict_parameters: dictionary of parameters. Must contain the following 
        keys:
        * "area_catchment_km2"
        * "costs_unmet_demand"
        * "groundwater_inflow_m3s"
        * "maximum_gw_discharge_m3s"
        * "maximum_reservoir_outflow_m3s"
        * "proportion_gw_discharge"
        * "proportion_precip_runoff"
        * "proportion_precip_infiltration"
        * "proportion_agricultural_water_runoff"
    """


    def __init__(self,
        dict_default_levers: Union[Dict[str, Union[int, float]], None] = None,
        dict_initial_states: Union[Dict[str, Union[int, float]], None] = None,
        dict_parameters: Union[Dict[str, Union[int, float]], None] = None,
    ) -> None:

        self._initialize_state_dictionaries(
            dict_default_levers,
            dict_initial_states,
            dict_parameters,
        )



    def __call__(self,
        *args,
        **kwargs,
    ) -> Any:

        out = self.project(*args, **kwargs)

        return out


    ##################################
    #    INITIALIZATION FUNCTIONS    #
    ##################################

    def _initialize_state_dictionaries(self,
        dict_default_levers: Union[dict, None] = None,
        dict_initial_states: Union[dict, None] = None,
        dict_parameters: Union[dict, None] = None,
    ) -> None:
        """
        Set model default (md_) dictionary values
        """

        # check initial states
        dict_default_levers = (
            md_dict_default_levers
            if not isinstance(dict_default_levers, dict)
            else dict_default_levers
        )

        dict_initial_states = (
            md_dict_initial_states
            if not isinstance(dict_initial_states, dict)
            else dict_initial_states
        )

        dict_parameters = (
            md_dict_parameters
            if not isinstance(dict_parameters, dict)
            else dict_parameters
        )


        ##  SET PARAMETERS

        self.dict_default_levers = dict_default_levers
        self.dict_initial_states = dict_initial_states
        self.dict_parameters = dict_parameters

        return None





    ##########################
    #    MODEL PROJECTION    #
    ##########################

    def project(self,
        df_in: pd.DataFrame, 
        dict_default_levers: Union[dict, None] = None, 
        dict_initial_states: Union[dict, None] = None, 
        dict_parameters: Union[dict, None] = None, 
        id_primary: Union[int, None] = None,
    ) -> Union[pd.DataFrame, Tuple[int, pd.DataFrame]]:
        """
        Project the water resources model over the DataFrame df_in. Returns a
            DataFrame of output metrics. Returns one of two options:


            df_out                  # IF id_primary is None (a data frame of 
                                        model outputs)
            (id_primary, df_out)    # tuple with primary index passed as first
                                        element + output data frame as second.
                                        Only applies if id_primary is not None

        Function Arguments
        ------------------
        - df_in: input dataframe containing model inputs

        Keyword Arguments
        -----------------
        - dict_default_levers: default lever values by final time period. NOTE:
            levers can be incorporated using df_in as well. If None, defaults to
            self.dict_default_levers
        - dict_initial_states: initial states for the model. If None, defaults
            to self.dict_initial_states
        - dict_parameters: dictionary of model parameters. If None, defaults to
            self.dict_parameters
        - id_primary: optional id to pass to the model. If this value is not
            None, then the function returns a tuple of the following form

            (
                id_primary,
                df_out
            )
        """

        ##  CHECK DICTIONARIES

        dict_default_levers = (
            self.dict_default_levers
            if not isinstance(dict_default_levers, dict)
            else dict_default_levers
        )

        dict_initial_states = (
            self.dict_initial_states
            if not isinstance(dict_initial_states, dict)
            else dict_initial_states
        )

        dict_parameters = (
            self.dict_parameters
            if not isinstance(dict_parameters, dict)
            else dict_parameters
        )


        ##  INITIALIZE PARAMETERS
        
        # get time steps (ensure this is run ONLY for a single scenario)
        df_in = df_in.sort_values(by = ["year", "month"]).reset_index(drop = True)
        time_steps = range(0, len(df_in))
        n_t = len(time_steps)
        ym_tup_0 = (int(df_in["year"].iloc[0]), int(df_in["month"].iloc[0]))


        # setup indices
        inds_supply = [1, 2]
        inds_demand = [1, 2, 3]
        
        # stream flow requirements
        inds_demand_sfr = [3]
        inds_demand_no_sfr = [x for x in inds_demand if (x not in inds_demand_sfr)]

        # get initial states
        res_0 = max(dict_initial_states["reservoir_storage_million_m3"], 0)*1000000
        gw_0 = max(dict_initial_states["groundwater_storage_million_m3"], 0)*1000000

        
        # get some parameter values
        param_area_precip_catchment = max(dict_parameters["area_catchment_km2"], 0)
        param_delta_discharge_prop = max(dict_parameters["proportion_gw_discharge"], 0)
        param_gw_inflow_cross_basin = max(dict_parameters["groundwater_inflow_m3s"], 0)
        param_max_groundwater_discharge = max(dict_parameters["maximum_gw_discharge_m3s"], 0)
        param_max_reservoir_outflow = max(dict_parameters["maximum_reservoir_outflow_m3s"], 0)
        param_omega_runoff_prop = max(dict_parameters["proportion_precip_runoff"], 0)
        param_rho_gw_infiltration_prop = max(
            dict_parameters["proportion_precip_infiltration"], 0
        )
        param_runoff_ag_prop = max(dict_parameters["proportion_agricultural_water_runoff"], 0)
        
        # check and scale if there's an issue with summing over 1
        sc_precip_prop = param_rho_gw_infiltration_prop + param_omega_runoff_prop
        if sc_precip_prop > 1:
            param_rho_gw_infiltration_prop = param_rho_gw_infiltration_prop/sc_precip_prop
            param_omega_runoff_prop = param_omega_runoff_prop/sc_precip_prop

        # get precip means and lookback values
        df_precip_means = (
            df_in[["month", "precipitation_mm"]]
            .groupby(["month"])
            .agg({"month": "first", "precipitation_mm": "mean"})
            .reset_index(drop = True)
        )
        m_p1 = sf.date_shift(ym_tup_0, -1)[1]
        m_p2 = sf.date_shift(ym_tup_0, -2)[1]

        # previous precip based on averages
        p_lb1 = float(df_precip_means[df_precip_means["month"] == m_p1]["precipitation_mm"].iloc[0])
        p_lb2 = float(df_precip_means[df_precip_means["month"] == m_p2]["precipitation_mm"].iloc[0])

        # get days per month vector
        vec_dpm = np.array(
            [
                sf.num_days_per_month(tuple(x)) for x in np.array(df_in[["year", "month"]])
            ]
        )


        ##  INITIALIZE OUTPUT TRACKING

        # basic zero initialization for output values (overwrite in loop)
        vec_0 = np.zeros(n_t)

        # variables to track
        vars_transmission = ["x_%s%s_m3"%(i,j) for i in inds_supply for j in inds_demand_no_sfr]
        vars_demand = ["d_%s_m3"%(j) for j in inds_demand]
        vars_supplied = ["s_%s_m3"%(j) for j in inds_demand]
        vars_unmet_demand = ["u_%s_m3"%(j) for j in inds_demand]
        vars_unmet_demand_proportion = ["u_%s_proportion"%(j) for j in inds_demand]
        vars_release = ["r_m3"]
        vars_return = ["f_%s_m3"%(j) for j in inds_demand_no_sfr]
        vars_storage = ["groundwater_storage_m3", "reservoir_storage_m3", "reservoir_release_m3", "reservoir_spillage_m3"]
        vars_other_transmission = ["gw_discharge_m3", "gw_recharge_m3", "precip_runoff_m3"]

        # set the headers
        header_out = vars_transmission + vars_demand + vars_supplied + vars_unmet_demand + vars_unmet_demand_proportion + vars_release + vars_return + vars_storage + vars_other_transmission
        # initialize dictionary of indices
        dict_running_var_indices = {}
        for k in vars_storage:
            dict_running_var_indices.update({k: header_out.index(k)})



        # all results to track
        array_vars_out = np.zeros((len(df_in), len(header_out)))



        ##  GET SOME TIME SERIES INPUT DATA

        # levers, which can be added via df_in, but default to a single value
        dict_levers = {}
        for k in list(dict_default_levers.keys()):
            # crude units check
            if "million_m3" in k:
                scalar = 1000000
            elif "m3s" in k:
                scalar = vec_dpm*86400
            else:
                scalar = 1

            if k in df_in.columns:
                # note: can multiply by a scalar or vector of equal dimension since it's an np.array
                dict_levers[k] = np.array(df_in[k])*scalar
            else:
                dict_levers[k] = dict_default_levers[k]*scalar*np.ones(n_t)
                
                
        # inputs modified by levers (where applicable)
        vec_precip = np.array(df_in["precipitation_mm"])
        vec_flow = np.array(df_in["flow_m3s"])
        
        area_ag = np.array(df_in["area_ag_km2"])
        vec_eff_ag = np.array([max(min(x, 0.25), 0) for x in dict_levers["increase_ag_efficiency_rate"]])
        vec_eff_mun = np.array([max(min(x, 0.15), 0) for x in dict_levers["increase_mun_efficiency_rate"]])
        vec_demand_agricultural = np.array(df_in["demand_agricultural_m3km2"])*area_ag*(1 - vec_eff_ag)
        vec_demand_municipal = np.array(df_in["demand_municipal_m3p"])*np.array(df_in["population"])*(1 - vec_eff_mun)
        

        t0 = time.time()
        for t in time_steps:

            #
            # variable order in matrices:
            # [x_11, x_12, x_21, x_22, u_1, u_2, u_3, r, f_1, f_2, f_2_aux]
            #

            ##  set the proportion of the catchment upstream of the reservoir
            prop_catch_up = 0.25


            ##  INPUT VARIABLES (everything in terms of m3)

            # specificed values w/levers
            p = vec_precip[t]
            q = vec_flow[t]*vec_dpm[t]*86400
            pvol_ag = p*float(df_in["area_ag_km2"].iloc[t])*1000*(1 - param_runoff_ag_prop)
            d_1 = max(vec_demand_agricultural[t] - pvol_ag, 0)
            
            phi_2 = max(min(dict_levers["recylcing_proportion_mun"][t], 0.5), 0)
            d_2 = vec_demand_municipal[t]/(1 + phi_2)


            # flow constraints (m3/month)
            fc_11 = dict_levers["transmission_gw_ag_m3s"][t]
            fc_12 = dict_levers["transmission_gw_mun_m3s"][t]
            fc_21 = dict_levers["transmission_res_ag_m3s"][t]
            fc_22 = dict_levers["transmission_res_mun_m3s"][t]
            # reservoir outflow
            fc_res_out = param_max_reservoir_outflow*vec_dpm[t]*86400

            # some scalars — subtract ag area to avoid double counting
            a = (param_area_precip_catchment - area_ag[t])*1000# can be multipled by precip to give m3 as unit



            ##  SET SOME CONSTRAINT MAXIMA

            # lookbacks for storage and precipitation
            if t == 0:
                p_prev_1 = p_lb1
                p_prev_2 = p_lb2
                s1_prev = gw_0
                s2_prev = res_0
            else:
                p_prev_1 = vec_precip[t - 1]
                if t == 1:
                    p_prev_2 = p_lb1
                else:
                    p_prev_2 = vec_precip[t - 2]

                # note: during modeling loop, *everything* is in m3, so ignore the million m3 subscript for now
                s1_prev = array_vars_out[t - 1, dict_running_var_indices["groundwater_storage_m3"]] 
                s2_prev = array_vars_out[t - 1, dict_running_var_indices["reservoir_storage_m3"]]#$dict_out["groundwater_storage_million_m3"][t - 1]


            ##  groundwater storage
            recharge_gw = a*param_rho_gw_infiltration_prop*(p_prev_1/3 + 2*p_prev_2/3)
            inflow_cross_basin = param_gw_inflow_cross_basin*86400*vec_dpm[t]
            # calculate discharge
            discharge_gw = min(param_max_groundwater_discharge*vec_dpm[t]*86400, s1_prev*param_delta_discharge_prop)

            s1_hat = s1_prev - discharge_gw + recharge_gw + inflow_cross_basin
            # set constraint
            const_gw = s1_hat


            ##  precipitation runoff — 50% above reservoir, 50% below
            runoff_precip = p*a*param_omega_runoff_prop

            ##  reservoir storage
            s2_hat = s2_prev + q + runoff_precip*prop_catch_up
            # account for loss from the reservoir (excluding ET)
            res_seepage = s2_hat * 0.025
            # reservoir constraint value
            const_res = s2_hat - res_seepage
            # reservoir storage capocity
            capac_res = dict_levers["capacity_reservoir_million_m3"][t]
            const_capac_res = capac_res - const_res


            ##  downstream demand

            # downstream demand - 70% of inflows
            dem_downstream = 100000000#0.7*q
            # release i
            const_downstream = discharge_gw + runoff_precip*(1 - prop_catch_up) - dem_downstream


            ##  wastewater treatment at municipal levelw

            const_ww = dict_levers["wastewater_treatment_capacity_m3s"][t]



            ##  CONSTRAINT MATRICES

            # set positive restriction
            submat_aleq_posvars = -np.identity(10).astype(int)
            # set coefficients in A
            A_leq = np.array([
                # groundwater supply
                [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                # reservoir supply
                [0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0],
                # reservoir capacity
                [0, 0, -1, -1, 0, 0, 0, -1, 0, 0, 0],
                # downstream demand (unmet demand plus the release)
                [0, 0, 0, 0, 0, 0, -1, -1, -1, -1, 0],
                # constraint on wastewater return to river from municipal source
                [0, -1, 0, -1, 0, 0, 0, 0, 0, 1, 0]
            ])
            # set leq constraints
            b_leq = np.array([
                const_gw,
                const_res,
                const_capac_res,
                const_downstream,
                0
            ])


            # set the outflow limit
            if const_res <= 0.3*capac_res:
                res_out_ub = fc_res_out/10
                # buffer
            elif const_res <= capac_res:
                res_out_ub = fc_res_out
            else:
                res_out_ub = const_res#None

            # set the variable bounds
            bounds = [(0, fc_11), (0, fc_12), (0, fc_21), (0, fc_22), (0, d_1), (0, d_2), (0, dem_downstream), (0, res_out_ub), (0, d_1*param_runoff_ag_prop), (0, const_ww), (0, None)]

            ##  add in some equality constraints

            # equality constraints
            A_eq = np.array([
                # agricultural demand
                [1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0],
                # municipal demand
                [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                # agricultural runoff
                [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                # municipal return flow
                [0, 1, 0, 1, 0, 0, 0, 0, 0, -1, -1]
            ])
            # b matrix
            b_eq = np.array([d_1, d_2, pvol_ag*param_runoff_ag_prop, 0])


            # vector c — minimize unmet demand, but cost it. Unmet demand at the munic
            costs = dict_parameters["costs_unmet_demand"]
            c = np.array([0, 0, 0, 0] + list(dict_parameters["costs_unmet_demand"]) + [0, 0, 0, 1])
            # get results
            res = sco.linprog(c, A_leq, b_leq, A_eq, b_eq, bounds = bounds, method = "revised simplex")
            x = res["x"]

            # update output vector
            vec_out_demand = np.array([
                d_1, 
                d_2, 
                dem_downstream
            ] )
            vec_out_supplied = np.array([
                x[0] + x[2], 
                x[1] + x[3], 
                np.dot(x[6:10], np.array([1, 1, param_runoff_ag_prop, 1]))
            ])

            vec_out_unmet = x[4:7]
            vec_out_unmet_prop = vec_out_unmet/vec_out_demand
            vec_out_other_transmission = np.array([discharge_gw, recharge_gw, runoff_precip])


            # get current storage
            s1_cur = s1_hat - x[0] - x[1]
            s2_cur = const_res - x[2] - x[3] - x[7]
            # reservoir release
            release = min(fc_res_out, x[7])
            spill = max(0, x[7] - release)

            # for the RDM effect!
            #time.sleep(0.01)
            # new row
            out_vec = np.concatenate([
                x[0:4], 
                vec_out_demand, 
                vec_out_supplied,
                vec_out_unmet,
                vec_out_unmet_prop,
                x[7:10],
                np.array([s1_cur, s2_cur, release, spill]),
                vec_out_other_transmission
            ])
            # add to output array
            array_vars_out[t] = out_vec
            
        # add date information
        df_out = pd.DataFrame(array_vars_out, columns = header_out)
        df_out = pd.concat([df_in[["year", "month"]], df_out], axis = 1)
        t1 = time.time()  
        #print("RI model done in %s seconds."%(round(t1 - t0, 2)))
        
        if id_primary is not None:
            return id_primary, df_out

        return df_out

        
            
            
            
        
        
        
        



