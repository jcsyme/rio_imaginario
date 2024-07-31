# PLOT OUTPUTS 

output_plot = "u_2_m3"

all_strategies = sorted(list(df_attribute_strategy[sa.field_key_strategy].unique()))

strategy_plot = 6
fig, ax = plt.subplots(figsize = (18, 12))

for strat in [strategy_plot]:
    
    primaries = get_primary_keys_from_strat(strat)
    if len(primaries) == 0:
        print(f"No information found for strategy {strat}")
        continue
        
    df_filt = df_output[
        df_output[sa.field_key_primary].isin(primaries)
    ]
    
    dfg = df_filt.groupby([sa.field_key_primary])
    
    for i, df in dfg:
        y = np.array(df[output_plot]) 
        x = range(len(y))

        ax.plot(
            x, 
            y, 
            alpha = 0.5,
        )

# add some properties
ax.set_xlabel(sa.field_time_period)
ax.set_ylabel(output_plot)
