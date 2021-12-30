def sorl_plot(run_log, timeslots_per_day, START_DAY=0, NO_OF_DAY_TO_PLOT = 500):
    
   # Get Environment Log
    ################################################################    

    reward_rec = run_log['reward_rec']
    sense_dc_log = run_log['sense_dc_log']
    env_log = run_log['env_log']
    
    henergy_obs_rec=env_log[:,1]
    penergy_obs_rec=env_log[:,2]
    benergy_obs_rec=env_log[:,3]
    menergy_obs_rec=env_log[:,4]
    req_obs_rec=env_log[:,5]
    

    NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
    NO_OF_DAY_TO_PLOT = int(min(NO_OF_DAY_TO_PLOT, len(henergy_obs_rec)/NO_OF_TIMESLOTS_PER_DAY))
    END_DAY = START_DAY + NO_OF_DAY_TO_PLOT

    start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
    end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY
    
    
    # Draw figure
    ##############
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=[20,4],
                            sharex=True)

    sense_dc_ax  = axs

    sense_dc_ax.grid(which='major', axis='x', linestyle='--')
    sense_dc_ax.set_ylim(-0.1,1.1)

    
    sense_dc_ax.plot(sense_dc_log[start_index:end_index], 
                     color='tab:blue', alpha=0.7,linewidth=1.0, label="sense_dc")
    sense_dc_ax.plot(req_obs_rec[start_index:end_index], 
                     color='tab:orange',linestyle='--',linewidth=1.0, label="req_dc")
    sense_dc_ax.plot(benergy_obs_rec[start_index:end_index], 
                     color='tab:red',linewidth=1.0, label="battery")
    sense_dc_ax.plot(menergy_obs_rec[start_index:end_index], 
                     color='tab:red',linestyle='--',linewidth=1.0, alpha=0.5,label="batt10d")
    sense_dc_ax.plot(reward_rec[start_index:end_index],
                     color='tab:purple', alpha=0.7,linewidth=1.0, label="reward")
#     sense_dc_ax.plot(henergy_obs_rec[start_index:end_index],
#                      color='k',linewidth=0.25,alpha=0.5, label="henergy")
#     sense_dc_ax.plot(penergy_obs_rec[start_index:end_index],
#                      color='k',linestyle='--',linewidth=0.25, label="penergy")

    
    sense_dc_ax.set_xlim([0,NO_OF_TIMESLOTS_PER_DAY*NO_OF_DAY_TO_PLOT])
    xtick_resolution = max(1,int(NO_OF_DAY_TO_PLOT/10))
    sense_dc_ax.set_xticks(np.arange(start=0,
                                      stop=NO_OF_TIMESLOTS_PER_DAY*(NO_OF_DAY_TO_PLOT+1),
                                      step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
    sense_dc_ax.set_xticklabels(np.arange(start=START_DAY,
                                           stop=END_DAY+1,
                                           step=xtick_resolution))
    sense_dc_ax.legend(loc="lower left",
                       ncol=7,
                       bbox_to_anchor=(0,1.0,1,1))
    
    sense_dc_ax.set_xlabel("Days")
    plt.show()
# End of sorl_plot
########################################################