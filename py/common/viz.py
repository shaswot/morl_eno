import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
# plt.rc('xtick', labelsize=8)
# plt.rc('ytick', labelsize=8)
# plt.rc('axes', labelsize=8)

import sys
import pathlib

# CONSTANT DICTIONARY
env_log_metric_dict = {
                        "time":0,
                        "henergy": 1,
                        "penergy": 2,
                        "benergy": 3,
                        "menergy": 4,
                        "req_obs": 5
                    }

def sorl_plot(run_log, 
              timeslots_per_day, 
              START_DAY=0, 
              NO_OF_DAY_TO_PLOT = 500,
              show_reward=True,
              show_henergy=False):
    
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
    
    time_trace = np.arange(timeslots_per_day*NO_OF_DAY_TO_PLOT)
    # Draw figure
    ##############
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=[20,4],
                            sharex=True)

    sense_dc_ax  = axs

    sense_dc_ax.grid(which='major', axis='x', linestyle='--')
    sense_dc_ax.set_ylim(-0.1,1.1)

    
    sense_dc_ax.step(time_trace,
                     sense_dc_log[start_index:end_index],
                     where='post',
                     color='tab:blue', alpha=0.7,linewidth=1.0, label="sense_dc")
    sense_dc_ax.step(time_trace,
                     req_obs_rec[start_index:end_index],
                     where='post',
                     color='tab:orange',linestyle='--',linewidth=1.0, label="req_dc")
    sense_dc_ax.step(time_trace,
                     benergy_obs_rec[start_index:end_index],
                     where='post',
                     color='tab:red',linewidth=1.0, label="battery")
    sense_dc_ax.step(time_trace,
                     menergy_obs_rec[start_index:end_index],
                     where='post',
                     color='tab:red',linestyle='--',linewidth=1.0, alpha=0.5,label="batt10d")
    if show_reward:
        sense_dc_ax.step(time_trace,
                         reward_rec[start_index:end_index],
                         where='post',
                         color='tab:purple', alpha=0.7,linewidth=1.0, label="reward")
    if show_henergy:
        sense_dc_ax.step(time_trace,
                         henergy_obs_rec[start_index:end_index],
                         where='post',
                         color='k',linewidth=0.25,alpha=0.5, label="henergy")
#     sense_dc_ax.step(penergy_obs_rec[start_index:end_index],
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
    plt.close()
    return fig
    # plt.show()
# End of sorl_plot
########################################################

def compare_trace(results_folder, # the folder holding results of all experiments
                  experiment_list,
                  label_list,
                  metric_list,
                  seed_no,
                  mode,
                  location,
                  year,
                  START_DAY=0,
                  NO_OF_DAY_TO_PLOT = 500):

    # generate experiment names and colors and linestyles
    experiment_label = dict(zip(experiment_list, label_list))

    # specify a color for each plot
    color_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
    experiment_color = dict(zip(experiment_list, color_list))

    # specify different linestyles for different metrics
    linestyle_list = ['solid', 'dotted', 'dashed', 'dashdot']
    metric_linestyle = dict(zip(metric_list, linestyle_list))
    
    # load the traces
    # dictionary to hold experimental data
    experiment_traces = {}

    # load data from each experiment
    timeslotsperday = None
    for experiment in experiment_list:
        # Load data of experiment and store in a dictionary
        experiment_instance_tag = experiment + '-' + str(seed_no)
        experiment_traces[experiment_instance_tag]={}
        exp_results_folder = os.path.join(results_folder, experiment, mode) # folder with results of the experiment
        exp_results_file = os.path.join(exp_results_folder, experiment_instance_tag + '-'+ mode + '.npy') # experiment data file
        trace = np.load(exp_results_file,allow_pickle='TRUE').item()
        experiment_traces[experiment] = trace["values"][location][year] # load to dictionary
        
        # check if timeslots_per_day are the same between experiments
        if timeslotsperday is None: 
            timeslotsperday = trace["params"]["env_params"]["timeslots_per_day"]
        else:
            assert timeslotsperday == trace["params"]["env_params"]["timeslots_per_day"], "timeslots_per_day don't match between experiments"
            timeslots_per_day = timeslotsperday
                

    # check if traces have the same length
    # we check the lenght of henergy obs trace for each experiment
    trace_length = 0
    for experiment in experiment_list:
        if trace_length != 0:
            assert trace_length == len(experiment_traces[experiment]["env_log"][:,1])
        else:
            trace_length = len(experiment_traces[experiment]["env_log"][:,1])

    # calculate the start and end index for the traces
    NO_OF_TIMESLOTS_PER_DAY = timeslots_per_day
    NO_OF_DAY_TO_PLOT = int(min(NO_OF_DAY_TO_PLOT, trace_length/NO_OF_TIMESLOTS_PER_DAY))
    END_DAY = START_DAY + NO_OF_DAY_TO_PLOT

    start_index = START_DAY*NO_OF_TIMESLOTS_PER_DAY
    end_index = END_DAY*NO_OF_TIMESLOTS_PER_DAY
    
    # plot the traces
    # create figure
    fig, axs = plt.subplots(nrows=1,
                            ncols=1,
                            figsize=[20,4],
                            sharex=True)
    
    for experiment in experiment_list:
        # set axis parameters
        axs.grid(which='major', axis='x', linestyle='--')
        axs.set_ylim(-0.1,1.1)
        axs.set_xlim([0,NO_OF_TIMESLOTS_PER_DAY*NO_OF_DAY_TO_PLOT])
        xtick_resolution = max(1,int(NO_OF_DAY_TO_PLOT/10))
        axs.set_xticks(np.arange(start=0,
                                  stop=NO_OF_TIMESLOTS_PER_DAY*(NO_OF_DAY_TO_PLOT+1),
                                  step=NO_OF_TIMESLOTS_PER_DAY*xtick_resolution))
        axs.set_xticklabels(np.arange(start=START_DAY,
                                       stop=END_DAY+1,
                                       step=xtick_resolution))

        for metric in metric_list:
            # if metric is in env_log
            if metric in ["henergy", "penergy", "benergy", "menergy", "req_obs"]:
                metric_trace = experiment_traces[experiment]["env_log"][:,env_log_metric_dict[metric]]
                time_trace = np.arange(timeslots_per_day*NO_OF_DAY_TO_PLOT)
                axs.step(time_trace,
                         metric_trace[start_index:end_index],
                         where='post',
                         color=experiment_color[experiment],
                         alpha=0.7,
                         linewidth=1.0,
                         linestyle=metric_linestyle[metric],
                         label=experiment + "-" + metric)
            # if metric is not in env_log
            else:
                metric_trace = experiment_traces[experiment][metric]
                time_trace = np.arange(timeslots_per_day*NO_OF_DAY_TO_PLOT)
                axs.step(time_trace,
                         metric_trace[start_index:end_index],
                         where='post',
                         color=experiment_color[experiment],
                         alpha=0.7,
                         linewidth=1.0,
                         linestyle=metric_linestyle[metric],
                         label=experiment + "-" + metric)
    
    axs.legend(loc="best",
                    # ncol=7,
                    # bbox_to_anchor=(0,1.0,1,1)
                  )
    plt.close()
    return fig
    # plt.show()
# End of compare_trace
########################################################


def compare_agents(results_folder, # the folder holding results of all experiments
                    environment_tag,
                    agent_list,
                    label_list,
                    seed_list,
                    mode,
                    location):
    
    # Get experiment name
    experiment_list = [environment_tag+'-'+agent for agent in agent_list]

    # human readable names for experiments
    label_list = label_list
    
    # Assign each experiment a label (to be used in figure legend)
    experiment_label = dict(zip(experiment_list, label_list))

    # specify a color for each experiment
    color_list = list(matplotlib.colors.TABLEAU_COLORS.keys())
    experiment_color = dict(zip(experiment_list, color_list))
    
    # Load all data in a dictionary 

    # dictionary to hold experimental data
    results = {}

    # load data from each experiment
    for experiment in experiment_list:
        results[experiment]={}
        for seed_no in seed_list:
            # Load data of experiment and store in a dictionary
            experiment_instance_tag = experiment + '-' + str(seed_no)
            root_folder = os.path.dirname(os.getcwd())
            # folder with results of the experiment
            exp_results_folder = os.path.join(root_folder,"results", experiment, mode)
            # experiment data file
            exp_results_file = os.path.join(exp_results_folder, experiment_instance_tag + '-'+ mode + '.npy') 
            exp_result = np.load(exp_results_file,allow_pickle='TRUE').item()
            results[experiment][seed_no] = exp_result["values"] # load to dictionary
    
    # Check if each experiment have the same location
    location_list = []
    for experiment in experiment_list:
        for seed_no in seed_list:
            if location_list: # if dummy_list is not empty
                # check if all experiments have the same location keys
                assert location_list == list(results[experiment][seed_no].keys()), "Locations differ among experiments and/or seeds"
            else: # if dummy_list is empty
                location_list = list(results[experiment][seed_no].keys())
                
    # Check if each experiment have the same number of years
    year_list = []
    for experiment in experiment_list:
        for seed_no in seed_list:
            for location in location_list:
                if year_list:
                    assert year_list == list(results[experiment][seed_no][location].keys()), "Years differ among experiments/seeds/locations" 
                else:
                    year_list = list(results[experiment][seed_no][location].keys())
    
    # Add statistical summary keys to dictionaries
    for experiment in experiment_list:
        results[experiment]["minimum"] = {}
        results[experiment]["first_q"] = {}
        results[experiment]["average"] = {}
        results[experiment]["third_q"] = {}
        results[experiment]["maximum"] = {}
        for location in location_list:
            results[experiment]["minimum"][location] = {}
            results[experiment]["first_q"][location] = {}
            results[experiment]["average"][location] = {}
            results[experiment]["third_q"][location] = {}
            results[experiment]["maximum"][location] = {}
            for year in year_list:
                results[experiment]["minimum"][location][year] = {}
                results[experiment]["first_q"][location][year] = {}
                results[experiment]["average"][location][year] = {}
                results[experiment]["third_q"][location][year] = {}
                results[experiment]["maximum"][location][year] = {}
    
    
    # get min, avg and max downtimes
    for experiment in experiment_list:
        for location in location_list:
            for year in year_list:
                dummy = []
                for seed in seed_list:
                    dummy.append(results[experiment][seed][location][year]['downtimes'])
                results[experiment]["minimum"][location][year]['downtimes'] = np.min(dummy)
                results[experiment]["first_q"][location][year]['downtimes'] = np.percentile(dummy, 25)
                results[experiment]["average"][location][year]['downtimes'] = np.mean(dummy)
                results[experiment]["third_q"][location][year]['downtimes'] = np.percentile(dummy, 75)
                results[experiment]["maximum"][location][year]['downtimes'] = np.max(dummy)

    # get min, avg and max avg_sense_reward
    for experiment in experiment_list:
        for location in location_list:
            for year in year_list:
                dummy = []
                for seed in seed_list:
                    avgsnsrwd = results[experiment][seed][location][year]['sense_reward_log'].mean()
                    results[experiment][seed][location][year]['avg_sense_reward'] = avgsnsrwd # add new entry
                    dummy.append(avgsnsrwd)
                results[experiment]["minimum"][location][year]['avg_sense_reward'] = np.min(dummy)
                results[experiment]["first_q"][location][year]['avg_sense_reward'] = np.percentile(dummy, 25)
                results[experiment]["average"][location][year]['avg_sense_reward'] = np.mean(dummy)
                results[experiment]["third_q"][location][year]['avg_sense_reward'] = np.percentile(dummy, 75)
                results[experiment]["maximum"][location][year]['avg_sense_reward'] = np.max(dummy)

    # get min, avg and max avg_enp_reward
    for experiment in experiment_list:
        for location in location_list:
            for year in year_list:
                dummy = []
                for seed in seed_list:
                    avgenprwd = results[experiment][seed][location][year]['enp_reward_log'].mean()
                    results[experiment][seed][location][year]['avg_enp_reward'] = avgenprwd # add new entry
                    dummy.append(avgenprwd)
                results[experiment]["minimum"][location][year]['avg_enp_reward'] = np.min(dummy)
                results[experiment]["first_q"][location][year]['avg_enp_reward'] = np.percentile(dummy, 25)
                results[experiment]["average"][location][year]['avg_enp_reward'] = np.mean(dummy)
                results[experiment]["third_q"][location][year]['avg_enp_reward'] = np.percentile(dummy, 75)
                results[experiment]["maximum"][location][year]['avg_enp_reward'] = np.max(dummy)
    
    # Plot Downtimes and Sense Rewards
    single_column_figure_width = 3.487
    double_column_figure_width = 7*2

    # fig_width = single_column_figure_width
    fig_width = double_column_figure_width
    fig_height = fig_width / 1.618

    fig, axs = plt.subplots(nrows=2,
                            ncols=1,
                            figsize=[fig_width,fig_height], # in inches
                            sharex=True)
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=.05)

    #######################################################################################
    # # left  = 0.125  # the left side of the subplots of the figure
    # # right = 0.9    # the right side of the subplots of the figure
    # # bottom = 0.1   # the bottom of the subplots of the figure
    # # top = 0.9      # the top of the subplots of the figure
    # # wspace = 0.2   # the amount of width reserved for blank space between subplots
    # # hspace = 0.2   # the amount of height reserved for white space between subplots
    #######################################################################################

    sense_reward_ax  = axs[0]
    downtimes_ax = axs[1]

    location = 'tokyo'
    print(location)

    # avg_sense_reward
    for experiment in experiment_list:
        min_data = [results[experiment]["minimum"][location][year]['avg_sense_reward'] for year in year_list]
        qt1_data = [results[experiment]["first_q"][location][year]['avg_sense_reward'] for year in year_list]
        avg_data = [results[experiment]["average"][location][year]['avg_sense_reward'] for year in year_list]
        qt3_data = [results[experiment]["third_q"][location][year]['avg_sense_reward'] for year in year_list]
        max_data = [results[experiment]["maximum"][location][year]['avg_sense_reward'] for year in year_list]


        sense_reward_ax.fill_between(year_list, y1=qt1_data, y2=qt3_data, color=experiment_color[experiment],alpha=0.2)
        sense_reward_ax.plot(year_list, avg_data, 
                             color=experiment_color[experiment], 
                             label=experiment_label[experiment])

    sense_reward_ax.text(0.035,0.5, 'sense-utility', 
             size='x-small', ha='center', va='center', 
            rotation='vertical',  transform=sense_reward_ax.transAxes)
    # sense_reward_ax.set_title('sense utility')
    # sense_reward_ax.set_ylabel('sense utility')    
    sense_reward_ax.legend(loc="lower left",
                           ncol=2,
                           # fontsize='x-small',
                            bbox_to_anchor=(-0.02,0.95,1.04,1),
                            mode="expand",
                           labelspacing=0.1,)
    sense_reward_ax.grid(which='major', axis='x', linestyle='--')

    # Downtimes
    for experiment in experiment_list:
        min_data = [results[experiment]["minimum"][location][year]['downtimes'] for year in year_list]
        qt1_data = [results[experiment]["first_q"][location][year]['downtimes'] for year in year_list]
        avg_data = [results[experiment]["average"][location][year]['downtimes'] for year in year_list]
        qt3_data = [results[experiment]["third_q"][location][year]['downtimes'] for year in year_list]
        max_data = [results[experiment]["maximum"][location][year]['downtimes'] for year in year_list]

        width = 0.8/len(experiment_list)  # the width of the bars 
        xroot = np.array(year_list) # label locations
        xoffset = -0.8/2 + experiment_list.index(experiment) 
        # IQR = Q3-Q1 = (Q3-AVG) + (AVG-Q1) So that we can anchor it at the average value

        yerr_min = np.array(avg_data) - np.array(qt1_data)
        yerr_max = np.array(qt3_data) - np.array(avg_data)
        downtimes_ax.bar(xroot+xoffset*width, avg_data,width, yerr = [yerr_min, yerr_max], 
                         color=experiment_color[experiment], 
                         label=experiment_label[experiment],
                         error_kw=dict(ecolor='black', lw=1, capsize=0.5, capthick=width*0.5, alpha=0.2))

    downtimes_ax.set_xticks(year_list[::5])
    downtimes_ax.set_xticklabels(year_list[::5], rotation=0)

    downtimes_ax.text(0.15,0.85, 'Downtimes', 
                     size='x-small', ha="center", 
                     transform=downtimes_ax.transAxes)
    # downtimes_ax.set_title('downtimes')
    # downtimes_ax.set_ylabel('downtimes')    
    # downtimes_ax.legend(loc="lower left",
    #                    ncol=1,
    #                    bbox_to_anchor=(0,0.8,1,1))
    downtimes_ax.grid(which='major', axis='y', linestyle='--')


    fig.suptitle(environment_tag)
    plt.close()
    return fig

# End of compare_agents
########################################################