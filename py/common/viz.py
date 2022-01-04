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
                  timeslots_per_day=24,
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
    for experiment in experiment_list:
        # Load data of experiment and store in a dictionary
        experiment_instance_tag = experiment + '-' + str(seed_no)
        experiment_traces[experiment_instance_tag]={}
        exp_results_folder = os.path.join(results_folder, experiment, mode) # folder with results of the experiment
        exp_results_file = os.path.join(exp_results_folder, experiment_instance_tag + '-'+ mode + '.npy') # experiment data file
        trace = np.load(exp_results_file,allow_pickle='TRUE').item()
        experiment_traces[experiment] = trace[location][year] # load to dictionary

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