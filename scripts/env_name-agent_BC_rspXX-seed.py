import random
import os
import os.path
import argparse

import gym
gym.logger.set_level(40) # remove gym warning about float32 bound box precision

import numpy as np
import matplotlib.pyplot as plt

import sys
import pathlib

# in jupyter (lab / notebook), based on notebook path
module_path = str(pathlib.Path.cwd().parents[0] / "py")
# in standard python
# module_path = str(pathlib.Path.cwd(__file__).parents[0] / "py")

if module_path not in sys.path:
    sys.path.append(module_path)

import common.env_lib
import common.agents

parser = argparse.ArgumentParser()
parser.add_argument("--env", default="sense", type=str, help="Environment. Specified in ./py/common/env_lib.py")
parser.add_argument("--location", default="tokyo", type=str, help="Environment Location")
# parser.add_argument("--agent", default="agent_BC", type=str, help="type of agent")
parser.add_argument("--param", default=70, type=int, help="agent_BC responsiveness parameter.")
parser.add_argument("--seed", default=238, type=int, help="Set seed [default: 230]")

args = parser.parse_args()

# arguments
seed      = args.seed
env_name  = args.env
rsp = args.param
env_location = args.location

# hard set arguments
agent_name = "agent_BC"

# set seed
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)

# create environment
env = eval("common.env_lib."+env_name+"()")

# other required arguments
START_YEAR = 1995
NO_OF_YEARS = 2#3
timeslots_per_day = 24
prediction_horizon = 10*timeslots_per_day
offset = timeslots_per_day/2
REQ_TYPE = "random"
henergy_mean= 0.13904705134356052 # 10yr hmean for tokyo
location = "tokyo"

# get root_folder
# should point to "../morl_eno/"
root_folder = os.path.dirname(os.getcwd())

# Create agent with corresponding rsp
agent = eval("common.agents." + agent_name + "("+ str(rsp)+")")

# Tags
env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
agent_tag = agent_name + "_rsp" + str(rsp)

# experiment tag
# name of folder to save models and results
experiment_type_tag = env_tag  + "-" + agent_tag
experiment_instance_tag =  experiment_type_tag + '-' + str(seed)

# Folder/file to save test results
test_results_folder = os.path.join(root_folder,"results", experiment_type_tag, "test")
if not os.path.exists(test_results_folder): 
        os.makedirs(test_results_folder) 
test_log_file = os.path.join(test_results_folder, experiment_instance_tag + '-test.npy')    


experiment_instance_result = {}
experiment_instance_result[env_location] = {}
for year in range(START_YEAR, START_YEAR+NO_OF_YEARS):
    env.set_env(env_location, 
                year, 
                timeslots_per_day, 
                REQ_TYPE, 
                offset,
                p_horizon=prediction_horizon,
                hmean=henergy_mean)    
    state = env.reset()
    reward_rec = []
    ep_done_rec = []
    done = False
    while not done:
        if env.RECOVERY_MODE:
            no_action = 0            
            next_state, reward, done, _ = env.step(no_action)       
        else:
            paction = agent(state)
            next_state, reward, done, _ = env.step(paction)                
        reward_rec.append(reward)
        ep_done = done or env.RECOVERY_MODE
        ep_done_rec.append(ep_done)
        state = next_state

    # Log the traces and summarize results
    year_trace={}

    # Saving traces
    year_trace['reward_rec'] = np.array(reward_rec)
    year_trace['ep_done_rec'] = np.array(ep_done_rec)
    year_trace['action_log'] = np.array(env.action_log)
    year_trace['sense_dc_log'] = np.array(env.sense_dc_log)
    year_trace['env_log'] = np.array(env.env_log)
    year_trace['sense_reward_log'] = np.array(env.sense_reward_log)
    year_trace['enp_reward_log'] = np.array(env.enp_reward_log)


    # Summarizing environmental traces for later reference

    env_log = year_trace['env_log']

    # Get henergy metrics
    henergy_rec = env_log[:,1]
    avg_henergy = henergy_rec.mean()
    year_trace['avg_henergy'] = avg_henergy

    # Get request metrics
    req_rec = env_log[:,5]
    avg_req = req_rec.mean()            
    year_trace['avg_req'] = avg_req

    # Get reward metrics
    # In this case, the reward metrics directly reflect the conformity
    reward_rec = year_trace['reward_rec']
    # zero rewards correspond to downtimes
    # To find average reward, remove zero values and then average the remaining non-zero rewards
    index = np.argwhere(reward_rec<=0)
    rwd_rec = np.delete(reward_rec, index)
    avg_rwd = rwd_rec.mean()
    year_trace['avg_rwd'] = avg_rwd

    # Get downtime metrics
    # Total number of times battery dropped to zero
    batt_rec = env_log[:,3].copy()
    batt_rec[batt_rec>0.1]=0 # all battery levels over B_MIN are toggled OFF
    batt_rec[batt_rec!=0]=1 # all remaining non-zeroed out battery levels are toggled ON
    # Toggle only those instances when there is a transition from 0 -> 1
    downtimes = np.count_nonzero(batt_rec[:-1] < batt_rec[1:])
    year_trace['downtimes'] = downtimes

    # Save yearly trace in experiment dictionary
    experiment_instance_result[env_location][year] = year_trace
# end for(year)
# end for(location)
np.save(test_log_file, experiment_instance_result)

# Load npy file and output to stdout
# Tags
env_tag = env_name + '_t' + str(timeslots_per_day) + '_' + REQ_TYPE
agent_tag = agent_name + "_rsp" + str(rsp)

# experiment tag
# name of folder to load models and results
experiment_type_tag = env_tag  + "-" + agent_tag
experiment_instance_tag =  experiment_type_tag + '-' + str(seed)

# Folder/file to load test results from
test_results_folder = os.path.join(root_folder,"results", experiment_type_tag, "test")
assert os.path.exists(test_results_folder), "'" + test_results_folder + "' folder does not exist"
test_log_file = os.path.join(test_results_folder, experiment_instance_tag + '-test.npy')   

# Load data
experiment_instance_result = np.load(test_log_file,allow_pickle='TRUE').item()    

print("Experiment:", experiment_instance_tag)
print("LOCATION".ljust(12), "YEAR".ljust(6), "HMEAN".ljust(8), "REQ_MEAN".ljust(8), "AVG_DC".ljust(8), 
  "SNS_RWD".ljust(8), "ENP_RWD".ljust(8), "AVG_RWD".ljust(8), "DOWNTIMES".ljust(9))

location_list = list(experiment_instance_result.keys())
for location in location_list:
    yr_list = list(experiment_instance_result[location].keys())
    for year in yr_list:
        year_trace = experiment_instance_result[location][year]
        # Print summarized metrics
        print(location.ljust(12), year, end=' ')
        sense_avg_rwd = year_trace['sense_reward_log'].mean()
        enp_avg_rwd = year_trace['enp_reward_log'].mean()

        average_rwd = year_trace['avg_rwd']
        total_downtimes = year_trace['downtimes']
        hmean = year_trace['avg_henergy']
        reqmean = year_trace['avg_req']
        sense_dc_mean = year_trace['sense_dc_log'].mean()

        print(f'{hmean:7.3f}',end='  ')
        print(f'{reqmean:7.3f}',end='  ')
        print(f'{sense_dc_mean:7.3f}',end='  ')
        print(f'{sense_avg_rwd:7.3f}',end='  ')
        print(f'{enp_avg_rwd:7.3f}',end='  ')
        print(f'{average_rwd:7.3f}',end='  ')
        print(f'{total_downtimes:5d}',end='  ')
        print("")
print('*'*90)
print('\n')


