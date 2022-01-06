import numpy as np
import yaml
from difflib import SequenceMatcher

def get_agent_type(agent_list):
    names = agent_list

    string2 = names[0]
    for i in range(1, len(names)):
        string1 = string2
        string2 = names[i]
        match = SequenceMatcher(None, string1, string2).find_longest_match(0, len(string1), 0, len(string2))

    return (string1[match.a: match.a + match.size])

def display_tabular_summary(log_file):
    # Load data
    experiment_instance_result = np.load(log_file,allow_pickle='TRUE').item()    

    # print environmet, agent, experiment information
    print("Experiment:", experiment_instance_result["params"]["experiment_instance_tag"])
    print()
    print("Environment:", experiment_instance_result["params"]["env_type"])
    print(yaml.dump(experiment_instance_result["params"]["env_params"], sort_keys=False, default_flow_style=False))
    print("Agent:", experiment_instance_result["params"]["agent_type"])
    print(yaml.dump(experiment_instance_result["params"]["agent_params"], sort_keys=False, default_flow_style=False))
    print(yaml.dump(experiment_instance_result["params"]["exp_params"], sort_keys=False, default_flow_style=False))

    print("LOCATION".ljust(12), "YEAR".ljust(6), "HMEAN".ljust(8), "REQ_MEAN".ljust(8), "AVG_DC".ljust(8), 
      "SNS_RWD".ljust(8), "ENP_RWD".ljust(8), "AVG_RWD".ljust(8), "DOWNTIMES".ljust(9))

    location_list = list(experiment_instance_result["values"].keys())
    for location in location_list:
        yr_list = list(experiment_instance_result["values"][location].keys())
        for year in yr_list:
            year_trace = experiment_instance_result["values"][location][year]
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
    
def experiment_tag_generator(env_type, 
                             env_name, 
                             agent_type, 
                             agent_name, 
                             seed):
    env_tag = env_type + '_' + env_name
    agent_tag = agent_type + '_' + agent_name
    
    experiment_meta_type_tag = env_tag  + "-" + agent_type
    experiment_type_tag = env_tag  + "-" + agent_tag
    experiment_instance_tag =  experiment_type_tag + '-' + str(seed)
    
    return [experiment_meta_type_tag, experiment_type_tag, experiment_instance_tag]