# create battery-centric agent parameterized by a responsiveness variable rsp
class agent_BC:
    def __init__(self, rsp=0.95):
        self.rsp = rsp
               
    def __call__(self, state):
        # state = [time, henergy, penergy, benergy, menergy, req_obs]
        batt = state[3] 
        # agent reacts to immediate battery level
        x = (batt-0.1)/0.9
        action = (x-self.rsp*x)/(self.rsp-2*self.rsp*x+1)
        return action
    
    def false_call(self, batt):
        # required when plotting the agent's policy characteristics
        # agent reacts to immediate battery level
        x = (batt-0.1)/0.9
        action = (x-self.rsp*x)/(self.rsp-2*self.rsp*x+1)
        return action

class agent_mBC:
    def __init__(self, rsp=0.95):
        self.rsp = rsp
               
    def __call__(self, state):
        # state = [time, henergy, penergy, benergy, menergy, req_obs]
        batt = state[4] 
        # agent reacts to mean battery level
        x = (batt-0.1)/0.9
        action = (x-self.rsp*x)/(self.rsp-2*self.rsp*x+1)
        return action
    
class agent_constant:
    def __init__(self, conformity):
        self.action = conformity
               
    def __call__(self, state):
        return self.action

