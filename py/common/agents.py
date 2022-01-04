# triggers ON when battery level increase above a threshold
# triggers OFF when battery level decreases below a threshold
# has a hysterisis effect
class schmitt:
    def __init__(self,
                 param = {"threshold_hi": 0.95,
                          "threshold_lo": 0.20,
                          "val_hi": 0.9,
                          "val_lo": 0.1}
                ):
        self.threshold_hi = param["threshold_hi"]
        self.threshold_lo = param["threshold_lo"]
        self.val_hi = param["val_hi"]
        self.val_lo = param["val_lo"]
        self.trigger = False
        
    def __call__(self, state):
        # state = [time, henergy, penergy, benergy, menergy, req_obs]
        batt = state[3] 
        # agent reacts to immediate battery level
        if not self.trigger:
            if batt > self.threshold_hi:
                self.trigger = True
                action = self.val_hi
            else:
                action = self.val_lo
                
        if self.trigger:
            if batt < self.threshold_lo:
                self.trigger = False
                action = self.val_lo
            else:
                action = self.val_hi
                
        return action
    
    def false_call(self, batt):
        # agent reacts to immediate battery level
        if not self.trigger:
            if batt > self.threshold_hi:
                self.trigger = True
                action = self.val_hi
            else:
                action = self.val_lo
        if self.trigger:
            if batt < self.threshold_lo:
                self.trigger = False
                action = self.val_lo
            else:
                action = self.val_hi
                
        return action

# triggers ON when battery level increase above a threshold
# triggers OFF when battery level decreases below that threshold
class bangbang:
    def __init__(self,
                 threshold=0.95,
                 val_hi=0.9,
                 val_lo=0.1):
        self.threshold = threshold
        self.val_hi = val_hi
        self.val_lo = val_lo
        
    def __call__(self, state):
        # state = [time, henergy, penergy, benergy, menergy, req_obs]
        batt = state[3] 
        # agent reacts to immediate battery level
        if batt > self.threshold:
            action = self.val_hi
        else:
            action = self.val_lo
        return action

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
    
# create battery-centric agent parameterized by a responsiveness variable rsp
# similar to agent_BC, difference being that it responds to mean battery level
# not immediate battery level
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

# always outputs the same conformity
# this does not translate to CONSTANT DUTY CYCLE!!!! 
class agent_constant:
    def __init__(self, conformity):
        self.action = conformity
               
    def __call__(self, state):
        return self.action

