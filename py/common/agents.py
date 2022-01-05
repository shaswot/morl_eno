class base_agent:
    # initialize agent with parameters
    # parameters is a dictionary
    def __init__(self,
                 params
                ):
        self.params = params
        assert type(self.params) is dict, "param should be in the form of a dictionary"
        self.myname = params["agent_name"]
    
    # returns action depending on input state
    def __call__(self, state):
        pass
    
    # returns actions depending on battery value
    # required for plotting agent characteristics
    def false_call(self, batt):
        pass

# triggers ON when battery level increase above a threshold
# triggers OFF when battery level decreases below a threshold
# has a hysterisis effect
class schmitt(base_agent):
    def __init__(self,
                 params
                ):
        super(base_agent, self).__init__()
        
        self.threshold_hi = params["threshold_hi"]
        self.threshold_lo = params["threshold_lo"]
        self.val_hi = params["val_hi"]
        self.val_lo = params["val_lo"]
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
class bangbang(base_agent):
    def __init__(self,
                 params
                ):
        super(base_agent, self).__init__()
        
        self.threshold = params["threshold"]
        self.val_hi = params["val_hi"]
        self.val_lo = params["val_lo"]
        
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
class agent_BC(base_agent):
    def __init__(self, 
                 params
                ):
        super(base_agent, self).__init__()
        self.rsp = params["rsp"]
               
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
class agent_mBC(base_agent):
    def __init__(self, 
                 params = {"rsp": 0.95}
                ):
        super(base_agent, self).__init__()
        self.rsp = params["rsp"]
               
    def __call__(self, state):
        # state = [time, henergy, penergy, benergy, menergy, req_obs]
        batt = state[4] 
        # agent reacts to mean battery level
        x = (batt-0.1)/0.9
        action = (x-self.rsp*x)/(self.rsp-2*self.rsp*x+1)
        return action

# always outputs the same conformity
# this does not translate to CONSTANT DUTY CYCLE!!!! 
class agent_constant(base_agent):
    def __init__(self, 
                 params
                ):
        super(base_agent, self).__init__()
        self.action = params["conformity"]
               
    def __call__(self, state):
        return self.action

