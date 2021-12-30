# create battery-centric agent parameterized by a responsiveness variable rsp
class agent_BC:
    def __init__(self):
        pass
               
    def __call__(self, batt,rsp=0.95):
        x = (batt-0.1)/0.9
        action = (x-rsp*x)/(rsp-2*rsp*x+1)
        return action

