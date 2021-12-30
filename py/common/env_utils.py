import numpy as np
import matplotlib.pyplot as plt
########################################################
# Class for battery
########################################################
class battery(object):
    def __init__(self, BINIT, BEFF):
        self.batt = BINIT # battery starts with 70% charge
        self.BEFF = BEFF
        
        assert 0 <= self.batt <= 1, 'Incorrect Battery Initialization'
        assert 0 < self.batt <= 1 , 'Incorrect Battery Efficiency'
    def charge(self, energy):
        assert energy >= 0, "Charging Energy should be >= 0"
        self.batt += energy*self.BEFF
        self.batt = np.clip(self.batt,0,1)
        
    def discharge(self, energy): #energy will be in negative
        assert energy < 0, "Discharging Energy should be < 0"
        self.batt += energy
        self.batt =  np.clip(self.batt,0,1)
        
    def get_batt_state(self):
        return np.clip(self.batt,0,1).item()
# End of class battery
########################################################