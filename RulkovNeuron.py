import matplotlib.patches as mpatches
import pylab
import random
import numpy as np

class RulkovNeuron(object):
    
    def __init__(self, fast=-1.0, slow=-3.93, act=0.0, mu=0.001, control=4.5, longAvg=[], medAvg=[], leak=-0.01):
        self.fast = fast
        self.slow = slow
        self.act = act
        self.mu = mu
        self.control = control
        self.longAvg = longAvg
        self.medAvg = medAvg
        self.leak = leak

    def update(self,current,inputs):
        dFast = self.membraneFunc(self.fast,self.slow+current,self.control)
        dSlow = self.slowFunc(self.slow,self.fast,self.mu,inputs+self.leak)
        
        self.fast = dFast
        self.slow = dSlow

    def membraneFunc(self, x, y, a):
        if x <= 0.0:
            self.act=0.0
            return (a/(1.0-x))+y
        elif 0.0 < x < (a+y):
            return a+y
        elif x >= (a+y):
            self.act=1.0
            return -1.0
            
    def membraneDeriv(self, x, a):
        return a/(x-1)**2
        
    def slowFunc(self, slow, fast, mu, inputs):
        return slow - mu*(fast+1)+mu*inputs
