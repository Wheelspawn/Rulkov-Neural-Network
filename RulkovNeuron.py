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

def test(sending, noise, inputted_control):
    
    r1=RulkovNeuron() # set up the receiving neuron.
    r1.control=inputted_control # set the control
    
    firstLayer = []
    
    for k in range(0,sending):
        firstLayer.append(RulkovNeuron(control=4)) # generate the sending layer.
    
    f1=[]
    f2=[]
    tStep=[]
    
    for n in range(0,4000): # timesteps
        
        avg=0 # avg action potential of sending layer
        median = []
        
        for neuron in firstLayer: # update sending neurons
            neuron.update(0.0,0.03+random.uniform(-noise,noise))
            avg += neuron.fast
            
        f1.append(avg/sending) # finish the average calculation.
        
        if n < 1000: # start off with a clamped input. This ensures a more stable spike train.
            r1.update(0.0,0.03+random.uniform(-noise,noise))
        else: # send the spikes from the sending layer to receiving layer.
            r1.update(0.0,-0.12+sum(list(neuron.act for neuron in firstLayer)))
        
        tStep.append(n)
        f2.append(r1.fast)
        
    pylab.plot(tStep,f1,color='r')
    pylab.plot(tStep,f2,color='b')
    
    layer1Plt = mpatches.Patch(color='red', label='Avg potential of (' + str(sending) + ') sending neurons')
    layer2Plt = mpatches.Patch(color='blue', label='Potential of receiving neuron')
    
    pylab.legend(handles=[layer1Plt,layer2Plt])
    
    pylab.xlim(xmin=500, xmax=4000)
    pylab.ylim(ymin=-1.6, ymax=1.6)
    pylab.xlabel('Time')
    pylab.ylabel('Action potential')
    pylab.show()

sending=input("This Rulkov network has two layers. How many neurons do you want in the sending layer? 60 (or more) is a good number.\n")
noise=input("Add noise? 0.01 is a good number.\n")
control=input("Adjust control? 2.8 is a good number, increase it and the receiving neurons to generate spiking behavior.\n")
if int(sending) <= 48:
    print("Careful! Lack of sending neurons may stop the neuron from spiking.")
if float(control) <= 2.75:
    print("Careful! A low value for the control may stop the neuron from spiking.")
test(int(sending), float(noise), float(control))