from RulkovNeuron import *

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
control=input("Adjust control? 2.8 is a good number, increase it (or the receiving neurons) to generate spiking behavior.\n")
if int(sending) <= 48:
    print("Careful! Lack of sending neurons may stop the neuron from spiking.")
if float(control) <= 2.75:
    print("Careful! A low value for the control may stop the neuron from spiking.")
test(int(sending), float(noise), float(control))
