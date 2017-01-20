#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
class Neuron(object):

    np = __import__('numpy')
    def __init__(self, VERBOSE=False):
        self.HI_FREQ = 250 #Hz
        self.LOW_FREQ = 50 #Hz
        self.prob_to_hi = 0.7
        self.prob_to_lo = 0.3
        self.spike = 0
        self._history = []
        self.VERBOSE = VERBOSE

        self.rate = self.np.random.choice([self.HI_FREQ, self.LOW_FREQ])

    def step(self):
        # roll state switch

        roll = np.random.uniform(0,1)

        self.spike = 0
        if self.rate == self.LOW_FREQ and roll < self.prob_to_hi:
            # change to high state
            if self.VERBOSE: print('Switching to HI_FREQ')
            self.rate = self.HI_FREQ

        elif self.rate == self.HI_FREQ and roll < self.prob_to_lo:
            # change to low state

            if self.VERBOSE: print('Switching to LOW_FREQ')
            self.rate = self.LOW_FREQ

        # roll for spike
        roll_spike = np.random.uniform(0,1)
        if roll_spike < self.rate/1000.0:
            self.spike = 1

        self._history.append(self.spike)
        return self.spike

    def status(self):
        print("Firing rate: %d" % self.rate)
        print("Probability -> HIGH: %.1f" % self.prob_to_hi)
        print("Probability -> LOW: %.1f" % self.prob_to_lo)
        print("Spike %d" % self.spike)

    def history(self, start=False, end=False):
            if start and end:
                return self._history[start:end]
            elif start:
                # Start = True, End = False
                return self._history[start:]
            elif end:
                # Start = False; End = True
                return self._history[:end]
            else:
                return self._history

def main():
    num_neurons = 30
    num_trials =5
    num_time = 266

    neurons = np.empty_like(range(num_neurons), dtype=object)
    print(neurons)
    data = np.empty((num_trials,num_neurons,num_time), dtype=object)
    def inc_n(neuron):
        return neuron.step()

    def hist_n(neuron, start=False, end=False):
        print(type(neuron))
        print(start)
        print(end)
        return neuron.history(start,end)

    # initialize 30 neurons
    for n_id in range(num_neurons):
        neurons[n_id] = Neuron()

    for trial in range(num_trials):
        for t in range(num_time):
            # increment Neurons
            i = 0
            for n in neurons:
                data[trial,i,t]= n.step()
                i=i+1

    print(np.shape(data))
    sio.savemat('sim_data_5.mat', {'test_data':data})
    print(sio.whosmat('sim_data_5.mat'))
    return data

if __name__ == "__main__": main()
