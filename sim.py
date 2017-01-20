#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
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

    @property
    def history(self):
        return self._history


def main():
    sim_time = 200000

    n = Neuron()
    for i in range(sim_time):
        #initialize
        n.step()
    plt.plot(n.history)
    plt.ylabel('spikes')
    plt.show()


if __name__ == "__main__": main()
