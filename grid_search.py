from scipy.optimize import brute

def run(x):
    # x is a 1-D array of each parameter to be optimized
    batch_size = x[0]
    epoch_size = x[1]
