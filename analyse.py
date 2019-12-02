#!/usr/bin/env python


import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
matplotlib.rcParams['text.usetex'] = True


def analyse(pref):
    scalar = np.load("deltas" + pref + "_50.npy")
    tensor = np.load("deltat" + pref + "_50.npy")
    logk = np.linspace(-2, 3, 11)

    plt.plot(logk, scalar)
    plt.xlabel(r"$\log k$")
    plt.ylabel(r"$\Delta^2_R$")
    plt.savefig("deltas_" + pref, format="pdf")

    plt.clf()
    
    plt.plot(logk, tensor)
    plt.xlabel(r"$\log k$")
    plt.ylabel(r"$\Delta^2_h$")
    plt.savefig("deltat_" + pref, format="pdf")

    logscalar = np.log(scalar)
    slope, intercept, r_value, p_value, std_err = linregress(logk, logscalar)
    
    r = (2 * tensor / scalar).mean()
    print "r = " + str(r)
    print "n_s = " + str(slope + 1)

    
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print "please enter the file index as a commandline argument"
    else:
        analyse(sys.argv[1])
