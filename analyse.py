#!/usr/bin/env python


import sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import linregress
matplotlib.rcParams['text.usetex'] = True


def analyse(pref, ilen, offset, plot=True):
    scalar = np.load("deltas" + pref + "_" + str(ilen) + ".npy")
    tensor = np.load("deltat" + pref + "_" + str(ilen) + ".npy")
    logk = np.linspace(-2, 3, 11)

    x = np.exp(logk) * 0.05 / offset
    
    if plot:
        plt.semilogx(x, np.log(scalar))
        plt.xlabel(r"$k [Mpc^{-1}]$")
        plt.ylabel(r"$\log \Delta^2_R$")
        if pref == str(2):
            plt.title(r"$\phi^2$ Inflation for " + str(ilen) + " efolds")
        if pref == str(1):
            plt.title(r"$\phi$ Inflation for " + str(ilen) + " efolds")
        if pref == str(23):
            plt.title(r"$\phi^{\frac{2}{3}}$ Inflation for " + str(ilen) + " efolds")
        if pref == str(43):
            plt.title(r"$\phi^{\frac{4}{3}}$ Inflation for " + str(ilen) + " efolds")
        plt.savefig("deltas_" + pref + "_" + str(ilen) + ".pdf", format="pdf")

        plt.clf()

        plt.semilogx(x, np.log(tensor))
        plt.xlabel(r"$k [Mpc^{-1}]$")
        plt.ylabel(r"$\log \Delta^2_t$")
        if pref == str(2):
            plt.title(r"$\phi^2$ Inflation for " + str(ilen) + " efolds")
        if pref == str(1):
            plt.title(r"$\phi$ Inflation for " + str(ilen) + " efolds")
        if pref == str(23):
            plt.title(r"$\phi^{\frac{2}{3}}$ Inflation for " + str(ilen) + " efolds")
        if pref == str(43):
            plt.title(r"$\phi^{\frac{4}{3}}$ Inflation for " + str(ilen) + " efolds")
        plt.savefig("deltat_" + pref + "_" + str(ilen) + ".pdf", format="pdf")

    logscalar = np.log(scalar)
    slope, intercept, r_value, p_value, std_err = linregress(logk, logscalar)
    
    r = ( tensor / scalar).mean()
    print "r" + pref + str(ilen) + " = " + str(r)
    print "n" + pref + str(ilen) + " = " + str(slope + 1)

    
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print "please enter the file index and inflation length as commandline arguments and offset"
    else:
        analyse(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]))
