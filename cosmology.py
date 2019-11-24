#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class InfModel:
    """All time-dependent quantities in this class are in terms of conformal time!"""
    def __init__(self, V, dV, phi0=7.501428561966892, dphi0=-9222, a0=0.001962224231203507, v0=0.1, dv0=0.1, niter=40000):
        self.V = V
        self.dV = dV
        self.init_params = [phi0, dphi0, a0]
        self.mukhanov_params = [v0, dv0]
        self.phi0 = phi0
        self.dphi0 = dphi0
        self.a0 = a0
        self.niter = niter

    def diffs1(self, U, tau):
        # returns a list of functions f'(tau) on which we wish to compute f(tau)
        phi, dphi, a = U
        return [dphi, # needed for initial condition 
                -a**2 * self.dV(phi) - 2 * dphi * math.sqrt(a**2 * self.V(phi) + dphi**2 / 2), # phi''
                math.sqrt(dphi**2 / 2 + self.V(phi) * a**2) * a] # a'
    
    def get_a_phi(self, plot=False):
        self.t = np.logspace(-10, np.log(2 * 10.6405) / np.log(10), num=self.niter)
        tmp = odeint(self.diffs1, self.init_params, self.t)
        self.phi = tmp[:, 0]
        self.dphi = tmp[:, 1]
        self.a = tmp[:, 2]
        if plot:
            plt.plot(self.t, self.a)
            plt.xlabel("Conformal Time")
            plt.ylabel("Scale Factor")
            plt.show()

            # Plot where inflation is happening! 
            plt.plot(self.t, self.phi)
            mask = np.where(4 * (self.dphi**2 / 2 / self.a**2) - 2 * self.V(self.phi) < 0, True, False)
            tinf = self.t[mask]
            phiinf = self.phi[mask]
            plt.plot(tinf, phiinf, 'go', markersize=2)
            plt.xlabel("Conformal Time")
            plt.ylabel("Phi")
            plt.show()
            
            plt.plot(self.t, self.dphi)
            plt.xlabel("Conformal Time")
            plt.ylabel("Phi'")
            plt.show()

    def get_vk(self, k, plot=False):
        self.get_a_phi(plot)
        H = np.sqrt(self.V(self.phi) + self.dphi**2 / 2 / self.a**2)
        z = self.dphi / H
        # Numerically compute z''
        ddz = np.empty(len(self.t))
        ddz[1:-1] = (((np.roll(z, -2) - np.roll(z, -1)) / (np.roll(self.t, -2) - np.roll(self.t, -1)) -
                     (np.roll(z, -1) - z) / (np.roll(self.t, -1) - self.t)) / (np.roll(self.t, -1) - self.t))[:-2]
        ddz[0] = ddz[1]
        ddz[-1] = ddz[-2]

        x = ddz / z
        tmp = odeint(self.diffs2, self.mukhanov_params, self.t, args=(x, k))
        vk = tmp[:, 0]
        dvk = tmp[:, 1]

        #  RR = (H * v / dphi / a)**2 at horizon crossing
        i0 = (np.abs(H * self.a - k)).argmin()
        print("D^2_R(k=" + str(k) + ") = " + str((H[i0] * vk[i0] / self.dphi[i0] / self.a[i0])**2))

        if plot:
            plt.plot(self.t, vk)
            plt.xlabel("Conformal Time")
            plt.ylabel("vk")
            plt.show()

    def closest(self, t, x):
        # x is the array of z'' / z
        # t is a single value of conformal time
        logt = math.log(t) / math.log(10)
        if -10 <= logt and logt <= 2 * math.log(10.6405) / math.log(10):
            return x[int((logt + 10) / (2 * math.log(10.6405) / math.log(10) + 10) * self.niter)]
        elif logt < -10:
            return x[0]
        else:
            return x[-1]

    def diffs2(self, V, tau, x, k):
        # returns [v', v''] so odeint can comptute [v, v']
        return [V[1], -(k**2 - self.closest(tau, x)) * V[0]]

        
################################################################################

            
def V(phi):
    return phi**4

def dV(phi):
    return 4*phi**3

model = InfModel(V, dV)
model.get_vk(5, False)

