#!/usr/bin/env python

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class InfModel:
    """All time-dependent quantities in this class are in terms of conformal time!"""
    def __init__(self, V, dV, k, phi0=1, dphi0=-0.1, a0=1, niter=40000, t0=-1):
        self.V = V
        self.dV = dV
        self.init_params = [phi0, dphi0, a0]
        self.k = k
        self.t0 = t0
        self.v0_re = math.cos(self.k * self.t0) / math.sqrt(2 * self.k)
        self.v0_im = -math.sin(self.k * self.t0) / math.sqrt(2 * self.k)
        self.dv0_re = -math.sin(self.k * self.t0) * math.sqrt(self.k / 2)
        self.dv0_im = -math.cos(self.k * self.t0) * math.sqrt(self.k / 2)
        self.mukhanov_params_re = [self.v0_re, self.dv0_re]
        self.mukhanov_params_im = [self.v0_im, self.dv0_im]
        self.phi0 = phi0
        self.dphi0 = dphi0
        self.a0 = a0
        self.niter = niter

    def diffs1(self, U, tau):
        # returns a list of functions f'(tau) on which we wish to compute f(tau)
        phi, dphi, a = U
        return [dphi,  # needed for initial condition
                -a**2 * self.dV(phi) - 2 * dphi * math.sqrt((a**2 * self.V(phi) + dphi**2 / 2) / 3),  # phi''
                math.sqrt(dphi**2 / 2 + self.V(phi) * a**2) * a]  # a'

    def get_a_phi(self, plot=False):
        self.t = np.linspace(self.t0, 0, num=self.niter)
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

    def get_vk(self, plot=False):
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
        tmp = odeint(self.diffs2, self.mukhanov_params_re, self.t, args=(x, self.k))
        vk_re = tmp[:, 0]
        dvk_re = tmp[:, 1]

        tmp = odeint(self.diffs2, self.mukhanov_params_im, self.t, args=(x, self.k))
        vk_im = tmp[:, 0]
        dvk_im = tmp[:, 1]

        #  RR = (H * v / dphi / a)**2 at horizon crossing
        # i0 = (np.abs(H * self.a - k)).argmin()
        # print("D^2_R(k=" + str(k) + ") = " + str((H[i0] * vk[i0] / self.dphi[i0] / self.a[i0])**2))

        if plot:
            plt.plot(self.t, vk_re**2 + vk_im**2)
            plt.xlabel("Conformal Time")
            plt.ylabel("vk")
            plt.show()

    def closest(self, t, x):
        # x is the array of z'' / z
        # t is a single value of conformal time
        if t <= 0 and t >= self.t0:
            return x[int((t - self.t0) * self.niter)]
        elif t < self.t0:
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
    return 4 * phi**3



model = InfModel(V, dV, 100)
# model.get_a_phi(plot=True)
model.get_vk(True)
