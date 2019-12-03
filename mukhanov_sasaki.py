#!/usr/bin/env python

import time
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.misc import derivative
from scipy.interpolate import interp1d as interp


class InfModel:
    """All time-dependent quantities in this class are in terms of conformal time!"""
    def __init__(self, V, dV, k, phi0=30, psi0=-0.00000001, nstar=50, efold=676, niter=4000, t0=0):
        self.V = V
        self.dV = dV
        self.init_params = [float(phi0), float(psi0)]  # solve for b(t) = log(a(t)) instead of a(t) as before
        self.k = k
        self.t0 = t0
        # This line just sets log a = 0 nstar efolds before the end of inflation
        self.a0 = math.exp(-efold + nstar)
        self.efold = efold
        self.niter = int(niter * efold / 200)

    def update_k(self, k):
        self.k = k

        # We can choose t0 = 0 without loss of generality because the Mukhanov-Sasaki equation is invariant to phase multiplication
        self.v0_re = math.cos(self.k * self.t0) / math.sqrt(2 * self.k)
        self.v0_im = -math.sin(self.k * self.t0) / math.sqrt(2 * self.k)
        self.dv0_re = -math.sin(self.k * self.t0) * math.sqrt(self.k / 2)
        self.dv0_im = -math.cos(self.k * self.t0) * math.sqrt(self.k / 2)
        self.mukhanov_params_re = [self.v0_re, self.dv0_re]
        self.mukhanov_params_im = [self.v0_im, self.dv0_im]

        # Trial and Error led us to solve the Mukhanov-Sasaki equation over the interval (- 4 + log k, 6 + log k) to avoid numerical instabilities.
        # See guide for more details
        self.lowerrange = int(((self.efold - nstar - 4 + math.log(self.k)) / self.efold) * self.niter)
        self.upperrange = int(((self.efold - nstar + 6 + math.log(self.k)) / self.efold) * self.niter)

    def diffs1(self, U, b):
        # returns the differential equations for the scalar field
        phi, psi = U
        return [psi / math.sqrt(1 / 3.0 * (1 / 2 * psi**2 + self.V(phi))),
                -3 * psi - self.dV(phi) / math.sqrt(1 / 3.0 * (1 / 2 * psi**2 + self.V(phi)))]

    def get_a_phi(self, plot=False):
        # Instead of solving for a as before, try solving for b = log a to protect against divergence
        self.b = np.linspace(math.log(self.a0), math.log(self.a0) + self.efold, num=self.niter)
        tmp = odeint(self.diffs1, self.init_params, self.b)
        self.phi = tmp[:, 0]
        self.psi = tmp[:, 1]

        H = np.sqrt(self.V(self.phi) + self.psi**2 / 2)
        Hprime = np.gradient(H, self.b)
        if plot:
            # Plot what you desire here. This is a plot of H vs num of efolds.

            plt.plot(self.b, H)
            plt.xlabel("Number of Efolds")
            plt.ylabel("H")
            plt.show()

            # This is code that plots phi vs num of efolds and labels the points at which inflation occurs.

            # plt.plot(self.b, self.phi)
            # mask = np.where(4 * (self.psi**2 / 2) - 2 * self.V(self.phi) < 0, True, False)
            # binf = self.b[mask]
            # phiinf = self.phi[mask]
            # plt.plot(binf, phiinf, 'go', markersize=2)
            # plt.plot(self.b, np.zeros(len(self.b)))
            # plt.xlabel("Number of Efolds")
            # plt.ylabel("phi")
            # plt.show()


    def get_vk(self, plot_a=False, plot=False):
        # This method computes Delta_R^2 = Delta_s^2 for a given k

        # self.get_a_phi(plot_a)
        H = np.sqrt(self.V(self.phi) + self.psi**2 / 2)
        z = np.exp(self.b) * self.psi / H

        # computation of z''/z. We create an array, x, with values of z''/z that we use to create a function that interpolates
        #   between elements of the array.
        Hprime = np.gradient(H, self.b)
        zprime = np.gradient(z, self.b)
        zdoubleprime = np.gradient(zprime, self.b)
        x = np.exp(2 * self.b) * H * (zprime * H + Hprime * zprime + zdoubleprime * H) / z

        # We restrict our Mukhanov-Sasaki solver to the range specified in update_k, as we will explain in the guide
        bsmall = self.b[self.lowerrange:self.upperrange]

        tmp = odeint(self.diffs2, self.mukhanov_params_re, bsmall, args=(x, H, self.k))
        vk_re = tmp[:, 0]
        dvk_re = tmp[:, 1]

        tmp = odeint(self.diffs2, self.mukhanov_params_im, bsmall, args=(x, H, self.k))
        vk_im = tmp[:, 0]
        dvk_im = tmp[:, 1]

        # index of horizon crossing
        i0 = (np.abs(H[self.lowerrange:self.upperrange] * np.exp(bsmall) - self.k)).argmin()
        # print bsmall[i0]
        psi_v = np.sqrt(vk_im**2 + vk_re**2) / np.exp(bsmall)
        R = H[self.lowerrange:self.upperrange] * psi_v / self.psi[self.lowerrange:self.upperrange]
        PR = R**2
        DeltaR = PR * self.k**3 / (2 * math.pi**2)

        if plot:
            # plot of vk in complex plane
            plt.plot(vk_re, vk_im)
            plt.show()

            # Plot that shows the minimum of Delta_R^2
            plt.plot(bsmall, DeltaR, 'o')
            plt.axvline(x=bsmall[i0])
            plt.xlabel("loga")
            plt.ylabel("DeltaR^2")
            plt.show()

        # We return the minimum of DeltaR rather than DeltaR[i0] for reasons we will explain in the guide
        return min(DeltaR)

    def get_vt(self, plot_a=False, plot=False):
        # This method computes Delta_t^2 = 2 Delta_h^2 for a given k

        # self.get_a_phi(plot_a)

        # computation of a''/a. We create an array, y, with values of a''/a that we use to create a function that interpolates
        #   between elements of the array.
        H = np.sqrt(self.V(self.phi) + self.psi**2 / 2)
        Hprime = np.gradient(H, self.b)
        y = H * np.exp(2 * self.b) * (Hprime + 2 * H)

        # We restrict our Mukhanov-Sasaki solver to the range specified in update_k, as we will explain in the guide
        bsmall = self.b[self.lowerrange:self.upperrange]

        tmp = odeint(self.diffs3, self.mukhanov_params_re, bsmall, args=(y, H, self.k))
        vk_re = tmp[:, 0]
        dvk_re = tmp[:, 1]

        tmp = odeint(self.diffs3, self.mukhanov_params_im, bsmall, args=(y, H, self.k))
        vk_im = tmp[:, 0]
        dvk_im = tmp[:, 1]

        # index of horizon crossing
        i0 = (np.abs(H[self.lowerrange:self.upperrange] * np.exp(bsmall) - self.k)).argmin()

        psi_v = np.sqrt(vk_im**2 + vk_re**2) / np.exp(bsmall)
        h = 2 * psi_v
        Pt = 2 * h**2
        Deltat = Pt * self.k**3 / (2 * math.pi**2)

        if plot:
            # plot of vk in complex plane
            plt.plot(vk_im, vk_re)
            plt.show()

            # Plot that shows the minimum of Deltat
            plt.plot(bsmall, Deltat, 'o')
            plt.axvline(x=bsmall[i0])
            plt.xlabel("loga")
            plt.ylabel("Deltat^2")
            plt.show()

        # We return the minimum of DeltaR rather than DeltaR[i0] for reasons we will explain in the guide
        return min(Deltat)

    # Interpolation function for z''/z and a''/a
    def generic_interp(self, b0, x):
        f = interp(self.b, x, kind='linear')
        if b0 > max(self.b):
            return f(max(self.b))
        elif b0 < min(self.b):
            return f(min(self.b))
        return f(b0)

    # Interpolation function for e^(log a) H
    def elogaH(self, b0, H):
        f = interp(self.b, H * np.exp(self.b), kind='linear')
        if b0 > max(self.b):
            return f(max(self.b))
        elif b0 < min(self.b):
            return f(min(self.b))
        return f(b0)

    def diffs2(self, V, tau, x, H, k):
        # returns [v', v''] so odeint can compute [v, v']
        # differential equation for v_k
        return [V[1] / self.elogaH(tau, H),
                -(k**2 - self.generic_interp(tau, x)) * V[0] / (self.elogaH(tau, H))]

    def diffs3(self, V, tau, y, H, k):
        # returns [v', v''] so odeint can compute [v, v']
        # differential equation for v^s_k
        return [V[1] / self.elogaH(tau, H),
                -(k**2 - self.generic_interp(tau, y)) * V[0] / (self.elogaH(tau, H))]


################################################################################


# computation of power spectra

def V(phi):
    return 0.1 * phi**(2 / 3)


def dV(phi):
    return 0.1 * 2 / 3 * phi**(2 / 3 - 1)


t1 = time.time()
nk = 11
# We choose k that exit close to loga = 0, i.e.,  close to nstar efolds before the end of inflation
logk = np.linspace(-2, 3, nk)
deltas23 = np.zeros(nk)
deltat23 = np.zeros(nk)
model = InfModel(V, dV, 100, phi0=30, nstar=50, efold=676)
model.get_a_phi(False)
for s in range(nk):
    k = math.exp(logk[s])
    model.update_k(k)
    deltas23[s] = model.get_vk(False, False)
    deltat23[s] = model.get_vt(False, False)
    if s % 1 == 0:
        print s

np.save("deltas23_50", deltas23)
np.save("deltat23_50", deltat23)


def V(phi):
    return 0.1 * phi**(1)


def dV(phi):
    return 0.1 * 1 * phi**(1 - 1)


deltas1 = np.zeros(nk)
deltat1 = np.zeros(nk)
model = InfModel(V, dV, k, phi0=30, nstar=50, efold=450)
model.get_a_phi(False)
for s in range(nk):
    k = math.exp(logk[s])
    model.update_k(k)
    deltas1[s] = model.get_vk(False, False)
    deltat1[s] = model.get_vt(False, False)

np.save("deltas1_50", deltas1)
np.save("deltat1_50", deltat1)


def V(phi):
    return 0.1 * phi**(4 / 3)


def dV(phi):
    return 0.1 * 4 / 3 * phi**(4 / 3 - 1)


deltas43 = np.zeros(nk)
deltat43 = np.zeros(nk)
model = InfModel(V, dV, k, phi0=30, nstar=50, efold=338)
model.get_a_phi(False)
for s in range(nk):
    k = math.exp(logk[s])
    model.update_k(k)
    deltas43[s] = model.get_vk(False, False)
    deltat43[s] = model.get_vt(False, False)

np.save("deltas43_50", deltas43)
np.save("deltat43_50", deltat43)


def V(phi):
    return 0.1 * phi**(2)


def dV(phi):
    return 0.1 * 2 * phi**(2 - 1)


deltas2 = np.zeros(nk)
deltat2 = np.zeros(nk)
model = InfModel(V, dV, 100, phi0=30, nstar=50, efold=225)
model.get_a_phi(False)
for s in range(nk):
    k = math.exp(logk[s])
    model.update_k(k)
    deltas2[s] = model.get_vk(False, False)
    deltat2[s] = model.get_vt(False, False)
    if s % 10 == 0:
        print s

np.save("deltas2_50", deltas2)
np.save("deltat2_50", deltat2)

t2 = time.time
print "time", str((t2 - t1) / 60.0)
