#!/usr/bin/env python

import time
import math
import numpy as np
import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
from scipy.integrate import odeint
from scipy.misc import derivative
from scipy.interpolate import interp1d as interp


class InfModel:
    """All time-dependent quantities in this class are in terms of conformal time!"""
    def __init__(self, V, dV, k, phi0=30, psi0=-0.00000001, a0=math.exp(-626), efold=676, niter=8000, t0=0):
        self.V = V
        self.dV = dV
        self.init_params = [float(phi0), float(psi0)]  # solve for b(t) = log(a(t)) instead of a(t) as before
        self.k = k
        self.t0 = t0
        self.a0 = a0
        self.efold = efold
        self.niter = int(niter * efold / 200)

    def update_k(self, k):
        self.k = k
        self.v0_re = math.cos(self.k * self.t0) / math.sqrt(2 * self.k)
        self.v0_im = -math.sin(self.k * self.t0) / math.sqrt(2 * self.k)
        self.dv0_re = -math.sin(self.k * self.t0) * math.sqrt(self.k / 2)
        self.dv0_im = -math.cos(self.k * self.t0) * math.sqrt(self.k / 2)
        self.mukhanov_params_re = [self.v0_re, self.dv0_re]
        self.mukhanov_params_im = [self.v0_im, self.dv0_im]
        self.lowerrange = int(((self.efold - 54 + math.log(self.k)) / self.efold) * self.niter)
        self.upperrange = int(((self.efold - 44 + math.log(self.k)) / self.efold) * self.niter)

    def diffs1(self, U, b):
        # returns a list of functions f'(tau) on which we wish to compute f(tau)
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
        
        print H[abs(self.b + 10).argmin()]
        if plot:
            # plt.plot(self.t, np.exp(self.b))
            # plt.xlabel("Conformal Time")
            # plt.ylabel("Scale Factor")
            # plt.show()
            plt.clf()
            # Plot where inflation is happening!
            plt.plot(self.b, self.phi)
            plt.title(r"$V(\phi) = \phi^{2}$")
            mask = np.where(4 * (self.psi**2 / 2) - 2 * self.V(self.phi) < 0, True, False)
            binf = self.b[mask]
            phiinf = self.phi[mask]
            plt.plot(binf, phiinf, 'go', markersize=2)
            # plt.plot(self.b, np.zeros(len(self.b)))
            plt.xlabel("Number of efolds")
            plt.ylabel(r"$\phi$")
            plt.show()

            # plt.plot(self.b, self.psi)
            # plt.xlabel("N efolds")
            # plt.ylabel("Phi'")
            # plt.show()

    def get_vk(self, plot_a=False, plot=False):
        # self.get_a_phi(plot_a)
        H = np.sqrt(self.V(self.phi) + self.psi**2 / 2)
        z = np.exp(self.b) * self.psi / H

        Hprime = np.gradient(H, self.b)
        zprime = np.gradient(z, self.b)
        zdoubleprime = np.gradient(zprime, self.b)
        x = np.exp(2 * self.b) * H * (zprime * H + Hprime * zprime + zdoubleprime * H) / z

        bsmall = self.b[self.lowerrange:self.upperrange]
        tmp = odeint(self.diffs2, self.mukhanov_params_re, bsmall, args=(x, H, self.k))
        vk_re = tmp[:, 0]
        dvk_re = tmp[:, 1]

        tmp = odeint(self.diffs2, self.mukhanov_params_im, bsmall, args=(x, H, self.k))
        vk_im = tmp[:, 0]
        dvk_im = tmp[:, 1]

        i0 = (np.abs(H[self.lowerrange:self.upperrange] * np.exp(bsmall) - self.k)).argmin()
        #print bsmall[i0]
        psi_v = np.sqrt(vk_im**2 + vk_re**2) / np.exp(bsmall)
        R = H[self.lowerrange:self.upperrange] * psi_v / self.psi[self.lowerrange:self.upperrange]
        PR = R**2
        DeltaR = PR * self.k**3 / (2 * math.pi**2)

        # #  RR = (H * v / dphi / a)**2 at horizon crossing
        # i0 = (np.abs(H * self.a - self.k)).argmin()
        # print self.t[i0]
        # # print("D^2_R(k=" + str(k) + ") = " + str((H[i0] * vk[i0] / self.dphi[i0] / self.a[i0])**2))
        #
        if plot:
            # plt.plot(vk_re, vk_im)
            plt.plot(vk_re, vk_im)
            plt.xlabel(r"$Re[v_k]$")
            plt.ylabel(r"$Im[v_k]$")
            plt.xlim((-2, 2))
            plt.ylim((-2, 2))
            plt.show()
            # plt.plot(bsmall, DeltaR, 'o')
            # plt.axvline(x=bsmall[i0])
            # plt.xlabel("loga")
            # plt.ylabel("DeltaR^2")
            # plt.show()
            #     plt.plot(self.b, np.zeros(len(self.b)))
            #     # plt.plot(self.t, vk_re**2 + vk_im**2)
            #     # plt.plot(vk_re, vk_im)
            #     plt.xlabel("Conformal Time")
            #     plt.ylabel("vk")

        return min(DeltaR)

    def get_vt(self, plot_a=False, plot=False):
        # self.get_a_phi(plot_a)
        # t1 = time.time()
        H = np.sqrt(self.V(self.phi) + self.psi**2 / 2)
        Hprime = np.gradient(H, self.b)
        y = H * np.exp(2 * self.b) * (Hprime + 2 * H)
        bsmall = self.b[self.lowerrange:self.upperrange]
        tmp = odeint(self.diffs3, self.mukhanov_params_re, bsmall, args=(y, H, self.k))
        vk_re = tmp[:, 0]
        dvk_re = tmp[:, 1]

        tmp = odeint(self.diffs3, self.mukhanov_params_im, bsmall, args=(y, H, self.k))
        vk_im = tmp[:, 0]
        dvk_im = tmp[:, 1]

        # #  RR = (H * v / dphi / a)**2 at horizon crossing
        i0 = (np.abs(H[self.lowerrange:self.upperrange] * np.exp(bsmall) - self.k)).argmin()
        psi_v = np.sqrt(vk_im**2 + vk_re**2) / np.exp(bsmall)
        h = 2 * psi_v
        Pt = 2 * h**2
        Deltat = Pt * self.k**3 / (2 * math.pi**2)
        # t2 = time.time()

        # print "runtime (mins)", (t2 - t1) / 60.0
        # # print("D^2_R(k=" + str(k) + ") = " + str((H[i0] * vk[i0] / self.dphi[i0] / self.a[i0])**2))
        #
        if plot:
            #plt.plot(vk_im, vk_re)
            # plt.plot(self.b, H)
        #     plt.plot(self.b, np.zeros(len(self.b)))
        #     # plt.plot(self.t, vk_re**2 + vk_im**2)
        #     # plt.plot(vk_re, vk_im)
        #     plt.xlabel("Conformal Time")
        #     plt.ylabel("vk")
            # plt.show()
            plt.plot(bsmall, Deltat, 'o')
            plt.axvline(x=bsmall[i0])
            plt.xlabel("loga")
            plt.ylabel("Deltat^2")
            plt.show()
        return min(Deltat)

    def generic_interp(self, b0, x):
        f = interp(self.b, x, kind='linear')
        if b0 > max(self.b):
            return f(max(self.b))
        elif b0 < min(self.b):
            return f(min(self.b))
        return f(b0)


    # def zdoubleprimebyz(self, b0, x):
    #     # x is the array of z'' / z
    #     # b0 is a single value of log a
    #     if b0 <= max(self.b) and b0 >= min(self.b):
    #         return x[(b0 - min(self.b)) / (max(self.b) - min(self.b)) * self.niter]
    #     elif b0 < min(self.b):
    #         return x[0]
    #     else:
    #         return x[-1]

    def elogaH(self, b0, H):
        f = interp(self.b, H * np.exp(self.b), kind='linear')
        if b0 > max(self.b):
            return f(max(self.b))
        elif b0 < min(self.b):
            return f(min(self.b))
        return f(b0)

    def diffs2(self, V, tau, x, H, k):
        # returns [v', v''] so odeint can comptute [v, v']
        return [V[1] / self.elogaH(tau, H),
                -(k**2 - self.generic_interp(tau, x)) * V[0] / (self.elogaH(tau, H))]

    def diffs3(self, V, tau, y, H, k):
        # returns [v', v''] so odeint can comptute [v, v']
        return [V[1] / self.elogaH(tau, H),
                -(k**2 - self.generic_interp(tau, y)) * V[0] / (self.elogaH(tau, H))]


################################################################################


def V(phi):
    # return np.exp(2 * phi)
    return 0.1 * phi**(2 / 3)


def dV(phi):
    return 0.1 * 2 / 3 * phi**(2 / 3 - 1)


t1 = time.time()
nk = 11
logk = np.linspace(-2, 3, nk)
deltas23 = np.zeros(nk)
deltat23 = np.zeros(nk)
model = InfModel(V, dV, 100, phi0=30, a0=math.exp(-626), efold=676)
model.get_a_phi(False)
# for s in range(nk):
# k = math.exp(logk[s])
model.update_k(0.31640304852404455)
deltas23[0] = model.get_vk(False, True)
# deltat23[s] = model.get_vt(False, False)


# np.save("deltas23_50", deltas23)
# np.save("deltat23_50", deltat23)


def V(phi):
    # return np.exp(2 * phi)
    return 0.1 * phi**(1)


def dV(phi):
    return 0.1 * 1 * phi**(1 - 1)


# deltas1 = np.zeros(nk)
# deltat1 = np.zeros(nk)
# model = InfModel(V, dV, 100, phi0=30, a0=math.exp(0), efold=450)
# model.get_a_phi(True)
# model.get_vk(False, True)
# # for s in range(nk):
# #     k = math.exp(logk[s])
# #     model.update_k(k)
# #     deltas1[s] = model.get_vk(False, False)
# #     deltat1[s] = model.get_vt(False, False)

# # np.save("deltas1_50", deltas1)
# # np.save("deltat1_50", deltat1)


def V(phi):
    # return np.exp(2 * phi)
    return 0.1 * phi**(4 / 3)


def dV(phi):
    return 0.1 * 4 / 3 * phi**(4 / 3 - 1)


# deltas43 = np.zeros(nk)
# deltat43 = np.zeros(nk)
# model = InfModel(V, dV, 100, phi0=30, a0=math.exp(0), efold=338)
# model.get_a_phi(True)
# # for s in range(nk):
# #     k = math.exp(logk[s])
# #     model.update_k(k)
# #     deltas43[s] = model.get_vk(False, False)
# #     deltat43[s] = model.get_vt(False, False)

# # np.save("deltas43_50", deltas43)
# # np.save("deltat43_50", deltat43)


def V(phi):
    # return np.exp(2 * phi)
    return 0.1 * phi**(2)


def dV(phi):
    return 0.1 * 2 * phi**(2 - 1)


# deltas2 = np.zeros(nk)
# deltat2 = np.zeros(nk)
# model = InfModel(V, dV, 100, phi0=30, a0=math.exp(0), efold=225)
# model.get_a_phi(True)
# # for s in range(nk):
# #     k = math.exp(logk[s])
# #     model.update_k(k)
# #     deltas2[s] = model.get_vk(False, False)
# #     deltat2[s] = model.get_vt(False, False)
# #     if s % 10 == 0:
# #         print s

# # np.save("deltas2_50", deltas2)
# # np.save("deltat2_50", deltat2)

# # t2 = time.time
# # print "time", str((t2 - t1) / 60.0)
