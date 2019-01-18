import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import scipy.sparse as sps


class WC_par:
    # according to Figure 11 W-C paper
    # Excitatory to excitatory  coupling coefficient
    c_ee = 16.0
    # Inhibitory to excitatory coupling coefficient
    c_ie = 12.0
    # Excitatory to inhibitory coupling coefficient
    c_ei = 15.0
    # Inhibitory to inhibitory coupling coefficient
    c_ii = 3.0
    # Excitatory population, membrane time-constant [ms]
    tau_e = 8.0
    # Inhibitory population, membrane time-constant [ms]
    tau_i = 8.0
    # The slope parameter for the excitatory response function
    a_e = 1.3
    # Excitatory threshold
    theta_e = 4.0
    # The slope parameter for the inhibitory response function
    a_i = 2.0
    # Inhibitory threshold
    theta_i = 3.7
    # refractory period
    se_max = 1.0
    # refractory period
    si_max = 1.0
    # External stimulus to the excitatory population. Constant intensity.Entry point for coupling
    P = 1.25
    # External stimulus to the inhibitory population. Constant intensity.Entry point for coupling.
    Q = 0.0
    alpha_e = 1.2
    alpha_i = 2.0


class WC:

    def dynamics(self, x, t):
        E = x[0]
        I = x[1]

        par = WC_par()

        x_e = par.c_ee * E - par.c_ei * I + par.P
        x_i = par.c_ie * E - par.c_ii * I + par.Q

        s_e = (1.0 / (1.0 + np.exp(-par.a_e * (x_e - par.theta_e)))) - (1 / (1.0 + np.exp(par.a_e * par.theta_e)))
        s_i = (1.0 / (1.0 + np.exp(-par.a_i * (x_i - par.theta_i)))) - (1 / (1.0 + np.exp(par.a_i * par.theta_i)))

        derivative = np.empty_like(x)

        derivative[0] = (-E + (par.se_max - E) * s_e) / par.tau_e
        derivative[1] = (-I + (par.si_max - I) * s_i) / par.tau_i

        return derivative


class WC_net:

    def __init__(self, n, d, W):
        self.n = n
        self.d = d
        self.W = W

    def inputs(self, t):
        mean_E = 0.1
        P_max = 1.25
        P = np.maximum(P_max - np.sum(self.W, axis=1) * mean_E, 0)
        return P

    def dynamics(self, x, t):
        E = x.reshape(self.d, self.n)[0, :]
        I = x.reshape(self.d, self.n)[1, :]

        par = WC_par()

        derivative_E = np.empty_like(E)
        derivative_I = np.empty_like(I)

        for i in range(n):
            x_e = par.c_ee * E[i] - par.c_ei * I[i] + np.dot(W[i, :], E) + self.inputs(t)[i]
            x_i = par.c_ie * E[i] - par.c_ii * I[i] + par.Q

            s_e = (1.0 / (1.0 + np.exp(-par.a_e * (x_e - par.theta_e)))) - (1 / (1.0 + np.exp(par.a_e * par.theta_e)))
            s_i = (1.0 / (1.0 + np.exp(-par.a_i * (x_i - par.theta_i)))) - (1 / (1.0 + np.exp(par.a_i * par.theta_i)))

            derivative_E[i] = (-E[i] + (par.se_max - E[i]) * s_e) / par.tau_e
            derivative_I[i] = (-I[i] + (par.si_max - I[i]) * s_i) / par.tau_i

        return np.hstack((derivative_E, derivative_I))


if __name__ == '__main__':

    d = 2
    n = 5

    N = 20000
    N0 = 500
    T = 20.
    ts = np.linspace(0, N / T, N)

    ws = sps.random(n, n, density=0.3)
    W = ws.todense()

    wc_net = WC_net(n, d, W)
    x0 = npr.random(n * d)

    y = odeint(wc_net.dynamics, x0, ts)

    y = y[N0:, :]

    z = np.stack(np.split(y, 2, axis=1), axis=2)
    z[:, 0, :]

    E_I_diff = (z[:, :, 0] - z[:, :, 1])
    E_I_sum = (z[:, :, 0] + z[:, :, 1])

    z = np.split(z, n, axis=1)
    z = [np.squeeze(z[i]).T for i in range(len(z))]

    h = plt.plot(y[:, 0:5])

    plt.show()

    f = h5py.File("data/wc1.hd5", "w")
    dset = f.create_dataset("y", data=y)
    dset = f.create_dataset("W", data=W)
    f.close()
