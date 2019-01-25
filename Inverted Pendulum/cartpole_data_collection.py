import h5py
import matplotlib.pyplot as plt
import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import scipy.sparse as sps
from scipy.interpolate import interp1d
import random


class cartpole:
    def __init__(self,m,M,L,g,d):
        self.m = m  # mass of the pole
        self.M = M  # mass of the cart
        self.L = L  # length of the pole
        self.g = g  # gravity constant, usually g = 10
        self.d = d  # damping coefficient

    def input(self,t):
        return input(t)

    def dynamics(self, x, t):
        dx = np.empty_like(x)
        Sx = np.sin(x[2])
        Cx = np.cos(x[2])
        D = self.M + self.m*Sx**2
        dx[0] = x[1]
        dx[1] = 1/D*(-self.m*self.g*Sx*Cx + self.m*self.L*x[2]**2*Sx - self.d*x[1] + self.input(t))
        dx[2] = x[3]
        dx[3] = 1/D/self.L*((self.m + self.M)*self.g*Sx - self.m*self.L*x[2]**2*Sx*Cx + Cx*self.d*x[1] - Cx*self.input(t))

        return dx

def u(t):
    return 0*t

def random_sample(InvPendulum, numTraj, sampling_rate, A, ts):
    x = np.zeros((1,4))
    y = np.zeros((1,4))
    for i in range(numTraj):

        x0 = (np.random.rand(4) - 0.5)*A

        traj = odeint(InvPendulum.dynamics, x0, ts)

        numSamples = traj.shape[0]
        input_index = random.sample(range(numSamples-1), int(numSamples*sampling_rate))
        output_index = [xx + 1 for xx in input_index]

        input_slice = traj[input_index, :]
        output_slice = traj[output_index, :]

        x = np.concatenate((x, input_slice), axis = 0)
        y = np.concatenate((y, output_slice), axis = 0)

    x = x[1:, :]
    y = y[1:, :]

    return x, y

def trajectory_sample(InvPendulum, A, ts):

    x0 = (np.random.rand(4) - 0.5) * A
    traj = odeint(InvPendulum.dynamics, x0, ts)

    x = traj[0:-1, :]
    y = traj[1:, :]

    return x, y

if __name__ == '__main__':
    m = 1
    M = 5
    L = 2
    g = -10
    d = 2

    A = 5  # input is uniformly sampled from [-A, A]

    T = 30.0 # End time
    Ts = 0.05 # Sampling time
    ts = np.arange(0,T,Ts) # Time span

    sampling_rate = 0.2 # The ratio of samples kept from a sampled trajectory
    numTraj = 50

    InvPendulum = cartpole(m, M, L, g, d)
    InvPendulum.input = u

    useq = u(ts)

    x, y = random_sample(InvPendulum, numTraj, sampling_rate, A, ts)

    x_traj, y_traj = trajectory_sample(InvPendulum, A, ts)

    f = h5py.File("cartpoledata/autosysdata_train2.h5", "w")
    dset = f.create_dataset("x", data = x)
    dset = f.create_dataset("y", data = y)
    dset = f.create_dataset("x_traj", data=x_traj)
    dset = f.create_dataset("y_traj", data=y_traj)
    dset = f.create_dataset("u", data = useq)
    dset = f.create_dataset("T", data=T)
    dset = f.create_dataset("Ts", data=Ts)
    f.close()



