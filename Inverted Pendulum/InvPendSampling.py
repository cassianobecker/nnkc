import numpy as np
import numpy.random as npr
from scipy.integrate import odeint
import scipy.sparse as sps
from scipy.interpolate import interp1d
import random
import matplotlib.pyplot as plt

class cartpole_par:
    # Parameters of the cartpole system
    m = 1
    M = 5
    L = 2
    g = -10
    d = 2

class cartpole:
    # Cart-pole model
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
        dx[1] = 1/D*(-self.m*self.g*Sx*Cx + self.m*self.L*x[3]**2*Sx - self.d*x[1] + self.input(t))
        dx[2] = x[3]
        dx[3] = 1/D/self.L*((self.m + self.M)*self.g*Sx - self.m*self.L*x[3]**2*Sx*Cx + Cx*self.d*x[1] - Cx*self.input(t))

        return dx

def control(t):
    # The control input to the system
    return 0*t

def create_cartpole(input = None):
    # Create a cartpole instance
    if input is None:
        input = lambda t: 0*t
    paras = cartpole_par()
    InvPendulum = cartpole(paras.m, paras.M, paras.L, paras.g, paras.d)
    InvPendulum.input = input
    return InvPendulum

def create_initpoints(v, angle, angular_v, N1, N2, N3, N4):
    # Do a grid sampling of initial conditions. N1 ~ N4 are number of samples in each coordinate.
    x0 = 0
    x1 = np.linspace(v[0], v[1], N2, endpoint = False)
    x2 = np.linspace(angle[0], angle[1], N3, endpoint = False)
    x3 = np.linspace(angular_v[0], angular_v[1], N4, endpoint = False)

    initpoints = np.zeros((1,4))
    for x_1 in x1:
        for x_2 in x2:
             for x_3 in x3:
                 point = np.array([x0, x_1, x_2, x_3]).reshape((1, 4))
                 initpoints = np.concatenate((initpoints, point), axis = 0)

    return initpoints[1:,:]


def trajectory_sample(InvPendulum, ts, A = 6, init = None, url = None):
    # Sample a trajectory. If url is given, save the trajectory in url.
    if init is None:
        # If initial point is not given, a random initial point will be generated.
        init = (np.random.rand(4) - 0.5) * A
    traj = odeint(InvPendulum.dynamics, init, ts)

    x = traj[0:-1, :]
    y = traj[1:, :]

    if url is not None:
        np.savez(url, x = x, y = y, x0 = init, ts = ts)
    return x, y

if __name__ == '__main__':
    A = 5

    v = [-6, 6]
    angular_v = [-6, 6]
    angle = [0, 2*np.pi]

    T = 30.0 # End time
    Ts = 0.05 # Sampling time
    ts = np.arange(0,T,Ts) # Time span

    InvPendulum = create_cartpole(control)

    N1, N2, N3, N4 = (1,20,20,20)
    initpoints = create_initpoints(v, angle, angular_v, N1, N2, N3, N4)

    numInit = initpoints.shape[0]

    for j in range(numInit):
        print("Iter {:d} out of {:d}".format(j, numInit))

        # Save the sampled trajectories.
        url = "data/sample_1/traj_" + str(j) + ".npz"
        x, y = trajectory_sample(InvPendulum, ts, init = initpoints[j, :], url = url)

    # Save the parameters of sampling.
    np.savez("data/sample_1/params.npz", model = InvPendulum, num = numInit, v = v, angular_v = angular_v, angle = angle, N1 = N1, N2 = N2, N3 = N3, N4 = N4, T = T, Ts = Ts, ts = ts)

    # Laugh out
    print('hhh')