import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam
from autograd.misc.optimizers import sgd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import random


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    # Build a list of (weights, biases) tuples, one for each layer.
    return [(rs.randn(insize, outsize) * scale,  # weight matrix
             rs.randn(outsize) * scale)  # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]

def nn_predict(params, inputs, nonlinearity = np.tanh):
    # Change the activation function to ReLU by setting nonlinearity = np.maximum
    # and inputs = nonlinearity(outputs, 0) in this function.
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs

def nn_linear(W, inputs):
    outputs = np.dot(inputs, W)
    return outputs

def nn_encode(weights, inputs):
    encoded = nn_predict(weights['enc'], inputs)
    return encoded

def nn_forward(weights, encoded):
    forwarded = nn_linear(weights['koop'][0][0], encoded)
    return forwarded

def nn_decode(weights, outputs):
    decoded = nn_predict(weights['dec'], outputs)
    return decoded

def nn_encode_decode(weights, inputs):
    encoded = nn_encode(weights, inputs)
    decoded = nn_predict(weights['dec'], encoded)
    return decoded


def nn_encode_forward(weights, inputs):
    encoded = nn_encode(weights, inputs)
    forwarded = nn_forward(weights, encoded)
    return forwarded

def nn_encode_foward_decode(weights, inputs):
    forwarded = nn_encode_forward(weights, inputs)
    decoded_forward = nn_decode(weights, forwarded)
    return decoded_forward

def initialize_weights(G, D):
    init_scale = np.sqrt(2/(G*D)) # By heuristics of setting layer's width
    init_weights_enc = init_random_params(init_scale, layer_sizes=[D, G * D, G * D])
    init_weights_dec = init_random_params(init_scale, layer_sizes=[G * D, G * D, D])
    init_weights_koop = init_random_params(init_scale, layer_sizes=[G * D , G * D ])
    init_weights = {'enc': init_weights_enc, 'dec': init_weights_dec, 'koop': init_weights_koop}

    return init_weights

def logprob_koop(weights, inputs, targets, noise_scale=0.1):

    decoded = nn_encode_decode(weights, inputs)
    decoded_forward = nn_encode_foward_decode(weights, inputs)

    encoded_targets = nn_encode(weights, targets)
    encoded_forward = nn_encode_forward(weights, inputs)

    # Loglikelihood error
    N = inputs.size
    t1 = np.sum(norm.logpdf(decoded, inputs, noise_scale))
    t2 = np.sum(norm.logpdf(decoded_forward, targets, noise_scale))
    t3 = np.sum(norm.logpdf(encoded_forward, encoded_targets, noise_scale))

    return -(t1 + t2 + t3)/N

def MSE(weights, inputs, targets):

    decoded = nn_encode_decode(weights, inputs)
    decoded_forward = nn_encode_foward_decode(weights, inputs)

    encoded_targets = nn_encode(weights, targets)
    encoded_forward = nn_encode_forward(weights, inputs)

    # Mean square error
    N = inputs.size
    t1 = np.sum(np.square(decoded - inputs))
    t2 = np.sum(np.square(decoded_forward - targets))
    t3 = np.sum(np.square(encoded_forward - encoded_targets))

    return (t1 + t2 + t3)/N


def objective(weights, t):
    return MSE(weights, inputs, targets)

def callback(params, t, g):
    print("Iteration {:3d} log likelihood {:+1.3e}".format(t, objective(params, t)))
    # # save the training error in each iteration.
    # training_error.append(objective(params, t))
    # np.savez('data/sample_1/training_error_adam_stepsize0dot001.npz', training_error = training_error)


def figplot(outputs, url = None):
    # Plot and compare outputs and targets in all four variables.

    fig = plt.figure(figsize=(12,8))
    plt.subplot(221)
    _ = plt.plot(outputs[:, 0])
    _ = plt.plot(targets[:, 0])
    plt.title('Trajectory prediction of x1',fontdict={'fontsize': 12, 'fontweight': 'medium'})

    plt.subplot(222)
    _ = plt.plot(outputs[:, 1])
    _ = plt.plot(targets[:, 1])
    plt.title('Trajectory prediction of x2',fontdict={'fontsize': 12, 'fontweight': 'medium'})

    plt.subplot(223)
    _ = plt.plot(outputs[:, 2])
    _ = plt.plot(targets[:, 2])
    plt.title('Trajectory prediction of x3',fontdict={'fontsize': 12, 'fontweight': 'medium'})

    plt.subplot(224)
    _ = plt.plot(outputs[:, 3])
    _ = plt.plot(targets[:, 3])
    plt.title('Trajectory prediction of x4',fontdict={'fontsize': 12, 'fontweight': 'medium'})

    if url is None:
        plt.show()
    else:
        fig.savefig(url)


def shuffle(x,y):
    # Shuffle the x, y pairs
    N = x.shape[0]
    index = list(range(N))
    random.shuffle(index)
    x_shuffled = x[index,:]
    y_shuffled = y[index,:]
    return x_shuffled, y_shuffled

def sample_multitraj(start, end, url = None):
    # Extract multiple trajectories from start to end. If url is given, extract the samples from url.
    x = np.zeros((1,4))
    y = np.zeros((1,4))

    for j in range(start, end):
        if url is None:
            url_load = 'data/sample_1/traj_' + str(j) + '.npz'
        else:
            url_load = url

        trajdata = np.load(url_load)
        x_traj = trajdata['x']
        y_traj = trajdata['y']

        x = np.concatenate((x, x_traj), axis = 0)
        y = np.concatenate((y, y_traj), axis = 0)

    x = x[1:,:]
    y = y[1:,:]
    return x,y

def randomsample(N, num = 8000):
    # Randomly extract N trajectories from num total trajectories.
    index = random.sample(list(range(num)), N)

    x = np.zeros((1,4))
    y = np.zeros((1,4))

    for j in index:
        xj, yj = sample_multitraj(j, j+1)
        x = np.concatenate((x, xj), axis = 0)
        y = np.concatenate((y, yj), axis = 0)

    return x[1:,:], y[1:, :], index

def shufflesample(N, sampling_rate = 0.1):
    # Randomly sample sampling_rate*length_of_trajectory data points from N trajectories.
    index = random.sample(list(range(8000)), N)

    x = np.zeros((1,4))
    y = np.zeros((1,4))

    for j in index:
        xj, yj = sample_multitraj(j, j+1)

        len_traj = xj.shape[0]
        input_index = random.sample(range(len_traj), int(len_traj*sampling_rate))

        input_slice = xj[input_index, :]
        output_slice = yj[input_index, :]

        x = np.concatenate((x, input_slice), axis = 0)
        y = np.concatenate((y, output_slice), axis = 0)

    return x[1:,:], y[1:, :], index


#############################################
### functions for processing relative errors
#############################################

def re_processing(url_load, url_re):
    # url_load gives the url for loading the optimal weight of NN.
    # url_re gives the url for saving the computed relative errors.
    global inputs, targets
    data = np.load(url_load)
    opt_weights = data['optweights'].item()
    x_scaler = data['x_scaler'].item()
    y_scaler = data['y_scaler'].item()

    RelativeError = []
    for datafile in range(8000):
        x_traj_test, y_traj_test = sample_multitraj(datafile, datafile+1)

        inputs = x_scaler.transform(x_traj_test)
        targets = y_scaler.transform(y_traj_test)
        outputs = nn_encode_foward_decode(opt_weights, inputs)

        outputs = y_scaler.inverse_transform(outputs)
        targets = y_scaler.inverse_transform(targets)

        re = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])

        print('sample {:d} relative training norm error {:+1.4e}'.format(datafile, re))
        RelativeError.append(re)
        np.savez(url_re, re = RelativeError, url_load = url_load)


def init_condition(url_init, num = 8000):
    # Extract and save the initial conditions from num trajecotries
    x = np.zeros((1,4))
    print('Initial conditions collection started.')
    for j in range(num):
        x_traj_test, y_traj_test = sample_multitraj(j, j+1)
        x0 = x_traj_test[0,:].reshape((1,4))
        x = np.concatenate((x, x0), axis=0)

    print('Initial conditions collection finished.')
    x = x[1:,:]
    np.savez(url_init, x0 = x)

def re_scatter(url_re, url_init, threshold = 0.05):
    # Scatter plot the relative errors given by the prediction of neural networks.
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d', num = 8000)

    redata = np.load(url_re)
    re = redata['re']

    initdata = np.load(url_init)
    initconditions = initdata['x0']

    xs = initconditions[:, 1]
    ys = initconditions[:, 2]
    zs = initconditions[:, 3]

    sup_index = re > threshold
    re[sup_index] = 1

    sub_index = re <= threshold
    re[sub_index] = 0

    # rate gives the ratio of well fitted trajectories.
    rate = (num - np.count_nonzero(re))/num
    print('Rate = {:+1.3e}'.format(rate))

    color = re
    p = ax.scatter(xs, ys, zs, c=color, cmap='viridis')
    fig.colorbar(p)

    ax.set_xlabel('x2')
    ax.set_ylabel('x3')
    ax.set_zlabel('x4')

    plt.show()

################################
## Simulation starts here.
################################

def run():
    # train and save the neural network
    global inputs, targets, training_error

    training_error = []

    # max number of iterations in optimization
    num_iters = 100

    N = 100 # Number of uniformly sampled trajectories in training data set.

    # sample training data
    # x_traj, y_traj, index = randomsample(N)
    x_traj, y_traj, index = shufflesample(N*10, sampling_rate = 0.1)

    # normalize the training data
    x_scaler = MinMaxScaler((-1,1))
    x_scaler.fit(x_traj)
    y_scaler = MinMaxScaler((-1,1))
    y_scaler.fit(y_traj)

    x_traj_scale = x_scaler.transform(x_traj)
    y_traj_scale = y_scaler.transform(y_traj)

    inputs = x_traj_scale
    targets = y_traj_scale

    # Decide NN architecture
    D = x_traj.shape[1]
    G = 20

    init_weights = initialize_weights(G, D)

    print('----------  Optimizing KOOPMAN NEURAL NET for {} iterations ..... \n'.format(num_iters))
    # use adam to optimize
    opt_weights = adam(grad(objective), init_weights, step_size=0.01, num_iters = num_iters, callback=callback)

    # use sgd to optimize
    # opt_weights = sgd(grad(objective), init_weights, step_size=0.1, num_iters = num_iters, callback=callback)

    print('done')

    # save the optimal weights and related parameters
    np.savez('data/sample_1/optweights_tanh_minmax_random1000shuffle_G20_layer2_sgd_2.npz', optweights = opt_weights, x_scaler = x_scaler, y_scaler = y_scaler, index = index, training_error = training_error)

    # Pick a trajectory and check the prediction of the nn on this trajectory

    x_traj_test, y_traj_test = sample_multitraj(6350, 6351)
    inputs = x_scaler.transform(x_traj_test)
    targets = y_scaler.transform(y_traj_test)
    outputs = nn_encode_foward_decode(opt_weights, inputs)
    re = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])
    print('Relative training norm error {:+1.4e}'.format(re))

    figplot(outputs, url = None )

def test():
    # Pick a trajectory and check the prediction of the nn on this trajectory
    global inputs, targets

    data = np.load('data/sample_1/optweights_tanh_minmax_random1000shuffle_G20_layer2_2.npz')
    opt_weights = data['optweights'].item()
    x_scaler = data['x_scaler'].item()
    y_scaler = data['y_scaler'].item()

    # the number of the trajectory picked
    datafile = 4412
    x_traj_test, y_traj_test = sample_multitraj(datafile, datafile+1)

    print(x_traj_test[0,:])

    inputs = x_scaler.transform(x_traj_test)
    targets = y_scaler.transform(y_traj_test)
    outputs = nn_encode_foward_decode(opt_weights, inputs)

    x0 = inputs[0,:].reshape((1,4))

    outputs = np.concatenate((x0, outputs), axis = 0)
    targets = np.concatenate((x0, targets), axis = 0)

    outputs = y_scaler.inverse_transform(outputs)
    targets = y_scaler.inverse_transform(targets)

    re = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])
    # re = np.mean([np.linalg.norm(targets[i] - outputs[i])**2 for i in range(len(targets))])

    print('sample Relative training norm error {:+1.4e}'.format(re))
    figplot(outputs, url = None)


def plot_re():
    # generate the relative errors of prediction by a given nn on all of the trajectories
    url_init = 'data/sample_1/init_conditions.npz'
    # init_condition(url_init)
    url_load = 'data/sample_1/optweights_tanh_minmax_random1000shuffle_G20_layer2_2.npz'
    url_re = 'data/sample_1/re_optweights_tanh_minmax_random1000shuffle_G20_layer2_2.npz'
    # re_processing(url_load, url_re)
    re_scatter(url_re, url_init, threshold=0.05)


if __name__ == '__main__':
    # run()
    # test()
    plot_re()