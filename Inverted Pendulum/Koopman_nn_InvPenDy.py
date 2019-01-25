import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

import matplotlib.pyplot as plt
import h5py

def build_dataset(url, ver = 0):
    file_url = url
    hf = h5py.File(file_url, 'r+')
    x = np.squeeze(hf['x'])
    u = np.squeeze(hf['u'])
    T = np.squeeze(hf['T'])
    Ts = np.squeeze(hf['Ts'])
    y = np.squeeze(hf['y'])

    if ver == 1:
        x_traj = np.squeeze(hf['x_traj'])
        y_traj = np.squeeze(hf['y_traj'])
        hf.close()
        return x, y, x_traj, y_traj, u, T, Ts
    else:
        hf.close()
        return x, y, u, T, Ts

def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,  # weight matrix
             rs.randn(outsize) * scale)  # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def nn_predict(params, inputs, nonlinearity=np.tanh):
    for W, b in params:
        outputs = np.dot(inputs, W) + b
        inputs = nonlinearity(outputs)
    return outputs

def nn_linear(W, inputs):
    outputs = np.dot(inputs, W)
    # outputs = np.concatenate((outputs, np.asmatrix(inputs[:,-1]).T), axis = 1)
    return outputs

def nn_encode(weights, inputs):
    encoded = nn_predict(weights['enc'], inputs)
    # encoded = np.concatenate((encodedx, np.asmatrix(inputs[:,-1]).T), axis = 1)
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


def initialize_weights(G, Dx):
    init_scale = 0.1
    init_weights_enc = init_random_params(init_scale, layer_sizes=[Dx, G * Dx, G * Dx])
    init_weights_dec = init_random_params(init_scale, layer_sizes=[G * Dx, G * Dx, Dx])
    init_weights_koop = init_random_params(init_scale, layer_sizes=[G * Dx , G * Dx ])
    init_weights = {'enc': init_weights_enc, 'dec': init_weights_dec, 'koop': init_weights_koop}

    return init_weights

def logprob_koop(weights, inputs, targets, noise_scale=0.1):

    decoded = nn_encode_decode(weights, inputs)
    decoded_forward = nn_encode_foward_decode(weights, inputs)

    encoded_targets = nn_encode(weights, targets)
    encoded_forward = nn_encode_forward(weights, inputs)

    t1 = np.sum(norm.logpdf(decoded, inputs, noise_scale))
    t2 = np.sum(norm.logpdf(decoded_forward, targets, noise_scale))
    t3 = np.sum(norm.logpdf(encoded_forward, encoded_targets, noise_scale))

    return t1 + t2 + t3

def objective(weights, t):
    return -logprob_koop(weights, inputs, targets)

def callback(params, t, g):
    print("Iteration {:3d} log likelihood {:+1.3e}".format(t, -objective(params, t)))

def run():
    global inputs, targets, hyper
    num_iters = 2500
    x, y, x_traj_val, y_traj_val, u_val, T_val, Ts_val = build_dataset('cartpoledata/autosysdata_train2.h5', ver = 1)

    # u = np.asmatrix(u).T
    # trajectory = np.concatenate((x,u), axis = 1)

    DoTraining = 1
    DoValidation = 1

    if DoTraining == 1:

        inputs = x
        targets = y

        Dx = x.shape[1]
        G = 20

        init_weights = initialize_weights(G, Dx)


        print('----------  Optimizing KOOPMAN NEURAL NET for {} iterations ..... \n'.format(num_iters))
        opt_weights = adam(grad(objective), init_weights, step_size=0.01, num_iters=num_iters, callback=callback)

        print('done')

        np.savez('cartpoledata/train_results_data2_iter2500.npz', optweights=opt_weights, initweights=init_weights)

        decoded = nn_encode_decode(opt_weights, inputs)
        outputs = nn_encode_foward_decode(opt_weights, inputs)

        plt.figure()
        _ = plt.scatter(targets, outputs, marker='D', c='g', alpha=0.1)
        plt.xlabel('targets')
        plt.ylabel('outputs')
        plt.title('Dynamic Scatter')
        plt.grid()

        plt.figure()
        _ = plt.scatter(inputs, decoded, marker='D', c='b', alpha=0.1)
        plt.xlabel('inputs')
        plt.ylabel('decoded')
        plt.title('Encoding-decoding Scatter')
        plt.grid()

        plt.figure()
        _ = plt.plot(outputs[:, 2])
        _ = plt.plot(targets[:, 2])

        plt.show()

        re = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])

        print('Relative norm error {:+1.4e}'.format(re))

    if DoValidation == 1:
        if DoTraining != 1:
            optdata = np.load('cartpoledata/train_results_data2.npz')
            opt_weights = optdata['optweights'].item()

        inputs = x_traj_val
        targets = y_traj_val

        decoded = nn_encode_decode(opt_weights, inputs)
        outputs = nn_encode_foward_decode(opt_weights, inputs)

        re_val = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])
        print('Relative validation norm error {:+1.4e}'.format(re_val))

        plt.figure()
        _ = plt.scatter(targets, outputs, marker='D', c='g', alpha=0.1)
        plt.xlabel('targets')
        plt.ylabel('outputs')
        plt.title('Dynamic Scatter')
        plt.grid()

        plt.figure()
        _ = plt.scatter(inputs, decoded, marker='D', c='b', alpha=0.1)
        plt.xlabel('inputs')
        plt.ylabel('decoded')
        plt.title('Encoding-decoding Scatter')
        plt.grid()

        plt.figure()
        _ = plt.plot(outputs[:, 0])
        _ = plt.plot(targets[:, 0])
        plt.title('Trajectory prediction of x1')

        plt.figure()
        _ = plt.plot(outputs[:, 1])
        _ = plt.plot(targets[:, 1])
        plt.title('Trajectory prediction of x2')
        plt.figure()
        _ = plt.plot(outputs[:, 2])
        _ = plt.plot(targets[:, 2])
        plt.title('Trajectory prediction of x3')
        plt.figure()
        _ = plt.plot(outputs[:, 3])
        _ = plt.plot(targets[:, 3])
        plt.title('Trajectory prediction of x4')

        plt.show()

    print('--- Finish ---')

if __name__ == '__main__':
    run()
