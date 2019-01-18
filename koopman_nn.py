# from __future__ import absolute_import
# from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.norm as norm
from autograd import grad
from autograd.misc.optimizers import adam

import matplotlib.pyplot as plt
import h5py


def init_random_params(scale, layer_sizes, rs=npr.RandomState(0)):
    """Build a list of (weights, biases) tuples, one for each layer."""
    return [(rs.randn(insize, outsize) * scale,  # weight matrix
             rs.randn(outsize) * scale)  # bias vector
            for insize, outsize in zip(layer_sizes[:-1], layer_sizes[1:])]


def func_id(x):
    return x


def nn_predict(params, inputs, nonlinearity=np.tanh):
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
    encoded = nn_predict(weights['enc'], inputs)
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


def logprob_koop(weights, inputs, targets, noise_scale=0.1):

    decoded = nn_encode_decode(weights, inputs)
    decoded_forward = nn_encode_foward_decode(weights, inputs)

    encoded_targets = nn_encode(weights, targets)
    encoded_forward = nn_encode_forward(weights, inputs)

    t1 = np.sum(norm.logpdf(decoded, inputs, noise_scale))
    t2 = np.sum(norm.logpdf(decoded_forward, targets, noise_scale))
    t3 = np.sum(norm.logpdf(encoded_forward, encoded_targets, noise_scale))

    return t1 + t2 + t3


def normalize(X):
    Y = X - np.sum(X, axis=0) / X.shape[0]
    Y = Y / np.std(Y, axis=0)
    return Y


def build_wc_dataset():
    file_url = 'data/wc1.hd5'
    hf = h5py.File(file_url, 'r+')
    # S = np.squeeze(hf['W']).T
    y = np.squeeze(hf['y']).T
    y = y[:, 0:10000:10]
    n = int(y.shape[0] / 2)
    inputs = y[:, 0:-1].T
    targets = y[:, 1:].T

    return inputs, targets


def callback(params, t, g):
    print("Iteration {} log likelihood {}".format(t, -objective(params, t)))


def initialize_weights(G, D):
    init_scale = 0.1

    init_weights_enc = init_random_params(init_scale, layer_sizes=[D, G * D, G * D])
    init_weights_dec = init_random_params(init_scale, layer_sizes=[G * D, G * D, D])
    init_weights_koop = init_random_params(init_scale, layer_sizes=[G * D, G * D])
    init_weights = {'enc': init_weights_enc, 'dec': init_weights_dec, 'koop': init_weights_koop}

    return init_weights


def objective(weights, t):
    return -logprob_koop(weights, inputs, targets)


def callback(params, t, g):
    print("Iteration {:3d} log likelihood {:+1.3e}".format(t, -objective(params, t)))


def run():

    global inputs, targets, hyper

    num_iters = 150

    # inputs, targets = build_tvb_dataset()
    inputs, targets = build_wc_dataset()

    D = inputs.shape[1]
    G = 20

    init_weights = initialize_weights(G, D)

    print('----------  Optimizing KOOPMAN NEURAL NET for {} iterations ..... \n'.format(num_iters))

    opt_weights = adam(grad(objective), init_weights, step_size=0.01, num_iters=num_iters, callback=callback)

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
    _ = plt.plot(outputs[:, 0:3], marker='x')
    _ = plt.plot(targets[:, 0:3], marker='+')

    plt.show()

    re = np.mean([np.linalg.norm(targets[i] - outputs[i]) / np.linalg.norm(targets[i]) for i in range(len(targets))])

    print('Relative norm error {:+1.4e}'.format(re))
    print('--- Finish ---')


if __name__ == '__main__':
    run()
