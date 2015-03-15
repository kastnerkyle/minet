# -*- coding: utf 8 -*-
from __future__ import division
import numpy as np
from scipy import linalg
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams


def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.tensor.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Stolen from Lasagne
    """
    if axis < 0:
        axis += tensor_list[0].ndim

    concat_size = sum(tensor.shape[axis] for tensor in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tensor in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tensor.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tensor)
        offset += tensor.shape[axis]

    return out


def relu(x):
    return x * (x > 1e-6)


def clip_relu(x, clip_lim=20):
    return x * (T.lt(x, 1e-6) and T.gt(x, clip_lim))


def dropout(random_state, X, keep_prob=0.5):
    if keep_prob > 0. and keep_prob < 1.:
        seed = random_state.randint(2 ** 30)
        srng = RandomStreams(seed)
        mask = srng.binomial(n=1, p=keep_prob, size=X.shape,
                             dtype=theano.config.floatX)
        return X * mask
    return X


def fast_dropout(random_state, X):
    seed = random_state.randint(2 ** 30)
    srng = RandomStreams(seed)
    mask = srng.normal(size=X.shape, avg=1., dtype=theano.config.floatX)
    return X * mask


def shared_zeros(shape):
    """ Builds a theano shared variable filled with a zeros numpy array """
    return theano.shared(value=np.zeros(*shape).astype(theano.config.floatX),
                         borrow=True)


def shared_rand(shape, rng):
    """ Builds a theano shared variable filled with random values """
    return theano.shared(value=(0.01 * (rng.rand(*shape) - 0.5)).astype(
        theano.config.floatX), borrow=True)


def np_rand(shape, rng):
    return (0.01 * (rng.rand(*shape) - 0.5)).astype(theano.config.floatX)


def np_randn(shape, rng, name=None):
    """ Builds a numpy variable filled with random normal values """
    return (0.01 * rng.randn(*shape)).astype(theano.config.floatX)


def np_ortho(shape, rng, name=None):
    """ Builds a theano variable filled with orthonormal random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return o_g.astype(theano.config.floatX)


def shared_ortho(shape, rng, name=None):
    """ Builds a theano shared variable filled with random values """
    g = rng.randn(*shape)
    o_g = linalg.svd(g)[0]
    return theano.shared(value=o_g.astype(theano.config.floatX), borrow=True)


def init_linear_layer(input_size, output_size, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    params = [W, b]
    return params


def build_linear_layer_from_params(params, input_variable):
    W, b = params
    output_variable = T.dot(input_variable, W) + b
    return output_variable, params


def build_linear_layer(input_size, output_size, input_variable, random_state):
    params = init_linear_layer(input_size, output_size, random_state)
    return build_linear_layer_from_params(params, input_variable)


def init_tanh_layer(input_size, output_size, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    params = [W, b]
    return params


def build_tanh_layer_from_params(params, input_variable):
    W, b = params
    output_variable = T.tanh(T.dot(input_variable, W) + b)
    return output_variable, params


def build_tanh_layer(input_size, output_size, input_variable, random_state):
    params = init_tanh_layer(input_size, output_size, random_state)
    return build_tanh_layer_from_params(params, input_variable)


def build_relu_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = relu(T.dot(input_variable, W) + b)
    params = [W, b]
    return output_variable, params


def build_sigmoid_layer(input_size, output_size, input_variable, random_state):
    W_values = np.asarray(random_state.uniform(
        low=-np.sqrt(6. / (input_size + output_size)),
        high=np.sqrt(6. / (input_size + output_size)),
        size=(input_size, output_size)), dtype=theano.config.floatX)
    W = theano.shared(value=4 * W_values, name='W', borrow=True)
    b_values = np.zeros((output_size,), dtype=theano.config.floatX)
    b = theano.shared(value=b_values, name='b', borrow=True)
    output_variable = T.nnet.sigmoid(T.dot(input_variable, W) + b)
    params = [W, b]
    return output_variable, params


def init_recurrent_conditional_lstm_layer(input_size, hidden_size, output_size,
                                          random_state):
    # input to LSTM
    W_ = np.concatenate(
        [np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state)],
        axis=1)

    W = theano.shared(W_, borrow=True)

    # LSTM to LSTM
    U_ = np.concatenate(
        [np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state)],
        axis=1)

    U = theano.shared(U_, borrow=True)

    # bias to LSTM
    # TODO: Ilya init for biases...
    b = shared_zeros((4 * hidden_size,))

    # Context to LSTM
    Wc = shared_rand((output_size, 4 * hidden_size), random_state)

    # attention: context to hidden
    Wc_att = shared_ortho((output_size, output_size), random_state)

    # attention: LSTM to hidden
    Wd_att = shared_rand((hidden_size, output_size), random_state)

    # attention: hidden bias
    b_att = shared_zeros((output_size,))

    # attention
    U_att = shared_rand((output_size, 1), random_state)
    c_att = shared_zeros((1,))

    params = [W, U, b, Wc, Wc_att, Wd_att, b_att, U_att, c_att]

    return params


def build_recurrent_conditional_lstm_layer(input_size, hidden_size, output_size,
                                           input_variable, mask, context,
                                           context_mask, init_state,
                                           init_memory, random_state,
                                           one_step=False):
    params = init_recurrent_conditional_lstm_layer(input_size, hidden_size,
                                                   output_size, random_state)

    return build_recurrent_conditional_lstm_layer_from_params(params,
                                                              input_variable,
                                                              mask, context,
                                                              context_mask,
                                                              init_state,
                                                              init_memory,
                                                              random_state,
                                                              one_step=one_step)


def build_recurrent_conditional_lstm_layer_from_params(params, input_variable,
                                                       mask, context,
                                                       context_mask, init_state,
                                                       init_memory,
                                                       random_state,
                                                       one_step=False):
    [W, U, b, Wc, Wc_att, Wd_att, b_att, U_att, c_att] = params

    n_steps = input_variable.shape[0]
    n_samples = input_variable.shape[1]
    # n_features = input_variable.shape[2]

    hidden_size = U.shape[0]

    # projected context
    projected_context = T.dot(context, Wc_att) + b_att

    # projected input
    x = T.dot(input_variable, W) + b

    def _slice(X, n, hidden_size):
        # Function is needed because tensor size changes across calls to step?
        if X.ndim == 3:
            return X[:, :, n * hidden_size:(n + 1) * hidden_size]
        return X[:, n * hidden_size:(n + 1) * hidden_size]

    def step(x_t, m, h_tm1, c_tm1, ctx_t, att, pctx_):
        projected_state = T.dot(h_tm1, Wd_att)
        pctx_ = T.tanh(pctx_ + projected_state[None, :, :])
        new_att = T.dot(pctx_, U_att) + c_att
        new_att = new_att.reshape([new_att.shape[0], new_att.shape[1]])
        new_att = T.exp(new_att) * context_mask
        new_att = new_att / new_att.sum(axis=0, keepdims=True)
        # Current context
        ctx_t = (context * new_att[:, :, None]).sum(axis=0)

        preactivation = T.dot(h_tm1, U)
        preactivation += x_t
        preactivation += T.dot(ctx_t, Wc)

        i_t = T.nnet.sigmoid(_slice(preactivation, 0, hidden_size))
        f_t = T.nnet.sigmoid(_slice(preactivation, 1, hidden_size))
        o_t = T.nnet.sigmoid(_slice(preactivation, 2, hidden_size))
        c_t = T.tanh(_slice(preactivation, 3, hidden_size))

        c_t = f_t * c_tm1 + i_t * c_t
        c_t = m[:, None] * c_t + (1. - m)[:, None] * c_tm1
        h_t = o_t * T.tanh(c_t)
        h_t = m[:, None] * h_t + (1. - m)[:, None] * h_tm1
        return (h_t, c_t, ctx_t, new_att.T, projected_state,
                i_t, f_t, o_t, preactivation)

    init_context = T.zeros((n_samples, context.shape[2]),
                           dtype=theano.config.floatX)
    init_att = T.zeros((n_samples, context.shape[0]),
                       dtype=theano.config.floatX)
    # Scan cannot handle batch sizes of 1?
    # Unbroadcast can fix it... but still weird
    # https://github.com/Theano/Theano/issues/1772
    # init_context = T.unbroadcast(init_context, 0)
    # init_att = T.unbroadcast(init_att, 0)

    if one_step:
        rval = step(x, mask, init_state, init_memory, None, None,
                    projected_context)
    else:
        rval, _ = theano.scan(step,
                              sequences=[x, mask],
                              outputs_info=[init_state, init_memory,
                                            init_context, init_att,
                                            None, None, None, None, None],
                              non_sequences=[projected_context, ],
                              n_steps=n_steps)

    # hidden = rval[0]
    # state = rval[1]
    # final_context = rval[2]
    # final_att = rval[3]
    return rval[:4], params


def init_recurrent_lstm_layer(input_size, hidden_size, output_size,
                              random_state):
    # input to LSTM
    W_ = np.concatenate(
        [np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state),
         np_rand((input_size, hidden_size), random_state)],
        axis=1)

    W = theano.shared(W_, borrow=True)

    # LSTM to LSTM
    U_ = np.concatenate(
        [np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state),
         np_ortho((hidden_size, hidden_size), random_state)],
        axis=1)

    U = theano.shared(U_, borrow=True)

    # bias to LSTM
    b = shared_zeros((4 * hidden_size,))

    params = [W, U, b]
    return params


def build_recurrent_lstm_layer(input_size, hidden_size, output_size,
                               input_variable, mask,
                               random_state, one_step=False):
    params = init_recurrent_lstm_layer(input_size, hidden_size, output_size,
                                       random_state)
    return build_recurrent_lstm_layer_from_params(params, input_variable, mask,
                                                  random_state,
                                                  one_step=one_step)


def build_recurrent_lstm_layer_from_params(params, input_variable, mask,
                                           random_state, one_step=False):
    [W, U, b] = params

    hidden_size = U.shape[0]

    n_steps = input_variable.shape[0]
    n_samples = input_variable.shape[1]
    # n_features = input_variable.shape[2]

    def _slice(X, n, hidden_size):
        # Function is needed because tensor size changes across calls to step?
        if X.ndim == 3:
            return X[:, :, n * hidden_size:(n + 1) * hidden_size]
        return X[:, n * hidden_size:(n + 1) * hidden_size]

    def step(x_t, m, h_tm1, c_tm1):
        preactivation = T.dot(h_tm1, U)
        preactivation += x_t
        preactivation += b

        i_t = T.nnet.sigmoid(_slice(preactivation, 0, hidden_size))
        f_t = T.nnet.sigmoid(_slice(preactivation, 1, hidden_size))
        o_t = T.nnet.sigmoid(_slice(preactivation, 2, hidden_size))
        c_t = T.tanh(_slice(preactivation, 3, hidden_size))

        c_t = f_t * c_tm1 + i_t * c_t
        c_t = m[:, None] * c_t + (1. - m)[:, None] * c_tm1
        h_t = o_t * T.tanh(c_t)
        h_t = m[:, None] * h_t + (1. - m)[:, None] * h_tm1
        return h_t, c_t, i_t, f_t, o_t, preactivation

    # Scan cannot handle batch sizes of 1?
    # Unbroadcast can fix it... but still weird
    # https://github.com/Theano/Theano/issues/1772
    init_hidden = T.zeros((n_samples, hidden_size))
    init_cell = T.zeros((n_samples, hidden_size))
    init_hidden = T.unbroadcast(init_hidden, 0)
    init_cell = T.unbroadcast(init_cell, 0)

    x = T.dot(input_variable, W) + b
    if one_step:
        rval = step(x, mask, init_hidden, init_cell)
    else:
        rval, _ = theano.scan(step,
                              sequences=[x, mask],
                              outputs_info=[init_hidden, init_cell,
                                            None, None, None, None],
                              n_steps=n_steps)

    hidden = rval[0]
    return hidden, params
