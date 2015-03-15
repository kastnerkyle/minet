# -*- coding: utf 8 -*-
from __future__ import division
import theano
import theano.tensor as T


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


def softmax_cost(y_hat_sym, y_sym):
    return -T.mean(T.log(y_hat_sym)[T.arange(y_sym.shape[0]), y_sym])


def recurrence_relation(size):
    """
    Based on code from Shawn Tan
    """

    eye2 = T.eye(size + 2)
    return T.eye(size) + eye2[2:, 1:-1] + eye2[2:, :-2] * (T.arange(size) % 2)


def path_probs(predict, y_sym):
    """
    Based on code from Rakesh - blank is assumed to be highest class in y_sym
    """
    pred_y = predict[:, y_sym]
    rr = recurrence_relation(y_sym.shape[0])

    def step(p_curr, p_prev):
        return p_curr * T.dot(p_prev, rr)

    probabilities, _ = theano.scan(
        step,
        sequences=[pred_y],
        outputs_info=[T.eye(y_sym.shape[0])[0]]
    )
    return probabilities


def _epslog(X):
    return T.cast(T.log(T.clip(X, 1E-12, 1E12)), theano.config.floatX)


def log_path_probs(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan with calculations in log space
    """
    pred_y = y_hat_sym[:, y_sym]
    rr = recurrence_relation(y_sym.shape[0])

    def step(logp_curr, logp_prev):
        return logp_curr + _epslog(T.dot(T.exp(logp_prev), rr))

    log_probs, _ = theano.scan(
        step,
        sequences=[_epslog(pred_y)],
        outputs_info=[_epslog(T.eye(y_sym.shape[0])[0])]
    )
    return log_probs


def ctc_cost(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan
    """
    forward_probs = path_probs(y_hat_sym, y_sym)
    backward_probs = path_probs(y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
    probs = forward_probs * backward_probs / y_hat_sym[:, y_sym]
    total_probs = T.sum(probs)
    return -T.log(total_probs)


def log_ctc_cost(y_hat_sym, y_sym):
    """
    Based on code from Shawn Tan with sum calculations in log space
    """
    log_forward_probs = log_path_probs(y_hat_sym, y_sym)
    log_backward_probs = log_path_probs(
        y_hat_sym[::-1], y_sym[::-1])[::-1, ::-1]
    log_probs = log_forward_probs + log_backward_probs - _epslog(
        y_hat_sym[:, y_sym])
    log_probs = log_probs.flatten()
    max_log = T.max(log_probs)
    # Stable logsumexp
    loss = max_log + T.log(T.sum(T.exp(log_probs - max_log)))
    return -loss
