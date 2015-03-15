# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import tempfile
import numpy as np
import theano
import theano.tensor as T
from theano.compat.python2x import OrderedDict


def minibatch_indices(X, minibatch_size):
    minibatch_indices = np.arange(0, len(X), minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [len(X)])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


def make_minibatch(X, y, one_hot_size=None, is_one_hot=True):
    if one_hot_size is not None:
        is_one_hot = False
    minibatch_size = len(X)
    X_max_sizes = np.max([xi.shape for xi in X], axis=0)
    X_max_sizes = np.asarray([minibatch_size] + list(X_max_sizes))
    # Order into time, samples, feature
    X_max_sizes = np.array([X_max_sizes[1], X_max_sizes[0],
                            X_max_sizes[2]])
    y_max_sizes = np.max([yi.shape for yi in y], axis=0)
    y_max_sizes = np.array([minibatch_size] + list(y_max_sizes))
    # Order into time, samples, label
    # dim is 1 for output label? This may need adjustment for regression
    if len(y_max_sizes) == 3:
        y_max_sizes = np.array([y_max_sizes[1], y_max_sizes[0], y_max_sizes[2]])
        is_one_hot = True
    elif len(y_max_sizes) < 3 and is_one_hot is False:
        if one_hot_size is None:
            raise ValueError("one_hot_size not provided and matrix is 2D!")
        y_max_sizes = np.array([y_max_sizes[1], y_max_sizes[0], one_hot_size])
    else:
        raise ValueError("y must be 2 or 3 dimensional!")

    X_n = np.zeros(X_max_sizes, dtype=X[0].dtype)
    y_n = np.zeros(y_max_sizes).astype(theano.config.floatX)
    X_mask = np.zeros((X_max_sizes[0], X_max_sizes[1])).astype(
        theano.config.floatX)
    y_mask = np.zeros((y_max_sizes[0], y_max_sizes[1])).astype(
        theano.config.floatX)
    for n, t in enumerate(X):
        xshp = X[n].shape
        X_n[:xshp[0], n, :xshp[1]] = X[n]
        X_mask[:xshp[0], n] = 1.

    for n, t in enumerate(y):
        yshp = y[n].shape
        if not is_one_hot:
            for i, v in enumerate(y[n]):
                y_n[i, n, v] = 1.
        else:
            y_n[:yshp[0], n, :yshp[1]] = y[n]
        y_mask[:yshp[0], n] = 1.
    return X_n, y_n, X_mask, y_mask


def labels_to_chars(labels):
    return "".join([chr(l + 97) for l in labels])


class PickleMixin(object):
    def __getstate__(self):
        if not hasattr(self, '_pickle_skip_list'):
            self._pickle_skip_list = []
            for k, v in self.__dict__.items():
                try:
                    f = tempfile.TemporaryFile()
                    cPickle.dump(v, f)
                except:
                    self._pickle_skip_list.append(k)
        state = OrderedDict()
        for k, v in self.__dict__.items():
            if k not in self._pickle_skip_list:
                state[k] = v
        return state

    def __setstate__(self, state):
        self.__dict__ = state


class TrainingMixin(object):
    def get_sgd_updates(self, params, cost, learning_rate,
                        momentum):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        for n, (param, gparam) in enumerate(zip(params, gparams)):
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        return updates

    def _norm_constraint(self, param, update_step, max_col_norm):
        stepped_param = param + update_step
        if param.get_value(borrow=True).ndim == 2:
            col_norms = T.sqrt(T.sum(T.sqr(stepped_param), axis=0))
            desired_norms = T.clip(col_norms, 0, max_col_norm)
            scale = desired_norms / (1e-7 + col_norms)
            new_param = param * scale
            new_update_step = update_step * scale
        else:
            new_param = param
            new_update_step = update_step
        return new_param, new_update_step

    def get_clip_sgd_updates(self, params, cost, learning_rate,
                             momentum, rescale=5.):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            # clip gradient directly, not momentum etc.
            gparam = T.switch(not_finite, 0.1 * param,
                              gparam * (scaling_num / scaling_den))
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * gparam
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step
        return updates

    def get_clip_rmsprop_updates(self, params, cost,
                                 learning_rate, momentum, rescale=5.):
        gparams = T.grad(cost, params)
        updates = OrderedDict()

        if not hasattr(self, "running_average_"):
            self.running_square_ = [0.] * len(gparams)
            self.running_avg_ = [0.] * len(gparams)
            self.updates_storage_ = [0.] * len(gparams)

        if not hasattr(self, "momentum_velocity_"):
            self.momentum_velocity_ = [0.] * len(gparams)

        # Gradient clipping
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), gparams)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
        grad_norm = T.sqrt(grad_norm)
        scaling_num = rescale
        scaling_den = T.maximum(rescale, grad_norm)
        for n, (param, gparam) in enumerate(zip(params, gparams)):
            gparam = T.switch(not_finite, 0.1 * param,
                              gparam * (scaling_num / scaling_den))
            combination_coeff = 0.9
            minimum_grad = 1e-4
            old_square = self.running_square_[n]
            new_square = combination_coeff * old_square + (
                1. - combination_coeff) * T.sqr(gparam)
            old_avg = self.running_avg_[n]
            new_avg = combination_coeff * old_avg + (
                1. - combination_coeff) * gparam
            rms_grad = T.sqrt(new_square - new_avg ** 2)
            rms_grad = T.maximum(rms_grad, minimum_grad)
            velocity = self.momentum_velocity_[n]
            update_step = momentum * velocity - learning_rate * (
                gparam / rms_grad)
            self.running_square_[n] = new_square
            self.running_avg_[n] = new_avg
            self.updates_storage_[n] = update_step
            self.momentum_velocity_[n] = update_step
            updates[param] = param + update_step

        return updates


def rnn_check_array(X, y=None):
    if type(X) == np.ndarray and len(X.shape) == 2:
        X = [X.astype(theano.config.floatX)]
    elif type(X) == np.ndarray and len(X.shape) == 3:
        X = X.astype(theano.config.floatX)
    elif type(X) == list:
        if type(X[0]) == np.ndarray and len(X[0].shape) == 2:
            X = [x.astype(theano.config.floatX) for x in X]
        else:
            raise ValueError("X must be a 2D numpy array or an"
                             "iterable of 2D numpy arrays")
    try:
        X[0].shape[1]
    except AttributeError:
        raise ValueError("X must be a 2D numpy array or an"
                         "iterable of 2D numpy arrays")

    if y is not None:
        if type(y) == np.ndarray and len(y.shape) == 1:
            y = [y.astype('int32')]
        elif type(y) == np.ndarray and len(y.shape) == 2:
            y = y.astype('int32')
        elif type(y) == list:
            if type(y[0]) == np.ndarray and len(y[0].shape) == 1:
                y = [yi.astype('int32') for yi in y]
            elif type(y[0]) != np.ndarray:
                y = [np.asarray(y).astype('int32')]
        try:
            y[0].shape[0]
        except AttributeError:
            raise ValueError("y must be an iterable of 1D numpy arrays")
        return X, y
    else:
        # If y is not passed don't return it
        return X
