# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import tempfile
import numpy as np
from numpy.lib.stride_tricks import as_strided
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


def make_regression(X):
    X_r = []
    y_r = []
    for s in X:
        null_padded = np.concatenate((np.zeros((1, s.shape[-1])), s), axis=0)
        Xi = null_padded[:-1]
        yi = null_padded[1:]
        X_r.append(Xi)
        y_r.append(yi)
    X_r = np.asarray(X_r).astype(theano.config.floatX).transpose(1, 0, 2)
    y_r = np.asarray(y_r).astype(theano.config.floatX).transpose(1, 0, 2)
    X_mask = np.ones((X_r.shape[0], X_r.shape[1]), dtype=theano.config.floatX)
    y_mask = np.ones((y_r.shape[0], y_r.shape[1]), dtype=theano.config.floatX)
    return X_r, y_r, X_mask, y_mask

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


def segment_axis(a, length, overlap=0, axis=None, end='cut', endvalue=0):
    """Generate a new array that chops the given array along the given axis
    into overlapping frames.
    Parameters
    ----------
    a : array-like
        The array to segment
    length : int
        The length of each frame
    overlap : int, optional
        The number of array elements by which the frames should overlap
    axis : int, optional
        The axis to operate on; if None, act on the flattened array
    end : {'cut', 'wrap', 'end'}, optional
        What to do with the last frame, if the array is not evenly
        divisible into pieces.
            - 'cut'   Simply discard the extra values
            - 'wrap'  Copy values from the beginning of the array
            - 'pad'   Pad with a constant value
    endvalue : object
        The value to use for end='pad'
    Examples
    --------
    >>> segment_axis(arange(10), 4, 2)
    array([[0, 1, 2, 3],
           [2, 3, 4, 5],
           [4, 5, 6, 7],
           [6, 7, 8, 9]])
    Notes
    -----
    The array is not copied unless necessary (either because it is
    unevenly strided and being flattened or because end is set to
    'pad' or 'wrap').
    use as_strided
    """

    if axis is None:
        a = np.ravel(a) # may copy
        axis = 0

    l = a.shape[axis]

    if overlap>=length:
        raise ValueError, "frames cannot overlap by more than 100%"
    if overlap<0 or length<=0:
        raise ValueError, "overlap must be nonnegative and length must be "\
                          "positive"

    if l<length or (l-length)%(length-overlap):
        if l>length:
            roundup = length + \
                      (1+(l-length)//(length-overlap))*(length-overlap)
            rounddown = length + \
                        ((l-length)//(length-overlap))*(length-overlap)
        else:
            roundup = length
            rounddown = 0
        assert rounddown<l<roundup
        assert roundup==rounddown+(length-overlap) or \
               (roundup==length and rounddown==0)
        a = a.swapaxes(-1,axis)

        if end=='cut':
            a = a[...,:rounddown]
        elif end in ['pad','wrap']: # copying will be necessary
            s = list(a.shape)
            s[-1]=roundup
            b = np.empty(s,dtype=a.dtype)
            b[...,:l] = a
            if end=='pad':
                b[...,l:] = endvalue
            elif end=='wrap':
                b[...,l:] = a[...,:roundup-l]
            a = b

        a = a.swapaxes(-1,axis)


    l = a.shape[axis]
    if l==0:
        raise ValueError, "Not enough data points to segment array in 'cut' "\
                          "mode; try 'pad' or 'wrap'"
    assert l>=length
    assert (l-length)%(length-overlap) == 0
    n = 1+(l-length)//(length-overlap)
    s = a.strides[axis]
    newshape = a.shape[:axis] + (n,length) + a.shape[axis+1:]
    newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                 a.strides[axis+1:]

    try:
        return as_strided(a, strides=newstrides, shape=newshape)
    except TypeError:
        warnings.warn("Problem with ndarray creation forces copy.")
        a = a.copy()
        # Shape doesn't change but strides does
        newstrides = a.strides[:axis] + ((length-overlap)*s, s) + \
                     a.strides[axis+1:]
        return as_strided(a, strides=newstrides, shape=newshape)
