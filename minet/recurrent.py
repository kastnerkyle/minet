# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import numpy as np
import theano
import theano.tensor as T
from utils import PickleMixin, minibatch_indices, make_minibatch
from optimizers import rmsprop, sgd_nesterov
from utils import make_regression
from extmath import logsumexp
from layers import concatenate, build_recurrent_lstm_layer, build_linear_layer
from layers import build_tanh_layer, init_recurrent_conditional_lstm_layer
from layers import build_recurrent_conditional_lstm_layer_from_params
from layers import init_linear_layer, build_linear_layer_from_params


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


def stack_forward_layers(X_sym, X_mask, layer_sizes, recurrent_builder,
                         random_state, one_step=False):
    theano_variable = X_sym
    input_sizes = layer_sizes[:-1]
    hidden_sizes = layer_sizes[1:]
    # set these to stop pep8 vim plugin from complaining
    for input_size, hidden_size in zip(input_sizes, hidden_sizes):
        forward_hidden, forward_params = recurrent_builder(
            input_size, hidden_size, theano_variable, X_mask,
            random_state, one_step=one_step)

        params = forward_params
        theano_variable = forward_hidden
    return theano_variable, params


def stack_bidirectional_layers(X_sym, X_mask, layer_sizes, recurrent_builder,
                               random_state, one_step=False):
    theano_variable = X_sym
    input_sizes = layer_sizes[:-1]
    hidden_sizes = layer_sizes[1:]
    for n, (input_size, hidden_size) in enumerate(zip(input_sizes,
                                                      hidden_sizes)):
        if n != 0:
            input_size = 2 * input_size
        forward_hidden, forward_params = recurrent_builder(
            input_size, hidden_size, theano_variable, X_mask,
            random_state, one_step=one_step)
        backward_hidden, backward_params = recurrent_builder(
            input_size, hidden_size, theano_variable[::-1],
            X_mask[::-1], random_state, one_step=one_step)
        params = forward_params + backward_params
        theano_variable = concatenate(
            [forward_hidden, backward_hidden[::-1]],
            axis=forward_hidden.ndim - 1)
    return theano_variable, params


class _BaseRNNClassifier(PickleMixin):
    def __init__(self, hidden_layer_sizes=[100], max_iter=1E2,
                 learning_rate=0.01, momentum=0., learning_alg="sgd",
                 recurrent_activation="lstm", minibatch_size=1,
                 bidirectional=False, save_frequency=10,
                 model_save_name="saved_model", random_seed=None,
                 input_checking=True):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.momentum = momentum
        self.bidirectional = bidirectional
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = int(max_iter)
        self.minibatch_size = minibatch_size
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name
        self.input_checking = input_checking

        if self.learning_alg == "rmsprop":
            self.optimizer = rmsprop
        elif self.learning_alg == "sgd":
            self.optimizer = sgd_nesterov
        else:
            raise ValueError("Value of self.learning_alg"
                             "not understood! Valid options"
                             "%s, got %s" % (["sgd", "rmsprop"],
                                             self.learning_alg))

    def fit(self, X, y, valid_X=None, valid_y=None):
        if self.input_checking:
            X, y = rnn_check_array(X, y)
        input_size = X[0].shape[1]
        # Assume that class values are sequential! and start from 0
        highest_class = np.max([np.max(d) for d in y])
        lowest_class = np.min([np.min(d) for d in y])
        if lowest_class != 0:
            raise ValueError("Labels must start from 0!")
        # Create a list of all classes, then get uniques
        # sum(lists, []) is list concatenation
        all_classes = np.unique(sum([list(np.unique(d)) for d in y], []))
        # +1 to include endpoint
        output_size = len(np.arange(lowest_class, highest_class + 1))
        X_sym = T.tensor3('x')
        y_sym = T.tensor3('y')
        X_mask = T.matrix('x_mask')
        y_mask = T.matrix('y_mask')

        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        if not hasattr(self, 'fit_function'):
            print("Building model!")
            self._setup_functions(X_sym, y_sym, X_mask, y_mask,
                                  self.layer_sizes_)
        self.training_loss_ = []
        if valid_X is not None:
            self.validation_loss_ = []
            if self.input_checking:
                valid_X, valid_y = rnn_check_array(valid_X, valid_y)
                for vy in valid_y:
                    if not np.in1d(np.unique(vy), all_classes).all():
                        raise ValueError(
                            "Validation set contains classes not in training"
                            "set! Training set classes: %s\n, Validation set \
                             classes: %s" % (all_classes, np.unique(vy)))

        best_valid_loss = np.inf
        best_train_loss = np.inf
        try:
            for itr in range(self.max_iter):
                print("Starting pass %d through the dataset" % itr)
                total_train_loss = 0
                for i, j in minibatch_indices(X, self.minibatch_size):
                    X_n, y_n, X_mask, y_mask = make_minibatch(X[i:j], y[i:j],
                                                              output_size)
                    train_loss = self.fit_function(X_n, y_n, X_mask, y_mask)
                    total_train_loss += train_loss
                current_train_loss = total_train_loss / len(X)
                print("Training loss %f" % current_train_loss)
                self.training_loss_.append(current_train_loss)
                if valid_X is not None:
                    total_valid_loss = 0
                    for i, j in minibatch_indices(valid_X, self.minibatch_size):
                        valid_X_n, valid_y_n, X_mask, y_mask = make_minibatch(
                            valid_X[i:j], valid_y[i:j], output_size)
                        valid_loss = self.loss_function(valid_X_n, valid_y_n,
                                                        X_mask, y_mask)
                        total_valid_loss += valid_loss
                    current_valid_loss = total_valid_loss / len(valid_X)
                    print("Validation loss %f" % current_valid_loss)
                    self.validation_loss_.append(current_valid_loss)

                if (itr % self.save_frequency) == 0:
                    f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
                    if current_train_loss < best_train_loss:
                        best_train_loss = current_train_loss
                        f = open(self.model_save_name + "_train_best.pkl", 'wb')
                        cPickle.dump(self, f, protocol=2)
                        f.close()

                if itr == (self.max_iter - 1):
                    f = open(self.model_save_name + "_last.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()

                # Shortcircuit if statement
                if valid_X is not None and current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_valid_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        except KeyboardInterrupt:
            print("User cancelled, saving last model!")
            f = open(self.model_save_name + "_interrupt.pkl", 'wb')
            cPickle.dump(self, f, protocol=2)
            f.close()

    def predict(self, X):
        raise ValueError("Not yet implemented!")
        X = rnn_check_array(X)
        predictions = []
        for n in range(len(X)):
            X_mask = np.ones((len(X[n]), 1)).astype(theano.config.floatX)
            pred = np.argmax(self.predict_function(X[n], X_mask)[0], axis=1)
            predictions.append(pred)
        return predictions

    def predict_proba(self, X):
        raise ValueError("Not yet implemented!")
        X = rnn_check_array(X)
        predictions = []
        for n in range(len(X)):
            X_n = X[n][None].transpose(1, 0, 2)
            X_mask = np.ones((len(X_n), 1)).astype(theano.config.floatX)
            pred = self.predict_function(X_n, X_mask)[0]
            predictions.append(pred)
        return predictions


class _BaseRNNRegressor(PickleMixin):
    def __init__(self, hidden_layer_sizes=[100], n_mixture_components=20,
                 max_iter=1E2, learning_rate=0.01, momentum=0.,
                 learning_alg="sgd", recurrent_activation="lstm",
                 minibatch_size=1, bidirectional=False, save_frequency=10,
                 model_save_name="saved_model", random_seed=None,
                 input_checking=True):
        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.learning_rate = learning_rate
        self.learning_alg = learning_alg
        self.momentum = momentum
        self.bidirectional = bidirectional
        self.hidden_layer_sizes = hidden_layer_sizes
        self.n_mixture_components = n_mixture_components
        self.max_iter = int(max_iter)
        self.minibatch_size = minibatch_size
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name
        self.input_checking = input_checking
        if self.learning_alg == "rmsprop":
            self.optimizer = rmsprop
        elif self.learning_alg == "sgd":
            self.optimizer = sgd_nesterov
        else:
            raise ValueError("Value of self.learning_alg"
                             "not understood! Valid options"
                             "%s, got %s" % (["sgd", "rmsprop"],
                                             self.learning_alg))

    def fit(self, X, valid_X=None):
        # Input size:
        # n_samples
        # n_timesteps
        # n_features
        if self.input_checking:
            X = rnn_check_array(X)

        self.n_features = X[0].shape[-1]

        # Regression features
        self.input_size_ = self.n_features
        self.output_size_ = self.n_features
        X_sym = T.tensor3('x')
        y_sym = T.tensor3('y')
        X_mask_sym = T.matrix('x_mask')
        y_mask_sym = T.matrix('y_mask')

        self.layers_ = []
        self.layer_sizes_ = [self.input_size_]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(self.output_size_)

        self.training_loss_ = []
        if valid_X is not None:
            self.validation_loss_ = []
            if self.input_checking:
                valid_X = rnn_check_array(valid_X)

        best_valid_loss = np.inf
        best_train_loss = np.inf
        try:
            for itr in range(self.max_iter):
                print("Starting pass %d through the dataset" % itr)
                total_train_loss = 0
                for i, j in minibatch_indices(X, self.minibatch_size):
                    X_n, y_n, X_mask, y_mask = make_regression(X[i:j])
                    if not hasattr(self, 'fit_function'):
                        # This is here to make debugging easier
                        X_sym.tag.test_value = X_n
                        y_sym.tag.test_value = y_n
                        X_mask_sym.tag.test_value = X_mask
                        y_mask_sym.tag.test_value = y_mask
                        print("Building model!")
                        print("Minibatch X size %s" % str(X_n.shape))
                        print("Minibatch y size %s" % str(y_n.shape))
                        self._setup_functions(X_sym, y_sym, X_mask_sym,
                                              y_mask_sym, self.layer_sizes_)
                    train_loss = self.fit_function(X_n, y_n, X_mask, y_mask)
                    total_train_loss += train_loss
                current_train_loss = total_train_loss / len(X)
                print("Training loss %f" % current_train_loss)
                self.training_loss_.append(current_train_loss)

                if valid_X is not None:
                    total_valid_loss = 0
                    for i, j in minibatch_indices(valid_X, self.minibatch_size):
                        valid_X_n, valid_y_n, _, _ = make_regression(
                            valid_X[i:j], self.window_size,
                            self.prediction_size)
                        valid_loss = self.loss_function(valid_X_n, valid_y_n)
                        total_valid_loss += valid_loss
                    current_valid_loss = total_valid_loss / len(valid_X)
                    print("Validation loss %f" % current_valid_loss)
                    self.validation_loss_.append(current_valid_loss)

                if (itr % self.save_frequency) == 0:
                    f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
                    if current_train_loss < best_train_loss:
                        best_train_loss = current_train_loss
                        f = open(self.model_save_name + "_train_best.pkl", 'wb')
                        cPickle.dump(self, f, protocol=2)
                        f.close()

                if itr == (self.max_iter - 1):
                    f = open(self.model_save_name + "_last.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()

                # Shortcircuit if statement
                if valid_X is not None and current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_valid_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        except KeyboardInterrupt:
            print("User cancelled, saving last model!")
            f = open(self.model_save_name + "_interrupt.pkl", 'wb')
            cPickle.dump(self, f, protocol=2)
            f.close()


class AGMMRNN(_BaseRNNRegressor):
    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        recurrent_sizes = layer_sizes[:-1]
        input_variable, params = stack_forward_layers(
            X_sym, X_mask, recurrent_sizes, build_recurrent_lstm_layer,
            self.random_state)
        sz = recurrent_sizes[-1]

        # Hardcoded, works for 3 dims/ handwriting *only*!
        # Up/down channel
        binary, binary_params = build_linear_layer(
            sz, 1, input_variable, self.random_state)
        params = params + binary_params

        # Means
        mu, mu_params = build_linear_layer(
            sz, self.n_mixture_components * 2,
            input_variable, self.random_state)
        params = params + mu_params

        # Diagonal
        var, var_params = build_linear_layer(
            sz, self.n_mixture_components * 2,
            input_variable,
            self.random_state)
        params = params + var_params

        # Off-diagonal
        corr, corr_params = build_linear_layer(
            sz,
            self.n_mixture_components * 1,
            input_variable,
            self.random_state)
        params = params + corr_params

        coeff, coeff_params = build_linear_layer(
            sz, self.n_mixture_components, input_variable,
            self.random_state)
        params = params + coeff_params

        mu_shp = mu.shape
        var_shp = var.shape
        corr_shp = corr.shape
        coeff_shp = coeff.shape
        y_shp = y_sym.shape

        # TODO: Masking!
        # Reshape everything to 2D
        coeff = coeff.reshape([coeff_shp[0] * coeff_shp[1], coeff_shp[2]])
        coeff = T.nnet.softmax(coeff)

        y_r = y_sym.reshape([y_shp[0] * y_shp[1], y_shp[2]])
        y_b = y_r[:, 0]
        y_r = y_r[:, 1:]
        mu = mu.reshape([mu_shp[0] * mu_shp[1], mu_shp[2]])
        var = var.reshape([var_shp[0] * var_shp[1], var_shp[2]])
        corr = corr.reshape([corr_shp[0] * corr_shp[1], corr_shp[2]])

        log_var = T.log(T.nnet.softplus(var) + 1E-15)
        # Negative due to sigmoid? AG paper has positive exponential
        binary = T.nnet.sigmoid(-binary)
        corr = T.tanh(corr)
        binary = binary.ravel()

        # Reshape using 2D shapes...
        y_r = y_r.dimshuffle(0, 1, 'x')
        mu = mu.reshape([mu.shape[0],
                        T.cast(mu.shape[1] / coeff.shape[-1], 'int32'),
                        coeff.shape[-1]])
        log_var = log_var.reshape([log_var.shape[0],
                           T.cast(log_var.shape[1] / coeff.shape[-1], 'int32'),
                           coeff.shape[-1]])
        corr = corr.reshape([corr.shape[0],
                            T.cast(corr.shape[1] / coeff.shape[-1], 'int32'),
                            coeff.shape[-1]])
        c1 = -y_b * T.log(binary + 1E-9) - (1 - y_b) * T.log(1 - binary + 1E-9)
        c2 = -T.log(2 * np.pi) - T.sum(log_var, axis=1) - .5 * T.log(
            1 - T.sum(corr, axis=1) ** 2 + 1E-9)
        c3 = -.5 * 1. / (1 - T.sum(corr, axis=1) ** 2)
        x1 = X_sym[:, :, 1]
        x2 = X_sym[:, :, 2]
        mu1 = mu[:, 0, :]
        mu2 = mu[:, 1, :]
        log_var1 = log_var[:, 0, :]
        log_var2 = log_var[:, 1, :]
        if 1:
            t1 = theano.printing.Print("mu1")(mu1.shape)
            t2 = theano.printing.Print("log_var1")(log_var1.shape)
            t3 = theano.printing.Print("x1")(x1.shape)
            t4 = theano.printing.Print("mu2")(mu2.shape)
            t5 = theano.printing.Print("log_var2")(log_var2.shape)
            t6 = theano.printing.Print("x2")(x2.shape)
            #t = [t1, t2, t3, t4, t5, t6]
            #f = theano.function([], t)
        #z = (x1 - mu1) / T.exp(log_var1) ** 2 + (x2 - mu2) / T.exp(log_var2) ** 2
        z = (x1 - mu1)
        #z -= 2 * T.sum(corr, axis=1) * (x1 - mu1) * (x2 - mu2) / T.exp(log_var1 + log_var2)
        cost = c1.dimshuffle(0, 'x') + c2 + c3 * z
        cost = T.sum(-logsumexp(T.log(coeff) + cost, axis=1))
        """
        c1 = -0.5 * T.sum(T.sqr(y_r - mu) * T.exp(-log_var) + log_var
                          + T.log(2 * np.pi), axis=1) - .5 * T.log(
                              1 - corr).sum(axis=1)
        c2 = 2 * corr.sum(axis=1) * T.prod(y_r - mu, axis=1) / T.exp(
            T.sum(log_var, axis=1))
        cost = c1 - c2
        cost = T.sum(-logsumexp(T.log(coeff) + cost, axis=1) -
                     y_b * T.log(binary) - (1 - y_b) * T.log(1 - binary))
        """

        grads = T.grad(cost, params)
        self.opt_ = self.optimizer(params)
        updates = self.opt_.updates(
            params, grads, self.learning_rate, self.momentum)

        self.fit_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                    y_mask],
                                            outputs=cost,
                                            updates=updates,
                                            on_unused_input="ignore")

        self.loss_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                     y_mask],
                                             outputs=cost,
                                             on_unused_input="ignore")

        self.generate_function = theano.function(inputs=[X_sym, X_mask],
                                                 outputs=[mu, log_var, coeff],
                                                 on_unused_input="ignore")
    """
    def sample(self, n_steps=100, bias=1., alg="soft", seed_sequence=None,
               random_seed=None):
        if random_seed is None:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(random_seed)
        if seed_sequence is not None:
            raise ValueError("Seeded generation not yet supported")
        samples = np.zeros((n_steps, self.n_features))
        s = random_state.rand(self.n_features)
        samples[0] = s
        for n in range(1, n_steps):
            X_n = rnn_check_array(samples[None])
            X_n = X_n.transpose(1, 0, 2)
            X_mask = np.ones((X_n.shape[0], X_n.shape[1]),
                             dtype=theano.config.floatX)
            r = self.generate_function(X_n[:n], X_mask[:n])
            # get samples
            # outputs are n_features, n_predictions, n_gaussians
            mu = r[0][-1]
            log_var = r[1][-1]
            coeff = r[2][-1]

            # Make sure it sums to 1
            coeff = coeff / coeff.sum()
            if alg == "hard":
                # Choice sample
                k = np.where(random_state.rand() < coeff.cumsum())[0][0]
                s = random_state.randn(mu.shape[0]) * np.sqrt(
                    np.exp(log_var[:, k])) + mu[:, k]
            elif alg == "soft":
                # Averaged sample
                s = bias * random_state.randn(*mu.shape) * np.sqrt(
                    np.exp(log_var)) + mu
                s = np.dot(s, coeff)
            else:
                raise ValueError("alg must be 'hard' or 'soft'")
            samples[n] = s
        return np.array(samples)

    def force_sample(self, X, bias=1., alg="soft", random_seed=None):
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array of (steps, features)")
        if random_seed is None:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(random_seed)
        samples = np.zeros((X.shape[0], self.n_features))
        X_n = rnn_check_array(X[None])
        X_n = X_n.transpose(1, 0, 2)
        X_mask = np.ones((X_n.shape[0], X_n.shape[1]),
                         dtype=theano.config.floatX)
        r = self.generate_function(X_n, X_mask)
        for n in range(X.shape[0]):
            # get samples
            # outputs are n_features, n_predictions, n_gaussians
            mu = r[0][n]
            log_var = r[1][n]
            coeff = r[2][n]

            # Make sure it sums to 1
            coeff = coeff / coeff.sum()
            if alg == "hard":
                # Choice sample
                k = np.where(random_state.rand() < coeff.cumsum())[0][0]
                s = random_state.randn(mu.shape[0]) * np.sqrt(
                    np.exp(log_var[:, k])) + mu[:, k]
            elif alg == "soft":
                # Averaged sample
                s = bias * random_state.randn(*mu.shape) * np.sqrt(
                    np.exp(log_var)) + mu
                s = np.dot(s, coeff)
            else:
                raise ValueError("alg must be 'hard' or 'soft'")
            samples[n] = s
        return np.array(samples)
    """


class GMMRNN(_BaseRNNRegressor):
    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        recurrent_sizes = layer_sizes[:-1]
        input_variable, params = stack_forward_layers(
            X_sym, X_mask, recurrent_sizes, build_recurrent_lstm_layer,
            self.random_state)
        sz = recurrent_sizes[-1]

        mu, mu_params = build_linear_layer(
            sz, self.n_mixture_components * self.n_features,
            input_variable, self.random_state)
        params = params + mu_params
        var, var_params = build_linear_layer(
            sz, self.n_mixture_components * self.n_features,
            input_variable,
            self.random_state)
        params = params + var_params
        coeff, coeff_params = build_linear_layer(
            sz, self.n_mixture_components, input_variable,
            self.random_state)
        params = params + coeff_params

        mu_shp = mu.shape
        var_shp = var.shape
        coeff_shp = coeff.shape
        y_shp = y_sym.shape

        # TODO: Masking!

        # Reshape everything to 2D
        coeff = coeff.reshape([coeff_shp[0] * coeff_shp[1], coeff_shp[2]])
        coeff = T.nnet.softmax(coeff)
        y_r = y_sym.reshape([y_shp[0] * y_shp[1], y_shp[2]])
        mu = mu.reshape([mu_shp[0] * mu_shp[1], mu_shp[2]])
        var = var.reshape([var_shp[0] * var_shp[1], var_shp[2]])

        # Reshape using 2D shapes...
        y_r = y_r.dimshuffle(0, 1, 'x')
        mu = mu.reshape([mu.shape[0],
                        T.cast(mu.shape[1] / coeff.shape[-1], 'int32'),
                        coeff.shape[-1]])
        var = var.reshape([var.shape[0],
                           T.cast(var.shape[1] / coeff.shape[-1], 'int32'),
                           coeff.shape[-1]])

        # Calculate GMM cost with minimum tolerance
        log_var = T.log(T.nnet.softplus(var) + 1E-15)
        cost = -0.5 * T.sum(T.sqr(y_r - mu) * T.exp(-log_var) + log_var
                            + T.log(2 * np.pi), axis=1)

        cost = -logsumexp(T.log(coeff) + cost, axis=1).sum()
        grads = T.grad(cost, params)
        self.opt_ = self.optimizer(params)
        updates = self.opt_.updates(
            params, grads, self.learning_rate, self.momentum)

        self.fit_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                    y_mask],
                                            outputs=cost,
                                            updates=updates,
                                            on_unused_input="ignore")

        self.loss_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                     y_mask],
                                             outputs=cost,
                                             on_unused_input="ignore")

        self.generate_function = theano.function(inputs=[X_sym, X_mask],
                                                 outputs=[mu, log_var, coeff],
                                                 on_unused_input="ignore")

    def sample(self, n_steps=100, bias=1., alg="soft", seed_sequence=None,
               random_seed=None):
        if random_seed is None:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(random_seed)
        if seed_sequence is not None:
            raise ValueError("Seeded generation not yet supported")
        samples = np.zeros((n_steps, self.n_features))
        s = random_state.rand(self.n_features)
        samples[0] = s
        for n in range(1, n_steps):
            X_n = rnn_check_array(samples[None])
            X_n = X_n.transpose(1, 0, 2)
            X_mask = np.ones((X_n.shape[0], X_n.shape[1]),
                             dtype=theano.config.floatX)
            r = self.generate_function(X_n[:n], X_mask[:n])
            # get samples
            # outputs are n_features, n_predictions, n_gaussians
            mu = r[0][-1]
            log_var = r[1][-1]
            coeff = r[2][-1]

            # Make sure it sums to 1
            coeff = coeff / coeff.sum()
            if alg == "hard":
                # Choice sample
                k = np.where(random_state.rand() < coeff.cumsum())[0][0]
                s = random_state.randn(mu.shape[0]) * np.sqrt(
                    np.exp(log_var[:, k])) + mu[:, k]
            elif alg == "soft":
                # Averaged sample
                s = bias * random_state.randn(*mu.shape) * np.sqrt(
                    np.exp(log_var)) + mu
                s = np.dot(s, coeff)
            else:
                raise ValueError("alg must be 'hard' or 'soft'")
            samples[n] = s
        return np.array(samples)

    def force_sample(self, X, bias=1., alg="soft", random_seed=None):
        if len(X.shape) != 2:
            raise ValueError("X must be a 2D array of (steps, features)")
        if random_seed is None:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(random_seed)
        samples = np.zeros((X.shape[0], self.n_features))
        X_n = rnn_check_array(X[None])
        X_n = X_n.transpose(1, 0, 2)
        X_mask = np.ones((X_n.shape[0], X_n.shape[1]),
                         dtype=theano.config.floatX)
        r = self.generate_function(X_n, X_mask)
        for n in range(X.shape[0]):
            # get samples
            # outputs are n_features, n_predictions, n_gaussians
            mu = r[0][n]
            log_var = r[1][n]
            coeff = r[2][n]

            # Make sure it sums to 1
            coeff = coeff / coeff.sum()
            if alg == "hard":
                # Choice sample
                k = np.where(random_state.rand() < coeff.cumsum())[0][0]
                s = random_state.randn(mu.shape[0]) * np.sqrt(
                    np.exp(log_var[:, k])) + mu[:, k]
            elif alg == "soft":
                # Averaged sample
                s = bias * random_state.randn(*mu.shape) * np.sqrt(
                    np.exp(log_var)) + mu
                s = np.dot(s, coeff)
            else:
                raise ValueError("alg must be 'hard' or 'soft'")
            samples[n] = s
        return np.array(samples)


class RNN(_BaseRNNClassifier):
    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        (input_variable, params, sz, input_size, hidden_sizes,
         output_size) = self._stack_layers(X_sym, X_mask, layer_sizes)
        output, output_params = build_linear_layer(sz, output_size,
                                                   input_variable,
                                                   self.random_state)
        params = params + output_params
        shp = output.shape
        output = output.reshape([shp[0] * shp[1], shp[2]])
        y_hat_sym = T.nnet.softmax(output)
        y_sym_reshaped = y_sym.reshape([shp[0] * shp[1], shp[2]])
        cost = -T.mean((y_sym_reshaped * T.log(y_hat_sym)).sum(axis=1))

        grads = T.grad(cost, params)
        self.opt_ = self.optimizer(params)
        updates = self.opt_.updates(
            params, grads, self.learning_rate, self.momentum)

        self.fit_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                    y_mask],
                                            outputs=cost,
                                            updates=updates,
                                            on_unused_input="ignore")

        self.loss_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                     y_mask],
                                             outputs=cost,
                                             on_unused_input="ignore")

        self.predict_function = theano.function(
            inputs=[X_sym, X_mask],
            outputs=y_hat_sym,
            on_unused_input="ignore")


class EncDecRNN(_BaseRNNClassifier):
    def predict(self, X):
        raise ValueError("Not yet implemented!")
        X = rnn_check_array(X)
        for n in range(len(X)):
            X_n = X[n][None].transpose(1, 0, 2)
            X_mask = np.ones((len(X_n), 1)).astype(theano.config.floatX)
            state, memory, ctx = self._encode(X_n, X_mask)
            for i in range(100):
                ctx = np.tile(ctx.squeeze(), [1, 1, 1]).transpose(1, 0, 2)
                """
                self._sampler_step = theano.function(
                    [y_sw_sampler, context, X_mask, init_state_sampler,
                        init_memory_sampler],
                    [y_hat_sampler, next_state, next_memory])
                from IPython import embed; embed()
                """

    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        (input_variable, params, sz, input_size,
         hidden_sizes, output_size) = self._stack_layers(X_sym, X_mask,
                                                         layer_sizes)

        # hardmode
        context = input_variable
        context_mean = context[0]

        init_state, state_params = build_tanh_layer(sz, hidden_sizes[-1],
                                                    context_mean,
                                                    self.random_state)
        init_memory, memory_params = build_tanh_layer(sz, hidden_sizes[-1],
                                                      context_mean,
                                                      self.random_state)
        # partial sampler setup
        self._encode = theano.function([X_sym, X_mask],
                                       [init_state, init_memory, context])
        init_state_sampler = T.matrix()
        init_memory_sampler = T.matrix()
        y_sw_sampler = T.tensor3()
        y_sw_mask = T.alloc(1., y_sw_sampler.shape[0], 1)

        # need this style of init to reuse params for sampler and actual
        # training. This makes this part quite nasty - dictionary
        # for initialization and params is making more and more sense.
        # conditional params will be reused below
        conditional_params = init_recurrent_conditional_lstm_layer(
            output_size, hidden_sizes[-1], sz, self.random_state)

        rval, _p = build_recurrent_conditional_lstm_layer_from_params(
            conditional_params, y_sw_sampler, y_sw_mask, context, X_mask,
            init_state_sampler, init_memory_sampler,
            self.random_state, one_step=True)
        next_state, next_memory, sampler_contexts, _ = rval
        # end sampler parts... for now

        params = params + state_params + memory_params
        shifted_labels = T.zeros_like(y_sym)
        shifted_labels = T.set_subtensor(shifted_labels[1:], y_sym[:-1])
        y_sym = shifted_labels

        rval, _p = build_recurrent_conditional_lstm_layer_from_params(
            conditional_params, shifted_labels, y_mask, context, X_mask,
            init_state, init_memory, self.random_state)
        projected_hidden, _, contexts, attention = rval

        params = params + conditional_params

        # once again, need to use same params for sample gen
        lh_params = init_linear_layer(hidden_sizes[-1], output_size,
                                      self.random_state)
        logit_hidden, _ = build_linear_layer_from_params(lh_params,
                                                         projected_hidden)
        params = params + lh_params

        lo_params = init_linear_layer(output_size, output_size,
                                      self.random_state)
        logit_out, _ = build_linear_layer_from_params(lo_params, y_sym)
        params = params + lo_params

        lc_params = init_linear_layer(sz, output_size,
                                      self.random_state)
        logit_contexts, _ = build_linear_layer_from_params(lc_params,
                                                           contexts)
        params = params + lc_params

        logit = T.tanh(logit_hidden + logit_out + logit_contexts)
        output_params = init_linear_layer(output_size, output_size,
                                          self.random_state)
        output, _ = build_linear_layer_from_params(output_params,
                                                   logit)
        params = params + output_params

        shp = output.shape
        output = output.reshape([shp[0] * shp[1], shp[2]])
        y_hat_sym = T.nnet.softmax(output)

        # Need to apply mask so that cost isn't punished
        y_sym_reshaped = (y_sym * y_mask.dimshuffle(0, 1, 'x')).reshape(
            [shp[0] * shp[1], shp[2]])
        y_sym_reshaped = y_sym.reshape([shp[0] * shp[1], shp[2]])
        cost = -T.mean((y_sym_reshaped * T.log(y_hat_sym)).sum(axis=1))

        # Finish sampler
        logit_sampler_hidden, _ = build_linear_layer_from_params(lh_params,
                                                                 next_state)
        logit_sampler_out, _ = build_linear_layer_from_params(lo_params,
                                                              y_sw_sampler)
        logit_sampler_contexts, _ = build_linear_layer_from_params(
            lc_params, sampler_contexts)
        logit_sampler = T.tanh(logit_sampler_hidden + logit_sampler_out
                               + logit_sampler_contexts)
        output_sampler, _ = build_linear_layer_from_params(output_params,
                                                           logit_sampler)
        shp = output_sampler.shape
        output_sampler = output_sampler.reshape([shp[0] * shp[1], shp[2]])
        y_hat_sampler = T.nnet.softmax(output_sampler)
        self._sampler_step = theano.function(
            [y_sw_sampler, context, X_mask, init_state_sampler,
                init_memory_sampler],
            [y_hat_sampler, next_state, next_memory])

        self.params_ = params
        updates = self._updates(X_sym, y_sym, params, cost)

        self.fit_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                    y_mask],
                                            outputs=cost,
                                            updates=updates,
                                            on_unused_input="warn")

        self.loss_function = theano.function(inputs=[X_sym, y_sym, X_mask,
                                                     y_mask],
                                             outputs=cost,
                                             on_unused_input="warn")

        self.predict_function = theano.function(
            inputs=[X_sym, X_mask, y_sym, y_mask],
            outputs=y_hat_sym)
