# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import numpy as np
import theano
import theano.tensor as T
from utils import PickleMixin, TrainingMixin, minibatch_indices, make_minibatch
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


class _BaseRNN(PickleMixin, TrainingMixin):
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
        self.recurrent_activation = recurrent_activation
        self.input_checking = input_checking
        if recurrent_activation == "lstm":
            self.recurrent_activation = build_recurrent_lstm_layer
        else:
            raise ValueError("Value %s not understood for recurrent_activation"
                             % recurrent_activation)

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

            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

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
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_best.pkl", 'wb')
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

    def _stack_layers(self, X_sym, X_mask, layer_sizes, one_step=False):
        input_variable = X_sym

        # layer_sizes consists of input size, all hidden sizes, and output size
        hidden_sizes = layer_sizes[1:-1]
        # set these to stop pep8 vim plugin from complaining
        input_size = None
        output_size = None
        for n in range(len(hidden_sizes)):
            if (n - 1) < 0:
                input_size = layer_sizes[0]
            else:
                if self.bidirectional:
                    # Accomodate for concatenated hiddens
                    input_size = 2 * output_size
                else:
                    input_size = output_size
            hidden_size = hidden_sizes[n]
            if (n + 1) != len(hidden_sizes):
                output_size = hidden_sizes[n + 1]
            else:
                output_size = layer_sizes[-1]

            forward_hidden, forward_params = self.recurrent_activation(
                input_size, hidden_size, output_size, input_variable, X_mask,
                self.random_state, one_step=one_step)

            if self.bidirectional:
                backward_hidden, backward_params = self.recurrent_activation(
                    input_size, hidden_size, output_size, input_variable[::-1],
                    X_mask[::-1], self.random_state, one_step=one_step)
                params = forward_params + backward_params
                input_variable = concatenate(
                    [forward_hidden, backward_hidden[::-1]],
                    axis=forward_hidden.ndim - 1)
            else:
                params = forward_params
                input_variable = forward_hidden


        if self.bidirectional:
            # Accomodate for concatenated hiddens
            sz = 2 * hidden_sizes[-1]
        else:
            sz = hidden_sizes[-1]
        return input_variable, params, sz, input_size, hidden_sizes

    def _updates(self, X_sym, y_sym, params, cost):
        if self.learning_alg == "sgd":
            updates = self.get_clip_sgd_updates(
                params, cost, self.learning_rate, self.momentum)
        elif self.learning_alg == "rmsprop":
            updates = self.get_clip_rmsprop_updates(
                params, cost, self.learning_rate, self.momentum)
        else:
            raise ValueError("Value of %s not a valid learning_alg!"
                             % self.learning_alg)
        return updates


class RNN(_BaseRNN):
    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        (input_variable, params, sz, input_size, hidden_sizes,
         output_size) = self._stack_layers(X_sym, X_mask, layer_sizes)
        # easy mode
        output, output_params = build_linear_layer(sz, output_size,
                                                   input_variable,
                                                   self.random_state)
        params = params + output_params
        shp = output.shape
        output = output.reshape([shp[0] * shp[1], shp[2]])
        y_hat_sym = T.nnet.softmax(output)
        y_sym_reshaped = y_sym.reshape([shp[0] * shp[1], shp[2]])
        cost = -T.mean((y_sym_reshaped * T.log(y_hat_sym)).sum(axis=1))

        self.params_ = params
        updates = self._updates(X_sym, y_sym, params, cost)

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


class GMMRNN(_BaseRNN):
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
        self.recurrent_activation = recurrent_activation
        self.input_checking = input_checking
        if recurrent_activation == "lstm":
            self.recurrent_activation = build_recurrent_lstm_layer
        else:
            raise ValueError("Value %s not understood for recurrent_activation"
                             % recurrent_activation)

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
                    self._setup_functions(X_sym, y_sym, X_mask_sym, y_mask_sym,
                                          self.layer_sizes_)
                train_loss = self.fit_function(X_n, y_n, X_mask, y_mask)
                total_train_loss += train_loss
            current_train_loss = total_train_loss / len(X)
            print("Training loss %f" % current_train_loss)
            self.training_loss_.append(current_train_loss)

            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

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
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()

    def _setup_functions(self, X_sym, y_sym, X_mask, y_mask, layer_sizes):
        (input_variable, params, sz, input_size,
         hidden_sizes) = self._stack_layers(X_sym, X_mask, layer_sizes)

        mu, mu_params = build_linear_layer(
            sz, self.n_mixture_components * self.n_features,
            input_variable, self.random_state)
        params = params + mu_params
        """
        log_var, log_var_params = build_linear_layer(
            sz, self.n_mixture_components * output_size, input_variable,
            self.random_state)
        params = params + log_var_params
        """
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
        """
        log_var_shp = log_var.shape
        """
        coeff_shp = coeff.shape
        y_shp = y_sym.shape

        # TODO: Masking!

        # Reshape everything to 2D
        coeff = coeff.reshape([coeff_shp[0] * coeff_shp[1], coeff_shp[2]])
        coeff = T.nnet.softmax(coeff)
        y_r = y_sym.reshape([y_shp[0] * y_shp[1], y_shp[2]])
        mu = mu.reshape([mu_shp[0] * mu_shp[1], mu_shp[2]])
        """
        log_var = log_var.reshape([log_var_shp[0] * log_var_shp[1],
                                   log_var_shp[2]])
        """
        var = var.reshape([var_shp[0] * var_shp[1], var_shp[2]])

        # Reshape using 2D shapes...
        y_r = y_r.dimshuffle(0, 1, 'x')
        mu = mu.reshape([mu.shape[0],
                        T.cast(mu.shape[1] / coeff.shape[-1], 'int32'),
                        coeff.shape[-1]])
        var = var.reshape([var.shape[0],
                           T.cast(var.shape[1] / coeff.shape[-1], 'int32'),
                           coeff.shape[-1]])
        """
        log_var = log_var.reshape([log_var.shape[0],
                                   T.cast(log_var.shape[1] / coeff.shape[-1],
                                          'int32'),
                                   coeff.shape[-1]])
        """

        # Calculate GMM cost with minimum sigma tolerance
        #log_var = T.log(T.exp(log_var) + 0.)
        log_var = T.log(T.nnet.softplus(var) + 1E-15)
        cost = -0.5 * T.sum(T.sqr(y_r - mu) * T.exp(-log_var) + log_var
                            + T.log(2 * np.pi), axis=1)

        cost = -logsumexp(T.log(coeff) + cost, axis=1).sum()
        self.params_ = params
        updates = self._updates(X_sym, y_sym, params, cost)

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

    def sample(self, n_steps=100, bias=1., random_seed=None):
        if random_seed is None:
            random_state = self.random_state
        else:
            random_state = np.random.RandomState(random_seed)
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
            # Choice sample
            #k = np.where(random_state.rand() < coeff.cumsum())[0][0]
            #s = random_state.randn(mu.shape[0]) * np.sqrt(
            #    np.exp(log_var[:, k])) + mu[:, k]
            # Averaged sample
            s = bias * random_state.randn(*mu.shape) * np.sqrt(np.exp(log_var)) + mu
            s = np.dot(s, coeff)
            samples[n] = s
            # slice back to 2D
        return np.array(samples)

    def force_sample(self, X, bias=1., random_seed=None):
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
            # Choice sample
            #k = np.where(random_state.rand() < coeff.cumsum())[0][0]
            #s = random_state.randn(mu.shape[0]) * np.sqrt(
            #    np.exp(log_var[:, k])) + mu[:, k]
            # Averaged sample
            s = bias * random_state.randn(*mu.shape) * np.sqrt(np.exp(log_var)) + mu
            s = np.dot(s, coeff)
            samples[n] = s
        return np.array(samples)


class EncDecRNN(_BaseRNN):
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
         hidden_sizes) = self._stack_layers(X_sym, X_mask, layer_sizes)

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
