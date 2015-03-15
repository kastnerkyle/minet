# -*- coding: utf 8 -*-
from __future__ import division
try:
    import cPickle
except ImportError:
    import pickle as cPickle
import numpy as np
import theano
import theano.tensor as T
from net_utils import PickleMixin, TrainingMixin
from net_utils import build_relu_layer, build_tanh_layer, build_sigmoid_layer
from net_utils import build_linear_layer, softmax_cost


class FeedforwardNetwork(PickleMixin, TrainingMixin):
    def __init__(self, hidden_layer_sizes=[500], batch_size=100, max_iter=1E3,
                 learning_rate=0.01, momentum=0., learning_alg="sgd",
                 activation="tanh", model_save_name="saved_model",
                 save_frequency=100, random_seed=None):

        if random_seed is None or type(random_seed) is int:
            self.random_state = np.random.RandomState(random_seed)
        self.max_iter = int(max_iter)
        self.hidden_layer_sizes = hidden_layer_sizes
        self.batch_size = batch_size
        self.save_frequency = save_frequency
        self.model_save_name = model_save_name

        self.learning_rate = learning_rate
        self.momentum = momentum
        self.learning_alg = learning_alg
        if activation == "relu":
            self.feedforward_function = build_relu_layer
        elif activation == "tanh":
            self.feedforward_function = build_tanh_layer
        elif activation == "sigmoid":
            self.feedforward_function = build_sigmoid_layer
        else:
            raise ValueError("Value %s not understood for activation"
                             % activation)

    def _setup_functions(self, X_sym, y_sym, layer_sizes):
        input_variable = X_sym
        params = []
        for i, (input_size, output_size) in enumerate(zip(layer_sizes[:-1],
                                                          layer_sizes[1:-1])):
            output_variable, layer_params = self.feedforward_function(
                input_size, output_size, input_variable, self.random_state)
            params.extend(layer_params)
            input_variable = output_variable

        output_variable, layer_params = build_linear_layer(
            layer_sizes[-2], layer_sizes[-1], input_variable, self.random_state)
        params.extend(layer_params)
        y_hat_sym = T.nnet.softmax(output_variable)
        cost = softmax_cost(y_hat_sym, y_sym)

        self.params_ = params

        if self.learning_alg == "sgd":
            updates = self.get_sgd_updats(X_sym, y_sym, params, cost,
                                          self.learning_rate,
                                          self.momentum)
        else:
            raise ValueError("Algorithm %s is not "
                             "a valid argument for learning_alg!"
                             % self.learning_alg)
        self.fit_function = theano.function(
            inputs=[X_sym, y_sym], outputs=cost, updates=updates)
        self.loss_function = theano.function(
            inputs=[X_sym, y_sym], outputs=cost)

        self.predict_function = theano.function(
            inputs=[X_sym],
            outputs=[y_hat_sym],)

    def partial_fit(self, X, y):
        return self.fit_function(X, y.astype('int32'))

    def fit(self, X, y, valid_X=None, valid_y=None):
        input_size = X.shape[1]
        output_size = len(np.unique(y))
        X_sym = T.matrix('x')
        y_sym = T.ivector('y')
        self.layers_ = []
        self.layer_sizes_ = [input_size]
        self.layer_sizes_.extend(self.hidden_layer_sizes)
        self.layer_sizes_.append(output_size)
        self.training_loss_ = []
        self.validation_loss_ = []

        if not hasattr(self, 'fit_function'):
            self._setup_functions(X_sym, y_sym,
                                  self.layer_sizes_)

        batch_indices = list(range(0, X.shape[0], self.batch_size))
        if X.shape[0] != batch_indices[-1]:
            batch_indices.append(X.shape[0])

        best_valid_loss = np.inf
        for itr in range(self.max_iter):
            print("Starting pass %d through the dataset" % itr)
            batch_bounds = list(zip(batch_indices[:-1], batch_indices[1:]))
            # Random minibatches
            self.random_state.shuffle(batch_bounds)
            for start, end in batch_bounds:
                self.partial_fit(X[start:end], y[start:end])
            current_train_loss = self.loss_function(X, y)
            self.training_loss_.append(current_train_loss)

            if (itr % self.save_frequency) == 0 or (itr == self.max_iter):
                f = open(self.model_save_name + "_snapshot.pkl", 'wb')
                cPickle.dump(self, f, protocol=2)
                f.close()

            if valid_X is not None:
                current_valid_loss = self.loss_function(valid_X, valid_y)
                self.validation_loss_.append(current_valid_loss)
                print("Validation loss %f" % current_valid_loss)
                # if we got the best validation score until now, save
                if current_valid_loss < best_valid_loss:
                    best_valid_loss = current_valid_loss
                    f = open(self.model_save_name + "_best.pkl", 'wb')
                    cPickle.dump(self, f, protocol=2)
                    f.close()
        return self

    def predict(self, X):
        return np.argmax(self.predict_function(X), axis=1)
