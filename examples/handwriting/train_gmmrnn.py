import matplotlib
matplotlib.use('Agg')
from minet.datasets import plot_scatter_iamondb_example
from minet.datasets import plot_lines_iamondb_example
from minet.datasets import fetch_iamondb
import matplotlib.pyplot as plt
from minet import AGMMRNN
import numpy as np

X, y = fetch_iamondb()
clf = AGMMRNN(learning_alg="rmsprop", n_mixture_components=5,
              hidden_layer_sizes=[500],
              max_iter=25000, learning_rate=.00001,
              bidirectional=False, momentum=0.8,
              recurrent_activation="lstm", minibatch_size=1000,
              save_frequency=100, random_seed=1999)

seq = X[0]
#seq = seq[1:] - seq[:-1]
seq = seq[:100]
mi0 = seq.min(axis=0)
ma0 = seq.max(axis=0)
seq = (seq - mi0) / (ma0 - mi0)
clf.fit(seq)
