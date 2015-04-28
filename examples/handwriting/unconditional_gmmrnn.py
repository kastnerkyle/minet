import matplotlib
matplotlib.use('Agg')
from minet.datasets import plot_scatter_iamondb_example
from minet.datasets import plot_lines_iamondb_example
from minet.datasets import fetch_iamondb
import matplotlib.pyplot as plt
from minet import GMMRNN
import numpy as np

X, y = fetch_iamondb()
clf = GMMRNN(learning_alg="rmsprop", n_mixture_components=5,
             hidden_layer_sizes=[500, 500, 500, 500, 500, 500],
             max_iter=500, learning_rate=.00001,
             bidirectional=False, momentum=0.9,
             recurrent_activation="lstm", minibatch_size=1000,
             save_frequency=500, random_seed=1999)

seq = X[0][:, 1:]
mi0 = seq.min(axis=0)
ma0 = seq.max(axis=0)
seq = (seq - mi0) / (ma0 - mi0)
m0 = seq.mean(axis=0)
seq -= m0
seq = seq[1:] - seq[:-1]

clf.fit(seq)
print("Generating unbiased sample")
t1 = clf.sample(n_steps=len(seq))
print("Generating techer forced sample")
t2 = clf.force_sample(seq)
print("Generating biased sample")
t3 = clf.sample(bias=0., n_steps=len(seq))

def undoit(t):
    return ((np.cumsum(t, axis=0) + m0) + mi0) * (ma0 - mi0)

s = undoit(seq)
t1 = undoit(t1)
t2 = undoit(t2)
t3 = undoit(t3)

pltt = plot_scatter_iamondb_example

pltt(s, y[0])
plt.axis('equal')
plt.savefig('seq.png')
plt.clf()

pltt(t1, y[0])
plt.axis('equal')
plt.savefig('t1.png')
plt.clf()

pltt(t2, y[0])
plt.axis('equal')
plt.savefig('t2.png')
plt.clf()

pltt(t3, y[0])
plt.axis('equal')
plt.savefig('t3.png')
plt.clf()
