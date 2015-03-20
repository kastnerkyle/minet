from minet import GMMRNN
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

n_u = 2
n_h =10
time_steps = 100
n_seq = 20
# n_y is equal to the number of classes
random_state = np.random.RandomState(1999)

"""
mu1 = np.array((20, 50))
var1 = np.array((1, 1))
mu2 = np.array((-50, -20))
var2 = np.array((7, 7))
seq = np.zeros((n_seq, time_steps, n_u))
for i in range(time_steps):
    if i < (time_steps // 2):
        samp = mu1[None] + np.sqrt(var1[None]) * random_state.randn(n_seq, n_u)
    else:
        samp = mu2[None] + np.sqrt(var2[None]) * random_state.randn(n_seq, n_u)
    seq[:, i, :] = samp
"""
sin = np.sin(np.linspace(-4 * np.pi, 4 * np.pi, time_steps))
seq = sin[None, :, None]

clf = GMMRNN(learning_alg="rmsprop", n_mixture_components=20,
             window_size=20, prediction_size=5,
             hidden_layer_sizes=[n_h],
             max_iter=1000, learning_rate=.001,
             bidirectional=False, momentum=0.99,
             recurrent_activation="lstm", minibatch_size=10,
             save_frequency=500, random_seed=1999)

shp = seq.shape
seq_r = seq.reshape(shp[0] * shp[1], shp[-1])
mean = seq_r.mean(axis=0)
std = seq_r.std(axis=0)
seq = seq - mean[None, None]
seq = seq / std[None, None]

clf.fit(seq)
t1 = clf.sample(init_seed=seq[0, :50, :], n_steps=100)
t2 = clf.force_sample(seq[0])


def rel_to_abs(X):
    return np.cumsum(X * std[None] + mean[None], axis=-1)

plt.plot(rel_to_abs(seq[0]))
plt.savefig('seq.png')
plt.clf()
plt.plot(rel_to_abs(t1))
plt.savefig('t1.png')
plt.clf()
plt.plot(rel_to_abs(t2))
plt.savefig('t2.png')
plt.clf()
