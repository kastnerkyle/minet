from minet import GMMRNN
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt

n_h = 50
time_steps = 100
# n_y is equal to the number of classes
random_state = np.random.RandomState(1999)

sin = np.sin(np.linspace(-3 * np.pi, 3 * np.pi, time_steps))[:, None]
cos = np.cos(np.linspace(-3 * np.pi, 3 * np.pi, time_steps))[:, None]
seq = np.concatenate((sin, cos), axis=-1)[None]

clf = GMMRNN(learning_alg="rmsprop", n_mixture_components=20,
             hidden_layer_sizes=[n_h],
             max_iter=1000, learning_rate=.0001,
             bidirectional=False, momentum=0.99,
             recurrent_activation="lstm", minibatch_size=10,
             save_frequency=1000, random_seed=1999)

shp = seq.shape
seq_r = seq.reshape(shp[0] * shp[1], shp[-1])
mean = seq_r.mean(axis=0, keepdims=True)
std = seq_r.std(axis=0, keepdims=True)
seq = seq - mean
seq = seq / std

clf.fit(seq)

t1 = clf.sample(n_steps=time_steps)
t2 = clf.force_sample(seq[0])
t3 = clf.sample(bias=0., n_steps=time_steps)


def rescale(X):
    return X * std + mean


def rel_to_abs(X):
    return np.cumsum(rescale(X), axis=-1)

t1 = rescale(t1)
t2 = rescale(t2)
t3 = rescale(t3)

plt.plot(rescale(seq[0]))
plt.savefig('seq.png')
plt.clf()
plt.plot(t1)
plt.savefig('t1.png')
plt.clf()
plt.plot(t2)
plt.savefig('t2.png')
plt.clf()
plt.plot(t3)
plt.savefig('t3.png')
plt.clf()
