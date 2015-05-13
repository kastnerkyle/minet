from minet.datasets import plot_scatter_iamondb_example
from minet.datasets import fetch_iamondb
import matplotlib.pyplot as plt
import sys
import numpy as np
import cPickle


X, y = fetch_iamondb()
seq = X[0]
seq = seq[:100]
seq_delta = seq[:, 1:]
#seq_delta = seq[1:, 1:] - seq[:-1, 1:]
mi0 = seq_delta.min(axis=0)
ma0 = seq_delta.max(axis=0)
seq_delta = (seq_delta - mi0) / (ma0 - mi0)
"""
sm = seq_delta.mean(axis=0)
ss = seq_delta.std(axis=0)
seq_delta = (seq_delta - sm) / ss
"""
seq[0, :] = 0.
seq[:, 1:] = seq_delta

model_path = sys.argv[1]
f = open(model_path, mode='rb')
clf = cPickle.load(f)
f.close()

plt.plot(clf.training_loss_)
plt.savefig('training.png')
plt.clf()

ss = seq[:75]
t1 = clf.sample(n_steps=len(seq), seed_sequence=ss)
t2 = clf.force_sample(seq)
t3 = clf.sample(n_steps=len(seq), bias=0., seed_sequence=ss)

def undoit(t):
    t2 = t
    t2[:, 1:] = t2[:, 1:] * (ma0 - mi0) + mi0
    #np.cumsum(t2[:, 1:] * (ma0 - mi0) + mi0, axis=0)
    return t2

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
#plt.axis('equal')
plt.savefig('t1.png')
plt.clf()

pltt(t2, y[0])
#plt.axis('equal')
plt.savefig('t2.png')
plt.clf()

pltt(t3, y[0])
#plt.axis('equal')
plt.savefig('t3.png')
plt.clf()
