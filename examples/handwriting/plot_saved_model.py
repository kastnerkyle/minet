from minet.datasets import plot_scatter_iamondb_example
from minet.datasets import fetch_iamondb
import matplotlib.pyplot as plt
import sys
import numpy as np
import cPickle


X, y = fetch_iamondb()
seq = X[0][:, 1:]
#seq = seq[1:] - seq[:-1]
seq = seq[:100]
mi0 = seq.min(axis=0)
ma0 = seq.max(axis=0)
seq = (seq - mi0) / (ma0 - mi0)

model_path = sys.argv[1]
f = open(model_path, mode='rb')
clf = cPickle.load(f)
f.close()
from IPython import embed; embed()
raise ValueError()

plt.plot(clf.training_loss_)
plt.savefig('training.png')
plt.clf()

t1 = clf.sample(n_steps=len(seq))
t2 = clf.force_sample(seq)
t3 = clf.sample(bias=0., n_steps=len(seq))

def undoit(t):
    return t * (ma0 - mi0) + mi0
#    return np.cumsum(t * (ma0 - mi0) + mi0, axis=0)

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
