import theano.tensor as T


def logsumexp(X, axis=None):
    """
    Blatantly "borrowed" from cle
    """
    X_max = T.max(X, axis=axis, keepdims=True)
    Z = T.log(T.sum(T.exp(X - X_max), axis=axis, keepdims=True)) + X_max
    return Z.sum(axis=axis)
