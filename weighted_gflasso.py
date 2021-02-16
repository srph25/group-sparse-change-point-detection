import numpy as np
import cvxpy as cp


def run(X, lmbd, M=None, diff_k=1, norm_type=2, verbose=False, use_indirect=True, gpu=False, max_iters=10000):

    #assume T x F shape
    assert(isinstance(X, np.ndarray) and X.ndim == 2)
    assert(lmbd >= 0)
    if M is None:
        M = np.ones(X.shape)
    else:
        assert(isinstance(M, np.ndarray) and M.ndim == 2)
    assert(isinstance(diff_k, int) and diff_k >=1)
    assert(norm_type in [1, 2, 'inf'])

    Y = cp.Variable(shape=(X.shape[0], X.shape[1]))
    obj = cp.Minimize(1/2 * cp.square(cp.norm(cp.multiply(M,(X-Y)), 'fro')) + lmbd * cp.mixed_norm(cp.diff(Y, k=diff_k, axis=0), p=norm_type, q=1))
    prob = cp.Problem(obj)
    prob.solve(solver='SCS', verbose=verbose, use_indirect=use_indirect, gpu=gpu, max_iters=max_iters)
    return Y.value

