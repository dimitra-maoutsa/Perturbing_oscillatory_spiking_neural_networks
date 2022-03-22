import numpy as np
import pylab as pl

def init_fig():
    pl.rc('figure', facecolor='white', dpi=90)
    pl.rc('image', cmap='gray_r', interpolation='nearest')
    pl.rc('xtick', labelsize=10)
    pl.rc('ytick', labelsize=10)
    pl.rc('xtick.major', pad=2, size=2)
    pl.rc('ytick.major', pad=2, size=2)
    pl.rc('font', **{'sans-serif' : 'Arial', 'weight' : 'light', 'family' : 'sans-serif', 'size' : 10})
    pl.rc('axes', titlesize=10, linewidth=0.5, labelsize=10)
    pl.rc( 'lines', linewidth=0.5)
    fig = pl.figure(figsize=(13,5))
    axspike = fig.add_axes((0.04,0.15,0.35,0.78))
    axprob = fig.add_axes((0.48,0.15,0.50,0.78))
    axspike.set_xlabel("Time")
    axspike.set_ylabel("Neuron")
    axprob.set_xlabel("State")
    axprob.set_ylabel("Probability")
    axspike.set_title("Spike pattern and z-state for 0 < t < 1000")
    axprob.set_title("Comparison of network samples and analytic distribution (first 5 neurons only)")
    return fig, axspike, axprob


def joint_from_data(Z, idx=None, priorObservations=0.):
    """
    Extract joint over binary z_k, k in idx, and marginalizes over other z_i, i not in idx
        Z: binary data vector of shape (T,K) with T = #samples and K = #variables
        idx: List of indices; None means all indices
        priorObservations: Assume priorObservations of each state (bias for counting)
    """
    if idx is None:
        idx = np.arange(Z.shape[-1])
    Z = Z[:,idx]
    T,K = Z.shape
    b = [2**i for i in reversed(range(K))]
    N = np.bincount(np.dot(Z,b))
    P = priorObservations * np.ones(2**K)
    P[:len(N)] += N
    P /= P.sum()
    states = [np.binary_repr(n,K) for n in xrange(2**K)]
    return states, P


def joint_of_BM(b, W):
    """
    Analytical calculation of Boltzmann machine.
        b, W: params as usual
    """
    from itertools import product as iterprod
    assert len(b) == len(W)
    assert (W.diagonal() == 0.).all()
    K = len(b)
    E = lambda z: -np.dot(z, 0.5 * np.dot(W,z) + b )
    P = np.zeros(2**K)
    for i,z in enumerate(iterprod(*[[0,1]]*K)):
        P[i] = np.exp(-E(z))
    P /= P.sum()
    states = [np.binary_repr(n,K) for n in xrange(2**K)]
    return states, P
