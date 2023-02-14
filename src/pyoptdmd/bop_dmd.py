import copy

import munkres
import scipy
import numpy as np
from pydmd.bopdmd import BOPDMD


def match_vectors(vector1, vector2):
    """ Wrapper for MUNKRES converted from MATLAB.

    Sets up a cost function so that the indices
    returned by munkres correspond to the permutation
    which minimizes the 1-norm of the difference
    between vector1[indices] and vector2[indices ~= 0].

    The python version of munkres returns both the column and
    row indices whereas the MATLAB version returns just the
    column indices.

    Example:

    row_indices, col_indices = match_vectors(vector1,vector2)

    See also the munkres package.
    """

    m = munkres.Munkres()

    # Deliberately expand to have matlab-like vectors
    if vector1.ndim == 1:
        vector1 = vector1[:, np.newaxis]
    if vector2.ndim == 1:
        vector2 = vector2[:, np.newaxis]

    if scipy.sparse.issparse(vector1) or scipy.sparse.issparse(vector2):
        raise ValueError('Numpy Kronecker product can fail silently with sparse matrices')

    cost_matrix = np.abs(np.kron(vector1.T, np.ones((len(vector2), 1))) - np.kron(vector2,
                                                                                  np.ones(
                                                                                      (1,
                                                                                       len(vector1)))))
    # indices, cost = munkres(costmat)
    indices = m.compute(cost_matrix)

    # For the use cases currently implementing this method, it is desirable to have the
    # row and column indices separate.
    row_indices, col_indices = list(zip(*indices))

    # Return numpy arrays instead of lists
    return np.array(row_indices), np.array(col_indices)


def fit(xdata, ts, modes, num_ensembles=10, ensemble_size=None,
        seed=None, long_term_mean=None, long_term_ts=None, forecast_x=None,
        t_forecast=None, pydmd_kwargs=None):
    """

    Warnings: currently only works on 2D data with a spatial dimension and a time
    dimension. Anything else will require more careful specification of dimensions.

    """

    if pydmd_kwargs is None:
        pydmd_kwargs = {}

    spatial_length = xdata.shape[0]
    time_length = ts.size

    if not time_length == xdata.shape[-1]:
        raise ValueError('xdata and ts did not have the expected shape.')

    # Create the lambda vector for ensembleDMD cycle
    e_ensembleDMD = np.zeros((modes, num_ensembles)).astype(complex)
    bopdmd_ensemble = []
    # w_ensembleDMD = np.zeros((spatial_length, modes, num_ensembles)).astype(complex)
    # b_ensembleDMD = np.zeros((modes, num_ensembles)).astype(complex)

    # Substantiate the random generator.
    rng = np.random.default_rng(seed)

    if ensemble_size is None:
        ensemble_size = ts.size // 2

    optdmd = BOPDMD(svd_rank=modes, num_trials=0, **pydmd_kwargs)

    # Try the optdmd without bagging to get an initial guess.
    # Extend the data using an assumption about the long term behavior.
    # if long_term_ts is not None and long_term_mean is not None:
    #     xdata_ext = np.append(xdata, long_term_mean, axis=1)
    #     ts_ext = np.append(ts, long_term_ts, axis=1)
    #     optdmd.fit(xdata_ext, ts_ext)
    #     e_opt = optdmd.eigs
    # else:
    optdmd.fit(xdata, ts)
    e_opt = optdmd.eigs

    j = 0

    while j < num_ensembles:
        bopdmd = BOPDMD(
            svd_rank=modes, num_trials=0, init_alpha=optdmd.eigs, **pydmd_kwargs)

        # Randomly select time indices for this ensemble member.
        # ind = rng.integers(low=0, high=ts.size - 1, size=ensemble_size)
        # Random selection without replacement.
        ind = rng.choice(ts.size, size=ensemble_size, replace=False)

        # Sort the index to be in ascending order, generating variable length time steps.
        ind = np.sort(ind)

        # Create the sub-selected data for this ensemble member using the sorted indices.
        xdata_cycle = xdata[:, ind]
        ts_ind = ts[:, ind]

        # Solve optdmd for this ensemble member.
        # Potential change: Use optDMD modes as initial conditions for BOP-DMD instead
        # of trapezoidal solution?
        try:
            # Extend the data using an assumption about the long term behavior.
            if long_term_ts is not None and long_term_mean is not None:
                xdata_cycle = np.append(xdata_cycle, long_term_mean, axis=1)
                ts_ind = np.append(ts_ind, long_term_ts, axis=1)

            bopdmd.fit(xdata_cycle, ts_ind)
        except np.linalg.LinAlgError:
            # Some draws with the long term prior will create an error. Continue drawing
            # until the fit stops failing.
            continue

        e_bop = bopdmd.eigs
        bopdmd_ensemble.append(copy.copy(bopdmd))

        # Match the eigenvalues to those from first optdmd call using the Munkres
        # algorithm. This step ensures that the eigenvalues have about the right
        # ordering.
        _, indices = match_vectors(e_bop, e_opt)

        # Assign to the outer container using the correct ordering.
        e_ensembleDMD[:, j] = e_bop[indices].flatten()
        j += 1

    return e_ensembleDMD, bopdmd_ensemble, e_opt, optdmd


if __name__ == "__main__":
    # Generate the synthetic data.

    # Set up modes in space.
    x0 = 0
    x1 = 1
    nx = 200

    # Space component is evenly spaced originally.
    xspace = np.linspace(x0, x1, nx)

    # Set up the spatial modes
    f1 = np.sin(xspace)[:, np.newaxis]
    f2 = np.cos(xspace)[:, np.newaxis]
    f3 = np.tanh(xspace)[:, np.newaxis]

    # Set up the time dynamics.
    t0 = 0
    t1 = 1
    nt = 100
    ts = np.linspace(t0, t1, nt)[np.newaxis, :]

    # Eigenvalues for each mode
    e1 = 1 + 0j
    e2 = -2 + 0j
    e3 = 0 + 1j
    true_eigenvalues = np.array([e1, e2, e3])

    # Generate the clean, noiseless dynamics.
    xclean = f1.dot(np.exp(e1 * ts)) + f2.dot(np.exp(e2 * ts)) + f3.dot(np.exp(e3 * ts))

    # Set the random seed
    rng = np.random.default_rng(1)

    # Number of time points
    n = len(ts)
    sigma = 1e-1

    # Create data for noise cycle (add random noise)
    xdata = xclean + sigma * rng.standard_normal(xclean.shape)

    # e, w, b = fit(xdata, ts, 3, num_ensembles=10, ensemble_size=50, verbose=False,
    #               seed=None)
