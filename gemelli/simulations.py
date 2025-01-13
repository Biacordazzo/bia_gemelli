from __future__ import division
# utils
import pandas as pd
import numpy as np
# blocks
from scipy.stats import norm
from numpy.random import poisson, lognormal
from skbio.stats.composition import closure
# Set random state
rand = np.random.RandomState(42)


def Homoscedastic(X_noise, intensity):
    """ uniform normally dist. noise """
    X_noise = np.array(X_noise)
    err = intensity * np.ones_like(X_noise.copy())
    X_noise = rand.normal(X_noise.copy(), err)

    return X_noise


def Heteroscedastic(X_noise, intensity):
    """ non-uniform normally dist. noise """
    err = intensity * np.ones_like(X_noise)
    i = rand.randint(0, err.shape[0], 5000)
    j = rand.randint(0, err.shape[1], 5000)
    err[i, j] = intensity
    X_noise = abs(rand.normal(X_noise, err))

    return X_noise


def Subsample(X_noise, spar, num_samples):
    """ yij ~ PLN( lambda_{ij}, /phi ) """
    # subsample
    mu = spar * closure(X_noise.T).T
    X_noise = np.vstack([poisson(lognormal(np.log(mu[:, i]), 1))
                         for i in range(num_samples)]).T
    # add sparsity

    return X_noise


def block_diagonal_gaus(
        ncols,
        nrows,
        nblocks,
        overlap=0,
        minval=0,
        maxval=1.0):
    """
    Generate block diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    ncol : int
        Number of columns

    nrows : int
        Number of rows

    nblocks : int
        Number of blocks, mucst be greater than one

    overlap : int
        The Number of overlapping columns (Default = 0)

    minval : int
        The min value output of the table (Default = 0)

    maxval : int
        The max value output of the table (Default = 1)


    Returns
    -------
    np.array
        Table with a block diagonal where the rows represent samples
        and the columns represent features.  The values within the blocks
        are gaussian distributed between 0 and 1.
    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.

    """

    if nblocks <= 1:
        raise ValueError('`nblocks` needs to be greater than 1.')
    mat = np.zeros((nrows, ncols))
    gradient = np.linspace(0, 10, nrows)
    mu = np.linspace(0, 10, ncols)
    sigma = 1
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    mat = np.vstack(xs).T

    block_cols = ncols // nblocks
    block_rows = nrows // nblocks
    for b in range(nblocks - 1):

        gradient = np.linspace(
            5, 5, block_rows)  # samples (bock_rows)
        # features (block_cols+overlap)
        mu = np.linspace(0, 10, block_cols + overlap)
        sigma = 2.0
        xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
              for i in range(len(mu))]

        B = np.vstack(xs).T * maxval
        lower_row = block_rows * b
        upper_row = min(block_rows * (b + 1), nrows)
        lower_col = block_cols * b
        upper_col = min(block_cols * (b + 1), ncols)

        if b == 0:
            mat[lower_row:upper_row,
                lower_col:int(upper_col + overlap)] = B
        else:
            ov_tmp = int(overlap / 2)
            if (B.shape) == (mat[lower_row:upper_row,
                                 int(lower_col - ov_tmp):
                                 int(upper_col + ov_tmp + 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp + 1)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp)] = B
            elif (B.shape) == (mat[lower_row:upper_row,
                                   int(lower_col - ov_tmp):
                                   int(upper_col + ov_tmp - 1)].shape):
                mat[lower_row:upper_row, int(
                    lower_col - ov_tmp):int(upper_col + ov_tmp - 1)] = B

    upper_col = int(upper_col - overlap)
    # Make last block fill in the remainder
    gradient = np.linspace(5, 5, nrows - upper_row)
    mu = np.linspace(0, 10, ncols - upper_col)
    sigma = 4
    xs = [norm.pdf(gradient, loc=mu[i], scale=sigma)
          for i in range(len(mu))]
    B = np.vstack(xs).T * maxval

    mat[upper_row:, upper_col:] = B

    return mat


def shape_noise(X,
                fxs,
                f_intervals,
                s_intervals,
                n_timepoints=10,
                col_handle='individual'):
    """
    Adds x-shaped noise (e.g. sine, sigmoid) to the
    true data

    Parameters
    ----------
    X : np.array
        The true data

    fxs : list
        List of functions to apply to the data

    f_intervals : list
        List of tuples of the form (f1, f2) where
        f1 is the start index and f2 is the end index
        of the features to apply the function to

    s_intervals : list
        List of tuples of the form (s1, s2) where
        s1 is the start index and s2 is the end index
        of the samples to apply the function to

    n_timepoints : int
        Number of timepoints per individual
        Assumes that all individuals have the
        same number of timepoints

    col_handle : str
        How to handle  (individuals)
        'individual': apply function to all
            timepoints in each individual
        'all': apply function to all columns

    Returns
    -------
    np.array
        The data with x-shaped noise added
    """

    # get shape of true data
    rows, cols = X.shape

    # loop through functions
    for func, features, individuals in zip(fxs, f_intervals,
                                           s_intervals):

        for f_coord, s_coord in zip(features, individuals):
            f1, f2 = f_coord
            s1, s2 = tuple(int(idx/n_timepoints) for idx in s_coord)
            # get sample subset
            if col_handle == 'individual':
                # loop through individuals
                for i in range(s1, s2):
                    idx1 = i*n_timepoints
                    idx2 = (i+1)*n_timepoints
                    X_sub = X[f1:f2, idx1:idx2]
                    X_sub_noise = np.apply_along_axis(func,
                                                      tps=10,
                                                      axis=1,
                                                      arr=X_sub)
                    # update data
                    X[f1:f2, idx1:idx2] = X_sub_noise
            else:
                X_sub = X[f1:f2, :]
                X_sub_noise = np.apply_along_axis(func,
                                                  tps=cols,
                                                  axis=1,
                                                  arr=X_sub)
                # update data
                X[f1:f2, :] = X_sub_noise
    return X


def build_block_model(
        rank,
        hoced,
        hsced,
        spar,
        C_,
        num_samples,
        num_features,
        num_timepoints,
        col_handle='individual',
        overlap=0,
        fxs=None,
        f_intervals=None,
        s_intervals=None,
        mapping_on=True,
        X_noise=None):
    """
    Generates hetero and homo scedastic noise on base truth block
    diagonal with Gaussian distributed values within blocks.

    Parameters
    ----------

    rank : int
        Number of blocks

    hoced : int
        Amount of homoscedastic noise

    hsced : int
        Amount of heteroscedastic noise

    inten : int
        Intensity of the noise

    spar : int
        Level of sparsity

    C_ : int
        Intensity of real values

    num_features : int
        Number of rows

    num_samples : int
        Number of columns

    num_timepoints : int
        Number of timepoints per individual. Assumes all
        individuals have the same number.

    col_handle : str
        How to handle  (individuals)
        'individual': apply function to all
            timepoints in each individual
        'all': apply function to all columns

    overlap : int
        The Number of overlapping columns (Default = 0)

    fxs : list
        List of functions to apply to the data

    f_intervals : list
        List of tuples of the form (f1, f2) where
        f1 is the start index and f2 is the end index
        of the features to apply the function to

    s_intervals : list
        List of tuples of the form (s1, s2) where
        s1 is the start index and s2 is the end index
        of the samples to apply the function to

    mapping_on : bool
        if true will return pandas dataframe mock mapping file by block

    X_noise: np.array, default is None
        Data with pre-added gaussian noise. Use this to ensure
        the same underlying data is used for multiple simulations

    Returns
    -------
    Pandas Dataframes
    Table with a block diagonal where the rows represent samples
    and the columns represent features.  The values within the blocks
    are gaussian.

    Note
    ----
    The number of blocks specified by `nblocks` needs to be greater than 1.
    """

    # make a mock OTU table
    X_true = block_diagonal_gaus(num_samples,
                                 num_features,
                                 rank, overlap,
                                 minval=.01,
                                 maxval=C_)
    if X_noise is None:
        if mapping_on:
            # make a mock mapping data
            mappning_ = pd.DataFrame(np.array([['Cluster %s' % str(x)] *
                                     int(num_samples / rank)
                                     for x in range(1, rank + 1)]).flatten(),
                                     columns=['example'],
                                     index=['sample_' + str(x)
                                            for x in range(1, num_samples+1)])
        X_noise = X_true.copy()
        X_noise = np.array(X_noise)
        # add Homoscedastic noise
        X_noise = Homoscedastic(X_noise, hoced)
        # add Heteroscedastic noise
        X_noise = Heteroscedastic(X_noise, hsced)
        # Induce low-density into the matrix
        X_noise = Subsample(X_noise, spar, num_samples)

    if fxs is not None:
        X_signal = X_noise.copy()
        # introduce specific signal(s)
        X_signal = shape_noise(X_signal, fxs,
                               f_intervals, s_intervals,
                               n_timepoints=num_timepoints,
                               col_handle=col_handle)
    else:
        X_signal = X_noise.copy()

    # return the base truth and noisy data
    if mapping_on:
        return X_true, X_noise, X_signal, mappning_
    else:
        return X_true, X_noise, X_signal
