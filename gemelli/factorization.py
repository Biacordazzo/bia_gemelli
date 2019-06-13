# ----------------------------------------------------------------------------
# Copyright (c) 2019--, gemelli development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file COPYING.txt, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
from numpy.random import randn
from numpy.linalg import norm
from .base import _BaseImpute
from scipy.spatial import distance


class TenAls(_BaseImpute):

    def __init__(
            self,
            rank=3,
            iteration=50,
            ninit=50,
            nitr_RTPM=50,
            tol=1e-8,
            pseudocount=1.0):
        """

        This class performs a low-rank 3rd order
        tensor factorization for partially observered
        non-symmetric sets. This method relies on a
        CANDECOMP/PARAFAC (CP) tensor decomposition.
        Missing values are handled by an alternating
        least squares (ALS) minimization between
        TE and TE_hat.

        Parameters
        ----------
        rank : int, optional
            The underlying low-rank, will be
            equal to the number of rank 1
            components that are output. The
            higher the rank given, the more
            expensive the computation will
            be.
        ninit : int, optional
            The number of initialization
            vectors. Larger values will
            give more accurate factorization
            but will be more computationally
            expensive.
        iteration : int, optional
            Max number of iterations.
        tol : float, optional
            The stopping point in the minimization
            of TE and the factorization between
            each iteration.

        Attributes
        ----------
        eigenvalues : array-like
            The singular value vectors (1,r)
        explained_variance_ratio : array-like
            The percent explained by each
            rank-1 factor. (1,r)
        sample_distance : array-like
            The euclidean distance between
            the sample_loading and it'self
            transposed of shape (samples, samples)
        conditional_loading  : array-like or list of array-like
            The conditional loading vectors
            of shape (conditions, r) if there is 1 type
            of condition, and a list of such matrices if
            there are more than 1 type of condition
        feature_loading : array-like
            The feature loading vectors
            of shape (features, r)
        sample_loading : array-like
            The sample loading vectors
            of shape (samples, r)
        loadings : list of array-like
            A list of loadings for all dimensions
            of the data
        s : array-like
            The r-dimension vector.
        dist : array-like
            A absolute distance vector
            between TE and TE_hat.

        References
        ----------
        .. [1] A. Anandkumar, R. Ge, D. Hsu,
               S. M. Kakade, M. Telgarsky,
               Tensor Decompositions for Learning
               Latent Variable Models
               (A Survey for ALT).
               Lecture Notes in
               Computer Science
               (2015), pp. 19–38.
        .. [2] P. Jain, S. Oh, in Advances in Neural
               Information Processing Systems
               27, Z. Ghahramani, M. Welling,
               C. Cortes, N. D. Lawrence,
               K. Q. Weinberger, Eds.
               (Curran Associates, Inc., 2014),
               pp. 1431–1439.

        Examples
        --------
        >>> from scipy.linalg import qr
        >>> r = 3 # rank is 2
        >>> n1 = 10
        >>> n2 = 10
        >>> n3 = 10
        >>> U01 = np.random.rand(n1, r)
        >>> U02 = np.random.rand(n2, r)
        >>> U03 = np.random.rand(n3, r)
        >>> U1, temp = qr(U01)
        >>> U2, temp = qr(U02)
        >>> U3, temp = qr(U03)
        >>> U1 = U1[:, 0:r]
        >>> U2 = U2[:, 0:r]
        >>> U3 = U3[:, 0:r]
        >>> T = np.zeros((n1, n2, n3))
        >>> for i in range(n3):
        >>>     T[:,:,i] = np.matmul(U1, np.matmul(np.diag(U3[i, :]), U2.T))
        >>> p = 2 * (r ** 0.5 * np.log(n1 * n2 * n3)) / np.sqrt(n1 * n2 * n3)
        >>> E = abs(np.ceil(np.random.rand(n1, n2, n3) - 1 + p))
        >>> E = T * E
        >>> noise = np.random.randn(n1, n2, n3)
        >>> TE_noise = TE + (0.0001 / np.sqrt(n1 * n2 * n3) * noise * E)
        >>> TF = TenAls()
        >>> TF.fit(TE_noise)
        """

        self.rank = rank
        self.iteration = iteration
        self.ninit = ninit
        self.tol = tol
        self.pseudocount = pseudocount
        self.nitr_RTPM = nitr_RTPM

    def fit(self, Tensor):
        """

        Run _fit() a wrapper
        for the tenals helper.

        Parameters
        ----------
        Tensor : array-like
            A tensor, often
            compositionally transformed,
            with missing values. The missing
            values must be zeros. Canonically,
            Tensor must be of the shape:
            first dimension = samples
            second dimension = features
            rest dimensions = types of conditions
        """

        self.sparse_tensor = Tensor.copy()
        self._fit()
        return self

    def _fit(self):
        """
        This function runs the
        tenals helper.

        """

        # make copy for imputation, check type
        sparse_tensor = self.sparse_tensor

        if not isinstance(sparse_tensor, np.ndarray):
            raise ValueError('Input data is should be type numpy.ndarray')

        if (np.count_nonzero(sparse_tensor) == np.product(sparse_tensor.shape) and
                np.count_nonzero(~np.isnan(sparse_tensor)) == np.product(sparse_tensor.shape)):
            raise ValueError('No missing data in the format np.nan or 0')

        if np.count_nonzero(np.isinf(sparse_tensor)) != 0:
            raise ValueError('Contains either np.inf or -np.inf')

        if self.rank > np.max(sparse_tensor.shape):
            raise ValueError('rank must be less than the maximum shape')

        # return tensor decomp
        E = np.zeros(sparse_tensor.shape)
        E[abs(sparse_tensor) > 0] = 1
        loadings, s_, dist = tenals(sparse_tensor,
                                    E,
                                    r=self.rank,
                                    ninit=self.ninit,
                                    nitr=self.iteration,
                                    nitr_RTPM=self.nitr_RTPM,
                                    tol=self.tol,
                                    pseudocount=self.pseudocount)

        self.loadings = loadings
        self.eigenvalues = np.diag(s_)
        self.explained_variance_ratio = \
            list(self.eigenvalues / self.eigenvalues.sum())
        self.sample_distance = distance.cdist(loadings[0], loadings[0])
        self.sample_loading = loadings[0]
        self.feature_loading = loadings[1]
        self.conditional_loading = loadings[2] if len(loadings[2:]) == 1 \
            else loadings[2:]
        self.distances = [distance.cdist(loading, loading) for loading in
                          loadings]
        self.eigenvalues = s_
        self.dist = dist


def tenals(
        TE,
        E,
        r=3,
        ninit=50,
        nitr=50,
        nitr_RTPM=50,
        tol=1e-8,
        pseudocount=1.0):
    """
    A low-rank 3rd order tensor factorization
    for partially observered non-symmetric
    sets. This method relies on a CANDECOMP/
    PARAFAC (CP) tensor decomposition. Missing
    values are handled by  and alternating
    least squares (ALS) minimization between
    TE and TE_hat.

    Parameters
    ----------
    TE : array-like
        A sparse `n` order tensor with zeros
        in place of missing values.
    E : array-like
        A masking array of missing values.
    r : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    ninit : int, optional
        The number of initialization
        vectors. Larger values will
        give more accurate factorization
        but will be more computationally
        expensive.
    nitr : int, optional
        Max number of iterations.
    tol : float, optional
        The stopping point in the minimization
        of TE and the factorization between
        each iteration.

    Returns
    -------
    loadings : list array-like
        The factors of TE. The `i`th entry of loadings corresponds to
        the mode-`i` factors of TE and hase shape (TE.shape[i], r).
    S : array-like
        The r-dimension vector of eigenvalues.
    dist : array-like
        A absolute distance vector
        between TE and TE_hat.

    Raises
    ------
    ValueError
        Nan values in input, factorization
        did not converge.

    References
    ----------
    .. [1] P. Jain, S. Oh, in Advances in Neural
            Information Processing Systems
            27, Z. Ghahramani, M. Welling,
            C. Cortes, N. D. Lawrence,
            K. Q. Weinberger, Eds.
            (Curran Associates, Inc., 2014),
            pp. 1431–1439.

    Examples
    --------
    >>> r = 3 # rank is 2
    >>> n1 = 10
    >>> n2 = 10
    >>> n3 = 10
    >>> U01 = np.random.rand(n1, r)
    >>> U02 = np.random.rand(n2, r)
    >>> U03 = np.random.rand(n3, r)
    >>> U1, temp = qr(U01)
    >>> U2, temp = qr(U02)
    >>> U3, temp = qr(U03)
    >>> U1 = U1[:, 0:r]
    >>> U2 = U2[:, 0:r]
    >>> U3 = U3[:, 0:r]
    >>> T = np.zeros((n1, n2, n3))
    >>> for i in range(n3):
    >>>     T[:,:,i] = np.matmul(U1, np.matmul(np.diag(U3[i, :]), U2.T))
    >>> p = 2 * (r ** 0.5 * np.log(n1 * n2 * n3)) / np.sqrt(n1 * n2 * n3)
    >>> E = abs(np.ceil(np.random.rand(n1, n2, n3) - 1 + p))
    >>> E = T * E
    >>> noise = np.random.randn(n1, n2, n3)
    >>> TE_noise = TE + (0.0001 / np.sqrt(n1 * n2 * n3) * noise * E)
    >>> loadings, eigenvalues, dist = tenals(TE_noise, E)

    """

    # start
    dims = TE.shape

    normTE = norm(TE)**2

    # initialization by Robust Tensor Power Method (modified for non-symmetric
    # tensors)
    S0, U = RTPM(TE, r, ninit, nitr)

    # apply alternating least squares
    V_alt = [Un.copy() for Un in U]
    S_alt = S0.copy()
    for itrs in range(nitr):
        for q in range(r):
            S_alt = S_alt.copy()
            S_alt[q] = 0
            A_alt = np.multiply(CPcomp(S_alt, V_alt), E)
            v_alt = [Vn[:, q].copy() for Vn in V_alt]
            for Vn in V_alt:
                Vn[:, q] = 0

            # den should end up as a list of np.zeros((dim_i, 1))
            den = [np.zeros(dim) for dim in dims]

            for dim, dim_size in enumerate(dims):
                dims_np = np.arange(len(dims))
                dot_across = dims_np[dims_np != dim]
                v_dim = np.tensordot(TE - A_alt,
                                     v_alt[dot_across[0]],
                                     axes=(1 if dim == 0 else 0, 0))
                den[dim] = np.tensordot(E,
                                        v_alt[dot_across[0]]**2,
                                        axes=(1 if dim == 0 else 0, 0))

                for inner_dim in dot_across[1:]:
                    v_dim = np.tensordot(v_dim,
                                         v_alt[inner_dim],
                                         axes=(1 if inner_dim > dim else 0, 0))
                    den[dim] = np.tensordot(den[dim],
                                            v_alt[inner_dim]**2,
                                            axes=(1 if inner_dim > dim else
                                                  0, 0))

                v_alt[dim] = V_alt[dim][:, q] + v_dim.flatten()
                # add pseudocount to prevent division by zero causing nan.
                den[dim][den[dim] == 0] = pseudocount
                v_alt[dim] = v_alt[dim] / den[dim]

                if dim == len(dims) - 1:
                    S_alt[q] = norm(v_alt[dim])

                v_alt[dim] = v_alt[dim] / norm(v_alt[dim])
                V_alt[dim][:, q] = v_alt[dim]

            for i, V_alt_i in enumerate(V_alt):
                V_alt_i[:, q] = v_alt[i]

        ERR = TE - E * CPcomp(S_alt, V_alt)
        normERR = norm(ERR)**2
        if np.sqrt(normERR / normTE) < tol:
            break

    dist = np.sqrt(normERR / normTE)
    # check that the factorization converged
    if any(sum(sum(np.isnan(Vn))) > 0 for Vn in V_alt):
        raise ValueError("The factorization did not converge.",
                         "Please check the input tensor for errors.")

    S_alt = np.diag(S_alt.flatten())
    # sort the eigenvalues
    idx = np.argsort(np.diag(S_alt))[::-1]
    S_alt = S_alt[idx, :][:, idx]
    # sort loadings
    loadings = [Vn[:, idx] for Vn in V_alt]

    return loadings, S_alt, dist


def RTPM(TE, r, ninit, nitr):
    """
    The Robust Tensor Power Method
    (RTPM). Is a generalization of
    the widely used power method for
    computing lead singular values
    of a matrix and can approximate
    the largest singular vectors of
    a tensor.

    Parameters
    ----------
    TE : array-like
        A sparse `n` order tensor with zeros
        in place of missing values.
    r : int, optional
        The underlying low-rank, will be
        equal to the number of rank 1
        components that are output. The
        higher the rank given, the more
        expensive the computation will
        be.
    ninit : int, optional
        The number of initialization
        vectors. Larger values will
        give more accurate factorization
        but will be more computationally
        expensive.
    nitr : int, optional
        Max number of iterations.

    Returns
    -------
    S0 : array-like
        The eigenvalues of the factorizations
    U : list of array-like
        The `i`-th entry of U corresponds to
        the factors along the `i`-th mode of TE

    References
    ----------
    .. [1] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning
            Latent Variable Models
            (A Survey for ALT).
            Lecture Notes in
            Computer Science
            (2015), pp. 19–38.
    .. [2] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.
    .. [3] P. Jain, S. Oh, in Advances in Neural
            Information Processing Systems
            27, Z. Ghahramani, M. Welling,
            C. Cortes, N. D. Lawrence,
            K. Q. Weinberger, Eds.
            (Curran Associates, Inc., 2014),
            pp. 1431–1439.

    """
    dims = TE.shape
    U = [np.zeros((n, r)) for n in dims]
    S0 = np.zeros((r, 1))
    for i in range(r):
        tU = [np.zeros((n, ninit)) for n in dims]
        tS = np.zeros((ninit, 1))
        for init in range(ninit):
            initializations = RTPM_single(
                TE - CPcomp(S0, U), max_iter=nitr)

            for tUn_idx, tUn in enumerate(tU):
                tUn[:, init] = initializations[tUn_idx]
                tUn[:, init] = tUn[:, init] / norm(tUn[:, init])

            tS[init] = TenProjAlt(TE - CPcomp(S0, U),
                                  [tUn[:, [init]] for tUn in tU])

        idx = np.argmin(tS, axis=0)[0]

        for tUn, Un in zip(tU, U):
            Un[:, i] = tUn[:, idx] / norm(tUn[:, idx])

        S0[i] = TenProjAlt(TE - CPcomp(S0, U),
                           [Un[:, [i]] for Un in U])

    return S0, U


def RTPM_single(tensor, max_iter=50):
    """
    Completes a single iteration of optimization
    for a random start of RTPM

    Parameters
    ----------
    tensor : array-like
        an `n` order tensor
    max_iter : int
        maximum iterations.

    Returns
    -------
    list of array-like
        entry `i` of list is a single
        vector corresponding to the `i`th
        mode of `tensor`

    References
    ----------
    .. [1] A. Anandkumar, R. Ge, D. Hsu,
            S. M. Kakade, M. Telgarsky,
            Tensor Decompositions for Learning
            Latent Variable Models
            (A Survey for ALT).
            Lecture Notes in
            Computer Science
            (2015), pp. 19–38.
    .. [2] A. Anandkumar, R. Ge, M., Janzamin,
            Guaranteed Non-Orthogonal Tensor
            Decomposition via Alternating Rank-1
            Updates. CoRR (2014),
            pp. 1-36.
    .. [3] P. Jain, S. Oh, in Advances in Neural
            Information Processing Systems
            27, Z. Ghahramani, M. Welling,
            C. Cortes, N. D. Lawrence,
            K. Q. Weinberger, Eds.
            (Curran Associates, Inc., 2014),
            pp. 1431–1439.

    """

    # RTPM_single
    n_dims = len(tensor.shape)

    all_u = [randn(n, 1) for n in tensor.shape]
    all_u = [vec / norm(vec) for vec in all_u]
    for itr in range(max_iter):
        # tensordot generalization to higher dims
        v = []
        dims = np.arange(n_dims)
        for dim in dims:
            dot_across = dims[dims != dim]
            v_dim = np.tensordot(tensor,
                                 all_u[dot_across[0]],
                                 axes=(1 if dim == 0 else 0, 0))
            for inner_dim in dot_across[1:]:
                v_dim = np.tensordot(v_dim,
                                     all_u[inner_dim],
                                     axes=(1 if inner_dim > dim else 0, 0))
            v.append(v_dim)

        new_shapes = [v_n.shape[:(-1 * (len(dims) - 2))] for v_n in v]
        v = [v_n.reshape(new_shape) for v_n, new_shape in zip(v, new_shapes)]

        all_u_previous = [u for u in all_u]
        all_u = [v_i / norm(v_i) for v_i in v]

        if sum(norm(u0 - u) for u0, u in zip(all_u_previous, all_u)) < 1e-7:
            break

    return [u.flatten() for u in all_u]


def CPcomp(S, U):
    """
    This function takes the
    CP decomposition of a 3rd
    order tensor and outputs
    the reconstructed tensor
    TE_hat.

    Parameters
    ----------
    S : array-like
        The r-dimension vector.
    U : list of array-like
        Element i is a factor of shape
        (n[i], r).

    Returns
    -------
    T : array-like
        TE_hat of shape
        tuple(n[i] for i in range(len(U))).
    """

    output_shape = tuple(u.shape[0] for u in U)
    to_multiply = [S.T * u if i == 0 else u for i, u in enumerate(U)]
    product = khatri_rao(to_multiply)
    T = product.sum(1).reshape(output_shape)

    return T


def TenProjAlt(D, U_list):
    """
    The Orthogonal tensor
    projection created by
    the TE - TE_hat distance.
    Used in the initialization
    step with RTPM_single.

    Parameters
    ----------
    D : array-like
        with shape (n[0], n[1], ..., )
    U_list : list of array-like
        Element i is a factor of shape
        (n[i], r). Same length as D.shape

    Returns
    -------
    M : float
        The multilinear mapping of D on U_list
    """
    current = D
    for u in U_list:
        current = np.tensordot(current, u, axes=(0, 0))
    return current


def khatri_rao(matrices):
    """
    Returns the Khatri Rao product of a list of matrices

    Modified from TensorLy

    Parameters
    ----------
    matrices : list of array-like
        Matrices to take the Khatri Rao Product of

    Returns
    -------
    array-like
        The Khatri Rao Product of the matrices in `matrices`

    References
    ----------
    .. [1] Jean Kossaifi, Yannis Panagakis, Anima Anandkumar and Maja
            Pantic, TensorLy: Tensor Learning in Python,
            https://arxiv.org/abs/1610.09555.
    """

    n_columns = matrices[0].shape[1]
    n_factors = len(matrices)

    start = ord('a')
    common_dim = 'z'
    target = ''.join(chr(start + i) for i in range(n_factors))
    source = ','.join(i + common_dim for i in target)
    operation = source + '->' + target + common_dim

    return np.einsum(operation, *matrices).reshape((-1, n_columns))