import copy
import numpy as np
import pandas as pd
# import warnings
from scipy.sparse.linalg import svds
from gemelli.optspace import svd_sort
from gemelli.ctf import ctf_table_processing
from gemelli.preprocessing import build_sparse
from gemelli._defaults import (DEFAULT_COMP,
                               DEFAULT_TEMPTED_EP,
                               DEFAULT_TEMPTED_SMTH,
                               DEFAULT_TEMPTED_RES,
                               DEFAULT_TEMPTED_MAXITER,
                               DEFAULT_TEMPTED_RH as DEFAULT_TRH,
                               # DEFAULT_TEMPTED_RHC as DEFAULT_RC,
                               DEFAULT_TEMPTED_SVDC,
                               DEFAULT_TEMPTED_SVDCN as DEFAULT_TSCN)

from gemelli.tempted import (freg_rkhs, bernoulli_kernel)


class concat_tensors():
    '''
    Concatenate the tensors from each modality into a
    single tensor class

    Parameters
    ----------
    tensors: dictionary, required
        Dictionary of tensors constructed.
        keys = modality
        values = tensor, required
            rows = features
            columns = samples

    Returns
    ----------
    self: object
        Returns the instance itself
    '''

    def __init__(self):
        pass

    def concat(self, tensors):
        '''
        Concatenate tensors from each modality into a
        single tensor. Note: tensors should have been
        preprocessed by this point.
        '''

        individual_id_tables = {}
        individual_id_state_orders = {}
        mod_id_ind = {}

        for mod, tensor in tensors.items():

            # concatenate tables
            tensor_items = tensor.individual_id_tables_centralized.items()
            for ind_id, table in tensor_items:
                individual_id_tables_ = individual_id_tables.get(ind_id, [])
                individual_id_tables[ind_id] = individual_id_tables_ + [table]
                mod_id_ind[mod] = mod_id_ind.get(mod, []) + \
                    [(ind_id, len(individual_id_tables[ind_id])-1)]
            # concatenate state orders
            for ind_id, order in tensor.individual_id_state_orders.items():
                ind_id_orders_ = individual_id_state_orders.get(ind_id, [])
                individual_id_state_orders[ind_id] = ind_id_orders_ + [order]

        # store all to self
        self.individual_id_tables = individual_id_tables
        self.individual_id_state_orders = individual_id_state_orders
        self.mod_id_ind = mod_id_ind

        return self


def update_residuals(table_mods, a_hat, b_hats,
                     phi_hats, times, lambdas):
    '''
    Update the tensor to be factorized by subtracting the
    approximation the previous iteration. In other words,
    calculate the residuals.

    Parameters
    ----------
    table_mods: dictionary, required
        Tables for each modality
        keys = modality
        values = DataFrame
            rows = features
            columns = samples

    a_hat: np.narray, required
        Subject loadings

    b_hats: dictionary, required
        Feature loadings
        keys = modality
        values = loadings

    phi_hats: dictionary, required
        Temporal loadings
        keys = modality
        values = loadings

    times: dictionary, required
        Time points for each modality
        keys = modality
        values = list of numpy.ndarray
            list[0] = time points within interval
            list[1] = individual indexes

    lambdas: dictionary, required
        Singular values
        keys = modality
        values = loadings

    Returns
    ----------
    tables_update: dictionary
        Residuals for each modality
        keys = modality
        values = DataFrame
            rows = features
            columns = samples
    '''

    tables_update = copy.deepcopy(table_mods)
    rsquared = {}

    for modality in tables_update.keys():

        # get key modality-specific variables
        table_mod = tables_update[modality]
        b_hat = b_hats[modality]
        phi_hat = phi_hats[modality]
        ti = times[modality][0]
        lambda_coeff = lambdas[modality]

        residual_mod = []
        y = []

        for i, (individual_id, m) in enumerate(table_mod.items()):
            y.append(np.concatenate(m.T.values))
            phi_ = phi_hat[ti[i]]
            new_m = np.outer(b_hat, phi_.T)
            new_m = a_hat[i] * new_m
            table_mod[individual_id] -= (lambda_coeff * new_m)
            residual_mod.append(table_mod[individual_id]**2)

        residual_mod = np.sum(np.concatenate(residual_mod, axis=1))
        y = np.concatenate(y)
        rsquared[modality] = 1 - residual_mod / (y.size * y.var())
        tables_update[modality] = table_mod

    return tables_update, rsquared


def get_prop_var(individual_loadings,
                 feature_loadings, lambda_coeffs,
                 n_components, centering=True):
    '''
    Get the proportion of variance explained by each component

    Parameters
    ----------
    individual_loadings: dataframe, required
        Subject loadings
        index = individual IDs
        columns = component

    feature_loadings: dictionary, required
        Feature loadings
        keys = modality
        values = dataframe of loadings for each component

    lambda_coeffs: dataframe, required
        Singular values
        index = modality
        columns = component

    n_components: int, required
        The underlying rank of the data and number of
        output dimentions.

    centering: bool, optional : Default is True
        Whether to re-center using a final svd

    Returns
    ----------
    var_explained: dictionary
        Proportion of variance explained by the
        lambdas in each modality
        keys = modality
        values = component
    '''

    # initialize subject and feature matrices
    n_individuals_all = individual_loadings.shape[0]
    a_hat_mat = pd.DataFrame(np.zeros((n_individuals_all, n_components)),
                             index=individual_loadings.index,
                             columns=individual_loadings.columns)
    
    # sort and center individual loadings
    for component in individual_loadings.columns:
        a_hat = individual_loadings[component]
        a_hat -= a_hat.mean(axis=0)
        a_hat_mat[component] = a_hat

    # initialize dataframe to store feature loadings
    b_hat_mat_lst = []

    for modality in feature_loadings.keys():

        b_hats = feature_loadings[modality]
        lambdas = lambda_coeffs.loc[modality]
        b_hats_centered = {}

        # sort and center feature loadings
        for component in b_hats.keys():
            b_hat = b_hats[component].copy()
            b_hat -= b_hat.mean(axis=0)
            # multiply by singular value
            lambda_mod = lambdas[component]
            b_hats_centered[component] = lambda_mod * b_hat

        # create dataframe block
        b_hat_mat_mod = pd.DataFrame(b_hats_centered).T
        b_hat_mat_lst.append(b_hat_mat_mod)

    # concat feature matrix
    b_hat_mat = pd.concat(b_hat_mat_lst, axis=1)

    if centering:
        # re-center using a final svd
        X = a_hat_mat.values @ b_hat_mat.values
        possible_comp = [np.min(X.shape), n_components]
        biplot_components = np.min(possible_comp)
        X = X - X.mean(axis=0)
        X = X - X.mean(axis=1).reshape(-1, 1)
        u, s, v = svds(X, k=biplot_components, which='LM')
        u, s, v = svd_sort(u, np.diag(s), v)
        p = s * (1 / s.sum())
        p = np.array(p[:biplot_components])

        # save to dataframe
        var_explained = pd.DataFrame(p.diagonal(),
                                     index=a_hat_mat.columns,
                                     columns=['var_explained'])
    else:
        var_explained = lambda_coeffs.div(lambda_coeffs.sum(axis=1),
                                          axis=0)
    return var_explained


def lambda_sort(feature_loadings,
                state_loadings, lambdas):
    '''
    Reorder and rename components based on the magnitude
    of their corresponding singular values

    Parameters
    ----------
    feature_loadings: dictionary, required
        Feature loadings
        keys = modality
        values = dataframe of loadings for each component

    state_loadings: dictionary, required
        Temporal loadings
        keys = modality
        values = dataframe of loadings for each component

    lambda_coeffs: dataframe, required
        Singular values
        index = modality
        columns = component

    Returns
    ----------
    Loadings with reordered components
    '''

    # save component labels
    components = lambdas.columns
    n_components = len(components)
    n_mods = len(lambdas.index)
    sorted_lambdas = pd.DataFrame(np.zeros((n_mods, n_components)),
                                  index=lambdas.index,
                                  columns=components)
    
    for mod in lambdas.index:

        # get mod-specific lambdas
        lambda_mod = lambdas.loc[mod].copy()
        order = np.argsort(-lambda_mod)
        # save sorted lambdas
        sorted_lambdas.loc[mod] = lambda_mod[order].values
        col_order = [components[i] for i in order]
        # sort feature loadings
        b_hat = feature_loadings[mod]
        b_hat = b_hat[col_order]
        b_hat.columns = components
        feature_loadings[mod] = b_hat
        # sort state loadings
        phi_hat = state_loadings[mod]
        phi_hat = phi_hat[col_order]
        phi_hat.columns = components
        state_loadings[mod] = phi_hat

    return feature_loadings, state_loadings, sorted_lambdas


def reformat_loadings(original_loadings,
                      table_mods, n_components,
                      features=False):
    '''
    Reformat the loadings to be more user-friendly.

    Parameters
    ----------
    original_loadings: dictionary, required
        Loadings for each modality
        keys = component number
        values = dictionary of modality-specific loadings
            keys = modality
            values = np.ndarray

    table_mods: dictionary, required
        Tables for each modality
        keys = modality
        values = DataFrame
            rows = features
            columns = samples
    
    n_components: int, required
        The underlying rank of the data and number of
        output dimentions.

    features: bool, optional
        Whether loadings are from features. When True,
        feature IDs are used as index.

    Returns
    ----------
    loadings_reformat: dictionary
        Re-formatted loadings
        keys = modality
        values = DataFrame
            rows = features
            columns = component number
    '''

    # reformat loadings
    loadings_reformat = {}

    for mod in table_mods.keys():

        if features:
            # get feature IDs
            first_ind = list(table_mods[mod].keys())[0]
            feature_ids = table_mods[mod][first_ind].index
            index = feature_ids
        else:
            index = None

        # get dimensions
        component_names = list(original_loadings.keys())
        first_component = component_names[0]
        n_rows = original_loadings[first_component][mod].shape[0]
        # initialize dictionary to store loadings per modality
        mod_loadings = pd.DataFrame(np.zeros((n_rows, n_components)),
                                    index=index,
                                    columns=component_names)
        # iterate over the components
        for i in range(n_components):
            component = component_names[i]
            mod_loadings[component] = original_loadings[component][mod]
        # save to dictionary
        loadings_reformat[mod] = (mod_loadings)

    return loadings_reformat


def summation_check(mod_keys,
                    feature_loadings,
                    individual_loadings,
                    state_loadings,
                    lambda_coeff,
                    prop_explained):
    '''
    Check that the summation of the loadings is nonnegative
    and revise the signs if necessary

    Parameters
    ----------
    mod_keys: list, required
        List of modality keys

    feature_loadings: dictionary, required
        Feature loadings
        keys = modality
        values = dataframe of loadings for each component

    individual_loadings: dataframe, required
        Individual loadings

    state_loadings: dictionary, required
        Temporal loadings

    lambda_coeff: dataframe, required
        Singular values
        rows = modality
        columns = component

    prop_explained: dataframe, required
        Proportion of variance explained by each
        component

    Returns
    ----------
    Updated loadings and singular values
    '''

    # initialize variables
    original_a_hat = copy.deepcopy(individual_loadings)
    individual_loadings = {}
    # determine sorting based on proportion of var explained
    new_order = np.argsort(-prop_explained.values.flatten())
    prop_explained = prop_explained.iloc[new_order]

    for modality in mod_keys:
        # revise the signs of eigenvalues
        lambda_ = np.array(lambda_coeff.loc[modality])
        lambda_ = np.where(lambda_ < 0, -lambda_, lambda_)
        a_hat = pd.DataFrame(np.where(lambda_[:, np.newaxis].T < 0,
                                      -original_a_hat,
                                      original_a_hat),
                             original_a_hat.index,
                             original_a_hat.columns)
        # get key modality-specific variables
        b_hat = feature_loadings[modality]
        phi_hat = state_loadings[modality]
        # revise the signs of feature loadings
        sgn_feature_loadings = np.sign(b_hat.sum(axis=0))
        b_hat *= sgn_feature_loadings
        a_hat = a_hat*sgn_feature_loadings
        # revise the signs of state loadings
        sgn_state_loadings = np.sign(phi_hat.sum(axis=0))
        phi_hat *= sgn_state_loadings
        # get original col names
        col_names = phi_hat.columns
        # reorder loadings
        feature_loadings[modality] = b_hat.iloc[:, new_order]
        individual_loadings[modality] = a_hat.iloc[:, new_order]
        state_loadings[modality] = phi_hat.iloc[:, new_order]
        lambda_coeff.loc[modality] = lambda_[new_order]
        # make sure col names are updated too
        feature_loadings[modality].columns = col_names
        individual_loadings[modality].columns = col_names
        state_loadings[modality].columns = col_names
        lambda_coeff.columns = col_names

    # rename columns in case order of components changed
    prop_explained.index = col_names

    return (feature_loadings, individual_loadings,
            state_loadings, lambda_coeff, prop_explained)


def feature_covariance(table_mods, b_hats, lambdas):
    '''
    Calculate the temporal feature covariance matrix

    Parameters
    ----------
    table_mods: dictionary, required
        Tables for each modality
        keys = modality
        values = DataFrame
            rows = features
            columns = samples

    b_hats: dictionary, required
        Feature loadings
        keys = modality
        values = loadings

    lambdas: dictionary, required
        Singular values
        keys = modality
        values = loadings

    Returns
    ----------
    feature_cov_mat: matrix
        rows, columns = features from all modalities
    '''

    feature_cov_vec = []
    all_feature_ids = []

    for modality in table_mods.keys():

        # get feature IDs for each modality
        table_mod = table_mods[modality]
        first_ind = list(table_mod.keys())[0]
        feature_ids = list(table_mod[first_ind].index)
        all_feature_ids.extend(feature_ids)

        # concat feature loadings from each modality
        b_hat = b_hats[modality]
        lambda_coeff = lambdas[modality]
        W_mod = lambda_coeff * b_hat
        feature_cov_vec.extend(W_mod)

    # calculate covariance matrix
    feature_cov_mat = np.outer(feature_cov_vec, feature_cov_vec)
    feature_cov_mat = pd.DataFrame(feature_cov_mat,
                                   index=all_feature_ids,
                                   columns=all_feature_ids)
    # normalize so that values are between -1 and 1
    max_value = np.max(np.abs(feature_cov_mat.values))
    feature_cov_mat = feature_cov_mat / max_value

    return feature_cov_mat


def update_lambda(individual_id_tables, ti,
                  a_hat, phi_hat, b_hat):
    '''
    Updates the singular values using the loadings
    from the most recent iteration

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of tables constructed. Note that at this point
        the tables have been subset to only include the time points
        within the previously defined interval.
        keys = individual_ids
        values = DataFrame, required
            rows = features
            columns = samples

    ti: list of int, required
        Time points within predefined interval for
        each individual

    a_hat: np.narray, required
        Subject loadings from the previous iteration

    phi_hat: np.narray, required
        Temporal loadings from the previous iteration

    b_hat: np.narray, required
        Feature loadings from the previous iteration

    Returns
    ----------
    lambda_new: dictionary
        Updated singular values
        keys = modality
        values = loadings
    '''

    nums = []
    denoms = []

    for i, m in enumerate(individual_id_tables.values()):
        
        phi_ = phi_hat[ti[i]]
        num = a_hat[i]*(b_hat.dot(m.values).dot(phi_))
        nums.append(num)
        denom = (a_hat[i]*phi_) ** 2
        denom = np.sum(denom)
        denoms.append(denom)

    lambda_new = np.sum(nums) / np.sum(denoms)

    return lambda_new


def update_a_mod(individual_id_tables,
                 n_individuals, n_features,
                 b_mod, phi_mod,
                 lambda_mod, ti):
    '''
    Update the tabular loadings (subjects and features) loadings

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of tables constructed. Note that at this point
        the tables have been subset to only include the time points
        within the previously defined interval.
        keys = individual_ids
        values = DataFrame, required
            rows = features
            columns = samples

    n_individuals: int, required
        Number of unique individuals/samples in modality

    n_features: int, required
        Number of unique features in modality

    b_mod: np.narray, required
        Feature loadings from a specific modality

    phi_mod: np.narray, required
        Temporal loadings from a specific modality

    lambda_mod: float, required
        Singular value from a specific modality

    ti: list of int, required
        Time points within predefined interval for
        each individual

    Returns
    ----------
    a_num: dictionary
        Modality-specific numerator for a_hat
        keys = individual_id
        values = float

    a_denom: dictionary
        Modality-specific denominator for a_hat

    b_num: np.narray
        Modality-specific numerator for b_hat
        Dimension = n_features x n_individuals

    common_denom: dictionary
        Modality-specific common denominator for
        a_hat and b_hat
        keys = individual_id
        values = float
    '''

    # initialize intermediate outputs
    a_num = {}
    a_denom = {}
    b_num = np.zeros((n_features, n_individuals))
    common_denom = {}

    for i, (individual_id, m) in enumerate(individual_id_tables.items()):

        # keep timepoints within interval
        phi_ = phi_mod[ti[i]]
        # save item needed for both a_hat and b_hat
        common_denom[individual_id] = np.sum(phi_ ** 2)
        # save item needed later for b_hat
        b_num[:, i] = (m.values).dot(phi_)  # vector per individual
        # a_hat specific operations
        a_num_mod = lambda_mod*b_mod.dot(m.values).dot(phi_)
        a_num[individual_id] = a_num_mod
        a_denom[individual_id] = (lambda_mod ** 2)*common_denom[individual_id]

    return a_num, a_denom, b_num, common_denom


def initialize_tabular(individual_id_tables,
                       n_individuals,
                       n_components=3):
    """
    Initialize subject and feature loadings

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of tables constructed.
        (see build_sparse class)
        keys = individual_ids
        values = DataFrame, required
            rows = features
            columns = samples

    n_individuals: int, required
        Number of unique individuals/samples

    n_components: int, optional : Default is 3
        The underlying rank of the data and number of
        output dimentions.

    Returns
    ----------
    b_hat: np.narray
        Initialized feature loadings

    a_hat: np.narray
        Initialized subject loadings
    """

    # initialize feature loadings
    data_unfold = np.hstack([m for m in individual_id_tables.values()])
    u, e, v = svds(data_unfold, k=n_components, which='LM')
    u, e, v = svd_sort(u, np.diag(e), v)
    b_hat = u[:, 0]
    # initialize subject loadings
    consistent_sign = np.sign(np.sum(b_hat))
    a_hat = (np.ones(n_individuals) / np.sqrt(n_individuals)) * consistent_sign

    return b_hat, a_hat


def decomposition_iter(table_mods, individual_id_lst,
                       times, Kmats, Kmat_outputs,
                       maxiter, epsilon,
                       smooth, n_components):
    '''
    Iterate over the available modalities

    Parameters
    ----------
    table_mods: dictionary, required
        Tables for each modality. Times are
        normalized and only points within defined
        interval are kept.
        keys = modality
        values = DataFrame
            rows = features
            columns = samples

    individual_id_lst: list, required
        List of unique individual IDs

    times: dictionary, required
        Updated time points for each modality
        keys = modality
        values = list of numpy.ndarray
            list[0] = time points within interval
            list[1] = individual indexes

    Kmats: dictionary, required
        Kernel matrix for each modality
        keys = modality
        values = numpy.ndarray
            rows, columns = time points

    Kmat_outputs: dictionary, required
        Bernoulli kernel matrix for each modality
        keys = modality
        values = numpy.ndarray
            rows = resolution
            columns = time points

    maxiter: int, optional : Default is 20
        Maximum number of iteration in for rank-1 calculation

    epsilon: float, optional : Default is 0.0001
        Convergence criteria for difference between iterations
        for each rank-1 calculation.

    smooth: float, optional : Default is 1e-6
        Smoothing parameter for the kernel matrix

    n_components: int, optional : Default is 3
        The underlying rank of the data and number of
        output dimentions.

    Returns
    ----------
    Rank-1 loadings
    a_hat: np.narray
        Subject loadings, shared across modalities

    b_hats: dictionary
        Feature loadings
        keys = modality
        values = loadings

    phi_hats: dictionary
        Temporal loadings

    lambdas: dictionary
        Singular values
    '''

    a_hats = {}
    b_hats = {}
    phi_hats = {}
    lambdas = {}
    common_denom = {}
    b_num = {}
    
    # iterate until convergence
    t = 0
    dif = 1
    while t <= maxiter and dif > epsilon:   

        # variables to save intermediate outputs
        a_num = {}
        a_denom = {}
        b_hat_difs = {}

        for modality in table_mods.keys():
            # get key modality-specific variables
            table_mod = table_mods[modality]
            ti, ind_vec = times[modality]
            Kmat = Kmats[modality]
            Kmat_output = Kmat_outputs[modality]
            n_individuals = len(table_mod)
            first_ind = list(table_mod.keys())[0]
            n_features = table_mod[first_ind].shape[0]

            if t == 0:
                # initialize feature and subject loadings
                b_hat, a_hat = initialize_tabular(table_mod,
                                                  n_individuals=n_individuals,
                                                  n_components=n_components)
                b_hats[modality] = b_hat
                a_hats[modality] = a_hat
            if t > 0:
                # update feature loadings
                b_temp = b_num[modality]
                common_denom_flat = list(common_denom[modality].values())
                common_denom_flat = np.array(common_denom_flat)
                b_new = np.dot(b_temp, a_hat) / np.dot(common_denom_flat,
                                                       a_hat ** 2)
                b_hat = b_new / np.sqrt(np.sum(b_new ** 2))
                b_hat_difs[modality] = np.sum((b_hats[modality] - b_hat) ** 2)
                b_hats[modality] = b_hat

            # calculate state loadings
            Ly = [a_hat[i] * b_hat.dot(m) for i, m in
                  enumerate(table_mod.values())]
            phi_hat = freg_rkhs(Ly, a_hat, ind_vec, Kmat, Kmat_output,
                                smooth=smooth)
            phi_hat = (phi_hat / np.sqrt(np.sum(phi_hat ** 2)))
            phi_hats[modality] = phi_hat
            # calculate lambda
            lambda_mod = update_lambda(table_mod, ti, a_hat, phi_hat, b_hat)
            lambdas[modality] = lambda_mod
            # begin updating subject and feature loadings
            (a_mod_num, a_mod_denom,
             b_mod_num,
             common_mod_denom) = update_a_mod(table_mod, n_individuals,
                                              n_features, b_hat, phi_hat,
                                              lambda_mod, ti)
            # save intermediate b-hat variables
            b_num[modality] = b_mod_num
            common_denom[modality] = common_mod_denom
            # add subject loading variables
            a_num = {**a_num, **{key: a_mod_num[key] + a_num.get(key, 0)
                                 for key in a_mod_num}}
            a_denom = {**a_denom,
                       **{key: a_mod_denom[key] + a_denom.get(key, 0)
                          for key in a_mod_denom}}
        # update subject loadings
        a_tilde = np.array(
            [a_num[id] / a_denom[id] for id in individual_id_lst]
        )
        a_scaling = np.sqrt(np.sum(a_tilde ** 2))
        a_new = np.array(
            [a_tilde[i] / a_scaling for i in range(len(a_tilde))]
        )
        a_hat_dif = np.sum((a_hat - a_new) ** 2)
        a_hat = a_new
        # check for convergence (maybe take mean of b_hat_difs?)
        dif = max([a_hat_dif]+list(b_hat_difs.values()))
        t += 1
    print("Reached convergence in {} iterations".format(t))

    return a_hat, b_hats, phi_hats, lambdas


def format_time(individual_id_tables,
                individual_id_state_orders,
                n_individuals, resolution,
                input_time_range, interval=None):
    '''
    Normalize time points to be in the same format and keep
    only the defined interval (if defined)

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of tables constructed.
        (see build_sparse class)
        keys = individual_ids
        values = DataFrame, required
            rows = features
            columns = samples

    individual_id_state_orders : dict
        Dictionary of time points for each individual

    n_individuals: int, required
        Number of unique individuals/samples

    resolution: int, optional : Default is 101
        Number of time points to evaluate the value
        of the temporal loading function.

    input_time_range: tuple, required
        Start and end time points for each individual

    interval : tuple, optional
        Start and end time points to keep

    Returns
    -------
    interval: tuple
        Normalized interval

    tables_update: dictionary
        Tables with updated time points. If interval is
        defined, only time points within interval are kept

    ti: list of numpy.ndarray
        List of time points within defined interval
        per subject

    ind_vec: numpy.ndarray
        Subject indexes for each time point

    tm: numpy.ndarray
        Concatenated normalized time points for all
        subjects
    '''

    # make copy of tables to update
    tables_update = copy.deepcopy(individual_id_tables)
    orders_update = copy.deepcopy(individual_id_state_orders)

    # normalize time points
    for individual_id in orders_update.keys():
        update_num = (orders_update[individual_id] - input_time_range[0])
        update_den = (input_time_range[1] - input_time_range[0])
        orders_update[individual_id] =  update_num / update_den
    # ensure interval is in the same format
    interval_num = (interval - input_time_range[0])
    interval_den = (input_time_range[1] - input_time_range[0])
    interval = tuple(interval_num / interval_den)

    # initialize variables to store time points (tps)
    Lt = []  # all normalized tps
    ind_vec = []  # individual indexes for each tp
    ti = [[] for i in range(n_individuals)]  # tps within interval per subject

    # populate variables above
    for i, (id_, time_range_i) in enumerate(orders_update.items()):
        # save all normalized time points
        Lt.append(time_range_i)
        ind_vec.extend([i] * len(Lt[-1]))
        # define time points within interval
        mask = (time_range_i >= interval[0]) & (time_range_i <= interval[1])
        temp = time_range_i[mask]
        temp = [(resolution-1)*(tp - interval[0])/(interval[1] - interval[0])
                for tp in temp]
        ti[i] = np.array(list(map(int, temp)))
        # update tables and orders
        tables_update[id_] = tables_update[id_].T[mask].T

    # convert variables to numpy arrays
    ind_vec = np.array(ind_vec)
    tm = np.concatenate(Lt)

    return interval, tables_update, ti, ind_vec, tm


def formatting_iter(individual_id_tables,
                    individual_id_state_orders,
                    mod_id_ind, input_time_range,
                    interval, resolution):
    '''
    Format the input data for downstream tasks and
    calculate tne kernel matrix.

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of 1 to n tables constructed,
        (see build_sparse class), where n is the
        number of modalities.
        keys = individual_ids
        values = list of DataFrame, required
            For each DataFrame (modality):
            rows = features
            columns = samples

    individual_id_state_orders: dictionary, required
        Dictionary of 1 to n lists of time points (one
        per modality) for each sample.
        keys = individual_ids
        values = list of numpy.ndarray
            Each numpy.ndarray contains the time points
            of the corresponding modality
            Note: array of dtype=object to allow for
            different number of time points per modality

    mod_id_ind: dictionary, required
        Dictionary of individual IDs for each modality
        keys = modality
        values = list of tuples
            Each tuple contains the individual id and
            the dataframe index in individual_id_tables

    input_time_range: tuple, required
        Start and end time points for each individual

    interval : tuple, optional
        Start and end time points to keep

    resolution: int, optional : Default is 101
        Number of time points to evaluate the value
        of the temporal loading function.

    Returns
    ----------
    table_mods: dictionary
        Updated tables for each modality. Times are
        normalized and only points within the interval
        are kept.
        keys = modality
        values = DataFrame
            rows = features
            columns = samples

    times: dictionary
        Updated time points for each modality
        keys = modality
        values = list of numpy.ndarray
            list[0] = time points within interval
            list[1] = individual indexes

    Kmats: dictionary
        Kernel matrix for each modality
        keys = modality
        values = numpy.ndarray
            rows, columns = time points

    Kmat_outputs: dictionary
        Bernoulli kernel matrix for each modality
        keys = modality
        values = numpy.ndarray
            rows = resolution
            columns = time points

    norm_interval: tuple
        Normalized interval
    '''

    # initialize dictionary to store outputs per modality
    table_mods = {}
    times = {}
    Kmats = {}
    Kmat_outputs = {}

    # iterate through each modality
    for modality in mod_id_ind.keys():

        # get the individual IDs
        ind_tuple_lst = mod_id_ind[modality]
        # keep modality-specific time points
        orders_mod = {ind[0]: individual_id_state_orders[ind[0]][ind[1]]
                      for ind in ind_tuple_lst}
        # keep modality-specific tables
        table_mod = {ind[0]: individual_id_tables[ind[0]][ind[1]]
                     for ind in ind_tuple_lst}
        n_individuals = len(table_mod)
        # format time points and keep points in the interval
        (norm_interval, table_mod,
         ti, ind_vec, tm) = format_time(table_mod, orders_mod,
                                        n_individuals, resolution,
                                        input_time_range, interval)
        # save key outputs
        table_mods[modality] = table_mod
        times[modality] = [ti, ind_vec]
        # construct the kernel matrix
        Kmats[modality] = bernoulli_kernel(tm, tm)
        Kmat_outputs[modality] = bernoulli_kernel(np.linspace(norm_interval[0],
                                                              norm_interval[1],
                                                              num=resolution),
                                                  tm)
        
    return table_mods, times, Kmats, Kmat_outputs, norm_interval


def joint_ctf_helper(individual_id_tables,
                     individual_id_state_orders,
                     mod_id_ind, interval,
                     resolution, maxiter,
                     epsilon, smooth,
                     n_components):
    '''
    Joint decomposition of two or more tensors

    Parameters
    ----------
    individual_id_tables: dictionary, required
        Dictionary of 1 to n tables constructed,
        (see build_sparse class), where n is the 
        number of modalities.
        keys = individual_ids
        values = list of DataFrame, required
            For each DataFrame (modality):
            rows = features
            columns = samples

    individual_id_state_orders: dictionary, required
        Dictionary of 1 to n lists of time points (one
        per modality) for each sample.
        keys = individual_ids
        values = list of numpy.ndarray
            Each numpy.ndarray contains the time points
            of the corresponding modality
            Note: array of dtype=object to allow for
            different number of time points per modality

    mod_id_ind: dictionary, required
        Dictionary of individual IDs for each modality
        keys = modality
        values = list of tuples
            Each tuple contains the individual id and
            the dataframe index in individual_id_tables

    interval : tuple, optional
        Start and end time points to keep

    resolution: int, optional : Default is 101
        Number of time points for the temporal
        loading function.

    maxiter: int, optional : Default is 20
        Maximum number of iteration in for rank-1 calculation.

    epsilon: float, optional : Default is 0.0001
        Convergence criteria for difference between iterations
        for each rank-1 calculation.

    smooth: float, optional : Default is 1e-8
        Smoothing parameter for RKHS norm. Larger means
        smoother temporal loading functions.

    n_components: int, optional : Default is 3
        The underlying rank of the data and number of
        output dimentions.

    Returns
    ----------
    individual_loadings: pd.DataFrame
        Subject loadings
        rows = individual IDs
        columns = component number

    feature_loadings: dictionary
        Feature loadings
        keys = component number
        values = dictionary of modality-specific loadings

    state_loadings: dictionary
        Temporal loadings

    lambda_coeff: dictionary
        Singular values

    time_return: np.ndarray
        Time points for the temporal loading function
    '''

    # make copy of tables to update
    tables_update = copy.deepcopy(individual_id_tables)
    orders_update = copy.deepcopy(individual_id_state_orders)
    # get all individual IDs
    individual_id_lst = list(orders_update.keys())
    n_individuals_all = len(individual_id_lst)
    # get all time points across all modalities
    timestamps_all = []
    for arr in list(orders_update.values()):
        timestamps_all.extend(arr)
    timestamps_all = np.concatenate(timestamps_all)
    timestamps_all = np.unique(timestamps_all)
    # set the interval if none is given
    if interval is None:
        interval = (timestamps_all[0], timestamps_all[-1])
    # define the input time range, equal to interval if none defined
    input_time_range = (timestamps_all[0], timestamps_all[-1])
    
    # format time points and keep points in defined interval
    (table_mods, times,
     Kmats, Kmat_outputs,
     norm_interval) = formatting_iter(tables_update,
                                      orders_update,
                                      mod_id_ind,
                                      input_time_range,
                                      interval, resolution)

    # init dataframes to fill
    n_component_col_names = ['component_' + str(i+1)
                             for i in range(n_components)]
    individual_loadings = pd.DataFrame(np.zeros((n_individuals_all,
                                                 n_components)),
                                       index=tables_update.keys(),
                                       columns=n_component_col_names)
    lambda_coeff = pd.DataFrame(np.zeros((len(table_mods), n_components)),
                                index=table_mods.keys(),
                                columns=n_component_col_names)
    prop_explained = pd.DataFrame(np.zeros((len(table_mods), n_components)),
                                  index=table_mods.keys(),
                                  columns=n_component_col_names)
    var_explained = pd.DataFrame(np.zeros((len(table_mods), n_components)),
                                 index=table_mods.keys(),
                                 columns=n_component_col_names)
    feature_cov_mats = {}

    # for the dicts below
    # key: component number, value: dictionary of modality-specific loadings
    feature_loadings = {}
    state_loadings = {}

    # perform decomposition
    for r in range(n_components):
        comp_name = 'component_' + str(r+1)
        print('Calculate components for {}'.format(comp_name))
        (a_hat, b_hats,
         phi_hats, lambdas) = decomposition_iter(table_mods, individual_id_lst,
                                                 times, Kmats, Kmat_outputs,
                                                 maxiter, epsilon,
                                                 smooth, n_components)
        # save rank-1 components
        individual_loadings.iloc[:, r] = a_hat
        lambda_coeff.iloc[:, r] = list(lambdas.values())
        feature_loadings[comp_name] = b_hats
        state_loadings[comp_name] = phi_hats
        # calculate residuals and update tables
        tables_update, rsquared = update_residuals(table_mods, a_hat, b_hats,
                                                   phi_hats, times, lambdas)
        table_mods = tables_update
        prop_explained.iloc[:, r] = list(rsquared.values())

        feature_cov_mat = feature_covariance(table_mods, b_hats, lambdas)
        feature_cov_mats[comp_name] = feature_cov_mat
        # note: could subset loadings/feature list first to calculate cov_mat

    # reformat feature and state loadings
    feature_loadings = reformat_loadings(feature_loadings,
                                         table_mods, n_components,
                                         features=True)
    state_loadings = reformat_loadings(state_loadings,
                                       table_mods, n_components)
    # make sure lambdas are sorted in descending order per modality
    # print("pre-lambda sorting lambdas\n", lambda_coeff)
    (feature_loadings,
     state_loadings,
     lambda_coeff) = lambda_sort(feature_loadings,
                                 state_loadings, lambda_coeff)
    # calculate prop of variance explained
    var_explained = get_prop_var(individual_loadings, feature_loadings,
                                 lambda_coeff, n_components)
    # revise the signs to make sure summation is nonnegative
    # and order based on coefficient of determination
    (feature_loadings,
     individual_loadings,
     state_loadings,
     lambda_coeff,
     var_explained) = summation_check(table_mods.keys(),
                                      feature_loadings,
                                      individual_loadings,
                                      state_loadings,
                                      lambda_coeff,
                                      var_explained)
    # return original time points
    time_return = np.linspace(norm_interval[0],
                              norm_interval[1],
                              resolution)
    time_return *= (input_time_range[1] - input_time_range[0])
    time_return += input_time_range[0]
    time_return = pd.DataFrame(time_return,
                               index=np.arange(resolution),
                               columns=['time_interval'])
    # concat time_return to state_loadings
    for key in state_loadings.keys():
        state_loadings[key] = pd.concat([state_loadings[key], time_return],
                                        axis=1)
    return (individual_loadings, feature_loadings,
            state_loadings, lambda_coeff,
            var_explained, feature_cov_mats)


def joint_ctf(tables,
              sample_metadatas,
              modality_ids,
              individual_id_column: str,
              state_column: str,
              interval: tuple = None,
              n_components: int = DEFAULT_COMP,
              replicate_handling: str = DEFAULT_TRH,
              svd_centralized: bool = DEFAULT_TEMPTED_SVDC,
              n_components_centralize: int = DEFAULT_TSCN,
              smooth: float = DEFAULT_TEMPTED_SMTH,
              resolution: int = DEFAULT_TEMPTED_RES,
              max_iterations: int = DEFAULT_TEMPTED_MAXITER,
              epsilon: float = DEFAULT_TEMPTED_EP):
    '''
    Joint decomposition of two or more tensors

    Parameters
    ----------
    tables: list of numpy.ndarray, required
        List of feature tables (1-n) from different modalities
        in biom format containing the samples over which
        metrics should be computed.
        Each modality should contain same number of samples
        or individuals. Length of features might vary.

    sample_metadatas: list of DataFrame, required
        Sample metadata files in QIIME2 formatting for each
        modality. The file must contain the columns for
        individual_id_column and state_column and the rows
        matched to the table.

    modality_ids: list of str, required
        Unique identifier for each modality.

    individual_id_column: str, required
        Metadata column containing subject IDs to use for
        pairing samples. WARNING: if replicates exist for an
        individual ID at either state_1 to state_N, that
        subject will be mean grouped by default.

    state_column: str, required
        Metadata column containing state (e.g.,Time,
        BodySite) across which samples are paired. At least
        one is required but up to four are allowed by other
        state inputs.

    interval: tuple, optional : Default is None
        Start and end time points to keep.

    n_components: int, optional : Default is 3
        The underlying rank of the data and number of
        output dimentions.

    replicate_handling: function, optional : Default is "sum"
        Choose how replicate samples are handled. If replicates are
        detected, "error" causes method to fail; "drop" will discard
        all replicated samples; "random" chooses one representative at
        random from among replicates.

    svd_centralized: bool, optional : Default is True
        Removes the mean structure of the temporal tensor.

    n_components_centralize: int
        Rank of approximation for average matrix in svd-centralize.

    smooth: float, optional : Default is 1e-8
        Smoothing parameter for RKHS norm. Larger means
        smoother temporal loading functions.

    resolution: int, optional : Default is 101
        Number of time points to evaluate the value
        of the temporal loading function.

    max_iterations: int, optional : Default is 20
        Maximum number of iteration in for rank-1 calculation.

    epsilon: float, optional : Default is 0.0001
        Convergence criteria for difference between iterations
        for each rank-1 calculation.

    Returns
    -------
    OrdinationResults - TODO
        Compositional biplot of subjects as points and
        features as arrows. Where the variation between
        subject groupings is explained by the log-ratio
        between opposing arrows.

    DataFrame
        Each components temporal loadings across the
        input resolution included as a column called
        'time_interval'.

    DistanceMatrix - TODO
        A subject-subject distance matrix generated
        from the euclidean distance of the
        subject ordinations and itself.

    DataFrame - TODO
        The loadings from the SVD centralize
        function, used for projecting new data.
        Warning: If SVD-centering is not used
        then the function will add all ones as the
        output to avoid variable outputs.

    Raises
    ------
    ValueError
        if features don't match between tables
        across the values of the dictionary
    ValueError
        if id_ not in mapping
    ValueError
        if any state_column not in mapping
    ValueError
        Table is not 2-dimensional
    ValueError
        Table contains negative values
    ValueError
        Table contains np.inf or -np.inf
    ValueError
        Table contains np.nan or missing.
    Warning
        If a conditional-sample pair
        has multiple IDs associated
        with it. In this case the
        default method is to mean them.
    ValueError
        `ValueError: n_components must be at least 2`.
    ValueError
        `ValueError: Data-table contains
         either np.inf or -np.inf`.
    ValueError
        `ValueError: The n_components must be less
         than the minimum shape of the input tensor`.
    '''
    
    # note: we assume each modality has a dif table and associated
    # metadata. We also assume filtering conditions are the same
    tensors = {}
    for table, metadata, mod_id in zip(tables,
                                       sample_metadatas,
                                       modality_ids):
        
        # check the table for validity and then filter
        process_results = ctf_table_processing(table,
                                               metadata,
                                               individual_id_column,
                                               [state_column],
                                               min_sample_count=0,
                                               min_feature_count=0,
                                               min_feature_frequency=0,
                                               feature_metadata=None)
        table = process_results[0]
        metadata = process_results[1]

        # build the sparse tensor format
        tensor = build_sparse()
        tensor.construct(table,
                         metadata,
                         individual_id_column,
                         state_column,
                         transformation=lambda x: x,
                         pseudo_count=0,
                         branch_lengths=None,
                         replicate_handling=replicate_handling,
                         svd_centralized=svd_centralized,
                         n_components_centralize=n_components_centralize)
        tensors[mod_id] = tensor

    # save all tensors to a class
    n_tensors = concat_tensors().concat(tensors)
    # run joint-CTF
    joint_ctf_res = joint_ctf_helper(n_tensors.individual_id_tables,
                                     n_tensors.individual_id_state_orders,
                                     n_tensors.mod_id_ind,
                                     interval=interval,
                                     resolution=resolution,
                                     maxiter=max_iterations,
                                     epsilon=epsilon,
                                     smooth=smooth,
                                     n_components=n_components)

    (individual_loadings, feature_loadings,
     state_loadings, eigenvalues,
     prop_explained, feature_cov_mats) = joint_ctf_res

    return (individual_loadings, feature_loadings,
            state_loadings, eigenvalues,
            prop_explained, feature_cov_mats)
