import unittest
import os
import inspect
import pandas as pd
import numpy as np
from skbio import OrdinationResults
from pandas import read_csv
from biom import load_table
from skbio.util import get_data_path
from gemelli.testing import assert_ordinationresults_equal
from gemelli.joint_ctf import (concat_tensors, update_residuals,
                               get_prop_var, lambda_sort,
                               reformat_loadings, summation_check,
                               feature_covariance, update_lambda,
                               update_a_mod, initialize_tabular, 
                               decomposition_iter, format_time,
                               formatting_iter, joint_ctf_helper,
                               joint_ctf)
from gemelli.rpca import (rpca_table_processing)
from gemelli.preprocessing import build_sparse
from numpy.testing import assert_allclose