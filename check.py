#%% Imports
from preamble import *
from DRN_interface import *
from pbarlike import *

#%% Inputs
m_DM = np.array([100.])
bf = np.array([0,0,0,0,1,0,0,0])
sv = 3e-26
pp_run1 = np.array([1.8, 1.79, 2.405, 2.357, 7.92e+03, 0.37, 2.05e+28, 0.419, 8.84, 0.09, 2.60])
pp_db = np.array([2.34, 2.28, 3.63e+28, -0.66, 0.52, -0.15, 3.83e+3, 0.4, 2.05e+5, 0.21])
pm = "run1"
prevent_extrapolation = False

#%% Results
print('Calling pbarlike...')
results = py_pbarlike(m_DM, bf,pp_db,propagation_model=pm)
print('del_chi2 = ', results["uncorrelated"])
print('del_chi_cov = ', results["correlated"])
