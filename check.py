#%% Imports
from preamble import *
from DRN_interface import *
from pbarlike import *

#%% Inputs
m_DM = np.array([100.,100.,100.]) #np.array([3000.,300.]) #
bf = np.array([0.00000000e+00, 1.02133998e-06, 2.40315289e-03, 0.00000000e+00,  7.41973454e-01, 2.55334994e-01, 0.00000000e+00, 0.00000000e+00])
sv = [1.70104582586906e-31,1.70104582586906e-31]
pp_run1 = np.array([1.8, 1.79, 2.405, 2.357, 7.92e+03, 0.37, 2.05e+28, 0.419, 8.84, 0.09, 2.60])
pp_db = np.array([2.34, 2.28, 3.63e+28, -0.66, 0.52, -0.15, 3.83e+3, 0.4, 2.05e+5])
pm = "DIFF.BRK"
prevent_extrapolation = True

#%% Results
print('Calling pbarlike...')
results = py_pbarlike(m_DM, bf,pp_run1,sv,propagation_model=pm,prevent_extrapolation = prevent_extrapolation)
print('del_chi2 = ', results["uncorrelated"])
print('del_chi_cov = ', results["correlated"])
print('nla',bbBar_grid(2,sv))
