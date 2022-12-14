#%% Imports
from preamble import *
from DRN_interface import *
from pbarlike import *

#%% Inputs
m_DM = np.array([10**(5.018181818181818 - 3)]) 
bf = np.array([1.000e-05, 1.000e-05, 9.993e-01, 1.000e-05, 1.000e-05, 1.000e-05,  1.000e-05, 1.000e-05])
sv = 5.3366992312063e-27
pp_marg = np.array([0.0])
pp_run1 = np.array([1.8, 1.79, 2.405, 2.357, 7.92e+03, 0.37, 2.05e+28, 0.419, 8.84, 0.09, 2.60])
pp_db = np.array([2.34, 2.28, 3.63e+28, -0.66, 0.52, -0.15, 3.83e+3, 0.4, 2.05e+5])
pm_options = ['run1','DIFF.BRK','INJ.BRK+vA']
pm = pm_options[1]
prevent_extrapolation = True
data_arg=phi_ams
errors_arg=error_ams
data_cov_arg=ams_7y_cov
xsection_cov_arg=True

#%% Results
print('Calling pbarlike...')
results = py_pbarlike(DM_mass=m_DM, brfr=bf,sigma_v=sv,propagation_parameters=pp_marg,propagation_model=pm,prevent_extrapolation = prevent_extrapolation,data=data_arg,errors = errors_arg,data_cov = data_cov_arg,xsection_cov = xsection_cov_arg)
print('del_chi2 = ', results["uncorrelated"])
print('del_chi_cov = ', results["correlated"])
