#%% Imports
from tkinter import E
from pbarlike import *

#%% Inputs
m_DM = np.array([10**(5.018181818181818 - 3)]) 
bf = np.array([1.000e-05, 1.000e-05, 9.993e-01, 1.000e-05, 1.000e-05, 1.000e-05,  1.000e-05, 1.000e-05])
sv = 3e-26
pp_db_ib = [create_DIFF_BRK_parameters(), create_INJ_BRK_parameters()]
pm_options = ['DIFF.BRK','INJ.BRK+vA']
pp = 0 # 0- DIFF.BRK; 1 - INJ.BRK+vA
prevent_extrapolation = True
data_arg=phi_ams
E_arg=E_ams
errors_arg=error_ams
data_cov_arg=ams_7y_cov
xsection_cov_arg=True
verbose_arg=False


#%% Results
print('\n Calling pbarlike...')

print('\n Test Simulation:1; Propagation Model: DIFF.BRK')
results = py_pbarlike(DM_mass=m_DM, brfr=bf,sigma_v=sv, propagation_parameters=pp_db_ib[pp],propagation_model=pm_options[pp],
                      prevent_extrapolation = prevent_extrapolation,
                      data=data_arg,E=E_arg, errors = errors_arg,data_cov = data_cov_arg,xsection_cov = xsection_cov_arg,
                      verbose=verbose_arg)

print('\n del_chi2 = ', results["uncorrelated"])
print('\n del_chi_cov = ', results["correlated"])

print('\n Test Simulation:2; Propagation Model: INJ.BRK+vA')
pp = 1
results = py_pbarlike(DM_mass=m_DM, brfr=bf,sigma_v=sv, propagation_parameters=pp_db_ib[pp],propagation_model=pm_options[pp],
                      prevent_extrapolation = prevent_extrapolation,
                      data=data_arg,E=E_arg, errors = errors_arg,data_cov = data_cov_arg,xsection_cov = xsection_cov_arg,
                      verbose=verbose_arg)

print('\n del_chi2 = ', results["uncorrelated"])
print('\n del_chi_cov = ', results["correlated"])

print('\n ------------------------------------------------------------'\
      '\n pbarlike performed likelihood calculation without errors! :)'\
      '\n ------------------------------------------------------------')