#%% Imports

from preamble import *
from DRN_interface import *
print("\033[32m Loaded required custom python modules - DRN_interface")

def py_pbarlike(DM_mass, brfr,sigma_v = 10**(-25.5228),propagation_parameters=np.array([0.0]),propagation_model='DIFF.BRK', prevent_extrapolation= False,data=phi_ams,E = E_ams,errors = error_ams,data_cov=ams_7y_cov,xsection_cov=True,verbose=False):
    '''
    Calculates difference of marginalized chi-squared value between cases of with and without DM.

    Arguments:
        *DM_mass*(int, float, list, or 1D array): dark matter mass in GeV
        *brfr*(list or array): branching fractions to specify the DM annihilation channel; format - [q qbar, c cbar, b bbar, t tbar, W+ W-, Z0 Z0, g g, h h]
        *sigma_v*(int, float, list or 1D array): thermally averaged annihilation cross-section in cm3s-1
        *propagation_parameters*(list or array): propagation parameters (for format and ranges see DRN.load_pp_data())
        *propagation_model*(str): "DIFF.BRK" or "INJ.BRK+vA"
        *prevent_extrapolation*(bool):  decides if DRN should be allowed to predict in parameter regions outside trained region; default-False
        *data*(1D array): antiproton flux measurements in m-2 sr-1 s-1 GeV-1 at energies E; default - 7 year AMS-02 data
        *E*(1D array): kinetic energy per nucleon values in GeV at which antiproton measurements are given; default - E_ams
        *errors*(1D array): statistical errors at corresponding kinetic energy per nucleon values; default - errors_ams
        *data_cov*(2D array): systematic errors at corresponding kinetic energy per nucleon values; default - ams_7y_cov
        *xsection_cov*(bool): decides if covariance arising from antiproton production cross-section uncertainties should be included; default - True
        *verbose*: default - False

    Returns:
        (dictionary): marginalized chi-squared differences using correlated and uncorrelated errors

    '''
    propagation_parameters = np.array(propagation_parameters)
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,E,errors,data_cov,xsection_cov,verbose)
    DRN.preprocessing_DMparams(DM_mass, brfr, sigma_v)
    phi_DM_LIS = DRN.LIS_sim()
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2 , 'correlated' : del_chi2_corr}
    return result

print("\033[37m Loaded pbarlike 1.0")