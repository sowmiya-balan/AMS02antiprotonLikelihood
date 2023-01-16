#%% Imports
from DRN_interface import *
print("\033[32m Loaded required custom python module - DRN_interface")

#%% C++ branching fractions dictionary from GAMBIT --> DRN branching fractions array
def br_fr(inputs, sigma_v=1):
    # To avoid 0/0 during normalization
    if sigma_v==0.:
        sigma_v=1
    # Normalizing bf 
    factorized_bf = {key: inputs[key] / sigma_v
                        for key in inputs.keys()}

    # DRBF - Dark Ray net Branching Fraction dictionary
    # Positions of channels in DRN input-bf-array
    DRBF = {"q qbar":0, "c cbar":1, "b bbar":2, "t tbar":3, "W+ W-":4, "Z0 Z0":5, "g g":6, "h h":7}
    bf = np.zeros((1,8))

    # Template for the DRN input-bf-array entry   -    Gambit annihilation channel : DRN annihilation channel(location in DRN input-bf-array )
    keys_to_location = {
                    "u_1 ubar_1":DRBF['q qbar'],"u_2 ubar_2":DRBF['c cbar'],"u_3 ubar_3":DRBF['t tbar'],
                    "ubar_1 u_1":DRBF['q qbar'],"ubar_2 u_2":DRBF['c cbar'],"ubar_3 u_3":DRBF['t tbar'],
                    "d_1 dbar_1":DRBF['q qbar'],"d_2 dbar_2":DRBF['q qbar'],"d_3 dbar_3":DRBF['b bbar'],
                    "dbar_1 d_1":DRBF['q qbar'],"dbar_2 d_2":DRBF['q qbar'],"dbar_3 d_3":DRBF['b bbar'],
                    "W+ W-":DRBF['W+ W-'], "W- W+":DRBF['W+ W-'],
                    "Z0 Z0":DRBF['Z0 Z0'], "g g":DRBF['g g'], 
                    "h0_1 h0_1":DRBF['h h']
                        }
    
    # For all possible Gambit annihilation channels, enter corresponding factorized_bf(if present, otherwise 0) in DRN input-bf-array in appropriate location
    for i in keys_to_location.keys() :
        bf[0,keys_to_location[i]] += factorized_bf.get(i,0)
    return bf

#%% Initializing DRN class, LIS simulation, solar modulation and delta-chi2 calculation

# Delta-chi2 calculation for Gambit; propagation parameters always input as a list in the yaml file; default setting in Gambit yaml file ams.yaml: faulty prop. params that force DRN to perform marginalization over multinest sample 
def DRN_initialization(propagation_parameters,propagation_model='DIFF.BRK', prevent_extrapolation= False,data = phi_ams,E=E_ams,errors=error_ams,data_cov=ams_7y_cov,xsection_cov=True):
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,E,errors,data_cov,xsection_cov)
    return DRN
    
def py_pbar_logLikes(DRN, DM_mass, brfr, sigma_v = 10**(-25.5228)):
    if type(brfr)==list and len(br_fr)!=8:
        bf = br_fr(brfr,sigma_v)
    if brfr.ndim >= 1 and brfr.shape[-1]!=8:
        bf = br_fr(brfr,sigma_v)
    DRN.preprocessing_DMparams(DM_mass, bf, sigma_v)
    phi_DM_LIS = DRN.LIS_sim()
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2[0] , 'correlated' : del_chi2_corr[0]}
    return 

#%% Delta-chi2 calculation for call from within python
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
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,E,errors,data_cov,xsection_cov,verbose)
    DRN.preprocessing_DMparams(DM_mass, brfr, sigma_v)
    phi_DM_LIS = DRN.LIS_sim()
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2 , 'correlated' : del_chi2_corr}
    return result

print("\033[37m Loaded pbarlike 1.0")
import banner