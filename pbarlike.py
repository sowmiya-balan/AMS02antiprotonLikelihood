#%% Imports

from preamble import *
from DRN_interface import *
print("\033[32m Loaded required custom python modules - DRN_interface, solar_mod")

#%% C++ branching fractions dictionary --> DRN branching fractions array

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
def DRN_initialization(propagation_parameters,propagation_model='DIFF.BRK', prevent_extrapolation= False,data = phi_ams,errors=error_ams,cov_inv = ams_cov_inv):
    # propagation_parameters = np.array(propagation_parameters)
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,errors,cov_inv)
    # print('\nPropagation model: ',DRN.propagation_model)
    return DRN
    
def py_pbar_logLikes(DRN, DM_mass, brfr, sigma_v = 10**(-25.5228)):
    bf = br_fr(brfr,sigma_v)
    DRN.preprocessing_DMparams(DM_mass, bf, sigma_v)
    # print('Normalized branching fractions: ',DRN.bf)
    # print('\n Rescaled cross-section: ', DRN.sv)
    phi_DM_LIS = DRN.LIS_sim()
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2 , 'correlated' : del_chi2_corr}
    return result

def py_pbarlike(DM_mass, brfr,propagation_parameters, sigma_v = 10**(-25.5228),propagation_model='DIFF.BRK', prevent_extrapolation= False,data=phi_ams,errors = error_ams,cov_inv = ams_cov_inv):
    propagation_parameters = np.array(propagation_parameters)
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,errors,cov_inv)
    DRN.preprocessing_DMparams(DM_mass, brfr, sigma_v)
    # print('Normalized branching fractions: ',DRN.bf)
    # print('\nPropagation model: ',DRN.propagation_model)
    phi_DM_LIS = DRN.LIS_sim()
    # print('reshaped:',(phi_DM_LIS))
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2 , 'correlated' : del_chi2_corr}
    return result
    
def bbBar_grid(n,sigma_v,propagation_model='DIF.BRK', prevent_extrapolation= False,data=phi_ams,errors = error_ams,cov_inv = ams_cov_inv):
    DM_mass = np.logspace(np.log10(10),np.log10(5000),n)
    propagation_parameters = [0.0]
    brfr = np.array([1.000e-05, 1.000e-05, 9.993e-01, 1.000e-05, 1.000e-05, 1.000e-05,  1.000e-05, 1.000e-05])
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation,data,errors,cov_inv)
    DRN.preprocessing_DMparams(DM_mass, brfr, sigma_v)
    phi_DM_LIS = DRN.LIS_sim()
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    return del_chi2,del_chi2_corr
print("\033[37m     Loaded pbarlike 1.0")