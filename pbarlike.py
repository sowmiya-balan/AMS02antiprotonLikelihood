#%% Imports

from preamble import *
from DRN_interface import *
print("\033[32m Loaded required custom python modules - DRN_interface, solar_mod")

#%% Applying solar modulation; Adding DM and CR contributions to antiproton flux

def br_fr(inputs, sigma_v=1):
    # DRBF - Dark Ray net Branching Fraction dictionary
    if sigma_v==0.:
        sigma_v=1
    DRBF = {"q qbar":0, "c cbar":1, "b bbar":2, "t tbar":3, "W+ W-":4, "Z0 Z0":5, "g g":6, "h h":7}
    bf = np.zeros((1,8))
    factorized_bf = {key: inputs[key] / sigma_v
                        for key in inputs.keys()}
    # Template for the dictionary entry - Gambit annihilation channel : DRN annihilation channel
    keys_to_location = {
                    "u1 ubar_1":DRBF['q qbar'],"u2 ubar_2":DRBF['c cbar'],"u3 ubar_3":DRBF['t tbar'],
                    "ubar_1 u1":DRBF['q qbar'],"ubar_2 u2":DRBF['c cbar'],"ubar_3 u3":DRBF['t tbar'],
                    "d1 dbar_1":DRBF['q qbar'],"d2 dbar_2":DRBF['q qbar'],"d3 dbar_3":DRBF['b bbar'],
                    "dbar_1 d1":DRBF['q qbar'],"dbar_2 d2":DRBF['q qbar'],"dbar_3 d3":DRBF['b bbar'],
                    "W+ W-":4, "W- W+":4,
                    "Z0 Z0":5, "g g":6, 
                    "h h":7
                        }
    for i in keys_to_location.keys() :
        bf[0,keys_to_location[i]] += factorized_bf.get(i,0)
    return bf

def DRN_initialization(propagation_parameters,propagation_model='DIFF.BRK', prevent_extrapolation= False):
    propagation_parameters = np.array(propagation_parameters)
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation)
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

def py_pbarlike(DM_mass, brfr,propagation_parameters, sigma_v = 10**(-25.5228),propagation_model='DIFF.BRK', prevent_extrapolation= False):
    propagation_parameters = np.array(propagation_parameters)
    DRN = DRNet(propagation_parameters,propagation_model,prevent_extrapolation)
    # print("phi_CR_LIS",(DRN.phi_CR_LIS).shape)
    # print("phi_CR",(DRN.phi_CR).shape)
    # print("chi2_CR_uncorr",(DRN.chi2_CR_uncorr).shape)
    # print("chi2_CR_corr",(DRN.chi2_CR_corr).shape)
    DRN.preprocessing_DMparams(DM_mass, brfr, sigma_v)
    print('Normalized branching fractions: ',DRN.bf)
    print('\nPropagation model: ',DRN.propagation_model)
    phi_DM_LIS = DRN.LIS_sim()
    # print("phi_DM_LIS",phi_DM_LIS.shape)
    phi_DMCR = DRN.TOA_sim(phi_DM_LIS)
    # print("phi_DMCR",phi_DMCR.shape)
    del_chi2 = DRN.del_chi2(phi_DMCR)
    del_chi2_corr = DRN.del_chi2_corr(phi_DMCR)
    result = {'uncorrelated' : del_chi2 , 'correlated' : del_chi2_corr}
    return result
    
print("\033[37m     Loaded pbarlike 1.0")