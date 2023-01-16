"""! @file pbar_ams02_7y.txt
     Antiproton data from 7-years of AMS-02 measurements.
     AMS Collaboration, M. Aguilar et al., *The Alpha Magnetic Spectrometer (AMS) on the
     international space station: Part II — Results from the first seven years*, Phys. Rept. 894
     (2021) 1–116.    

     @file E.npy
     Values of KE per nucleon at which sNet predicts flux.    

     @file CovMatrix_AMS02_pbar_7y.txt
     Covariance matrix of errors arising from statistical and systematic uncertainties.
 
     @file CovMatrix_AMS02_pbar_CrossSection.txt
     Covariance matrix of errors arising from antiproton production cross-section uncertainties.

     @file preamble.py
     Loads necessary modules; loads and processes data.
"""
# %% Imports and global variables
# print("\033[32m Loading pbarlike 1.0")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import gc
import numpy as np
import tensorflow as tf
for gpu in tf.config.experimental.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, True)
from iminuit import Minuit
print("\033[32m Imported required python modules - numpy, tensorflow, iminuit.Minuit")
import sys
from pbarlike import dirpath
# script_directory = os.path.dirname(os.path.realpath(__file__))
script_directory = dirpath
m_p = 0.9382720881604903  # Mass of proton in GeV (938.2720881604903 MeV)


# %% Loading and preparing data

# Loading 7 year ams02_data (for column description, see pbar_ams02_7y.txt file)
ams_data = np.genfromtxt(script_directory+'/dependencies/pbar_ams02_7y.txt')
R_ams = (ams_data[:,0]+ams_data[:,1])/2 
phi_in_R = ams_data[:,10] * ams_data[:,13]
error_R_ams = np.sqrt(ams_data[:,11]**2+ams_data[:,12]**2)*ams_data[:,13]

# Loading ams02 covariance matrix (58,58) for the 7-year antiproton data:
ams_7y_cov = np.matrix(np.genfromtxt(script_directory+'/dependencies/CovMatrix_AMS02_pbar_7y.txt'))[14:,14:]

def R_to_Ekin(R,z = -1, A = 1, m = m_p ):
    '''
    Converts R in GV to kinetic energy per nucleon in GeV: 
    .. math:: 
        E_k / A = \sqrt{(RZ/A)^2 + m^2}
    '''
    Z = np.abs(z)
    return np.sqrt(R**2 * (Z/A)**2 + m**2) - m

def flux_in_Ekin(flux_in_R,R,z=-1,A=1,m=m_p):
    '''
    Converting flux (m-2 sr-1 s-1 GV-1) in R to flux (m-2 sr-1 s-1 GeV-1) in E:
    .. math::
        \frac{d \phi}{d R} \frac{d R}{d (E_k/A)} = \frac{d \phi}{d R} \frac{1}{R} (\frac{A}{Z})^2 \sqrt{( \frac{RA}{Z})^2 + m^2}
    '''
    Z = np.abs(z)
    return flux_in_R /R * (A/Z)**2 * np.sqrt((R*Z/A)**2 + m**2)

# E_ams -(58,) array of KE per nucleon values at which AMS02 flux measurements are recorded for pbar
E_ams = np.array(R_to_Ekin(R_ams))
# phi_ams - (58,) array of flux values measured at E_ams
phi_ams = np.array(flux_in_Ekin(phi_in_R,R_ams))
# error_ams - (58,) array of error values of flux values measured at E_ams
error_ams = np.array(flux_in_Ekin(error_R_ams,R_ams))

# E_drn - (28,) array of KE per nucleon values at which sNet predicts flux values (same values at which training data are given to the sNet)
E_drn = np.array(np.load(script_directory+'/dependencies/E.npy'))
print("\033[32m Loaded and processed AMS-02 dataset and covariance matrices")
