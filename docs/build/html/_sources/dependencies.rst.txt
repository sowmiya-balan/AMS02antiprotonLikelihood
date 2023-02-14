dependencies
------------

- ``pbar_ams02_7y.txt``: AMS-02 7-year antiproton data; description of columns included in same file.

- ``E.npy``: Energies at which sNet makes anitproton flux predictions.

- ``CovMatrix_AMS02_pbar_7y.txt``: Data correlations in AMS-02 antiproton data in the same energy
  bins as the antiproton flux measurements, as modeled in :ref:`5 <5>`

- ``<propagation_model>/DM_model_x.h5``: DMNet for ``<propagation_model>``

- ``<propagation_model>/S_model.h5``: sNet for ``<propagation_model>``

- ``<propagation_model>/CovMatrix_AMS02_pbar_CrossSection.txt``: Covariance matrix from antiproton production cross-section uncertainties as described in section 3.1 of :ref:`4 <4>`.

- ``<propagation_model>/multinest_sample.dat``: 
        Posterior sample of propagation parameters from fit to AMS-02 and Voyager data, see section 3.2 of :ref:`4 <4>`. For description of CR parameters, refer DRN `documentation <https://github.com/kathrinnp/DarkRayNet>`_.
        
        DIFF.BRK:

        ============  =======================  ======================================  
        Column no.    DRN parameter name       Description
        ============  =======================  ======================================   
        0             gamma 2,p	               CR propagation parameter	
        1             gamma_2                  CR propagation parameter	
        2             D_0                      CR propagation parameter	
        3             delta_l                  CR propagation parameter	
        4             delta                    CR propagation parameter	
        5             delta_h - delta          CR propagation parameter	
        6             R_D,0                    CR propagation parameter	
        7             s_D                      CR propagation parameter	
        8             R_D,1                    CR propagation parameter	
        9             v_0,c                    CR propagation parameter	
        10            A_XS,3He                 Nuisance parameter for 3He production	
        11            delta_XS3He              Nuisance parameter for 3He production	
        12            V_p                      Solar modulation potential	
        13            V_pbar-V_p               Solar modulation	potential
        14            A_p                      Overall normalization
        15            Abd_He                   Helium 4 isotopic abundance
        ============  =======================  ======================================	

        INJ.BRK+vA:

        ============  =======================  ======================================  
        Column no.    DRN parameter name       Description
        ============  =======================  ======================================   
        0             gamma_1,p                CR propagation parameter
        1             gamma_1                  CR propagation parameter
        2             R_0                      CR propagation parameter
        3             s                        CR propagation parameter
        4             gamma_2,p                CR propagation parameter
        5             gamma_2                  CR propagation parameter
        6             D_0                      CR propagation parameter
        7             delta                    CR propagation parameter
        8             delta_h - delta          CR propagation parameter
        9             R_1,D                    CR propagation parameter
        10            v_0                      CR propagation parameter
        11            v_A                      CR propagation parameter
        12            A_XS,3He                 Nuisance parameter for 3He production
        13            delta_XS3He              Nuisance parameter for 3He production
        14            V_p                      Solar modulation
        15            V_pbar-V_p               Solar modulation
        16            A_p                      Overall normalization
        17            Abd_He                   Helium 4 isotopic abundance
        ============  =======================  ======================================
