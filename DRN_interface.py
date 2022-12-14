#%% Imports
from preamble import *

#%% DRN class

class DRNet:
    def __init__(self,propagation_parameters,propagation_model,prevent_extrapolation,data,errors,data_cov,xsection_cov):
        print("Initializing DRN")
        self.errors = errors
        self.data = data
        self.pe = prevent_extrapolation
        model_options = ['run1', 'DIFF.BRK', 'INJ.BRK+vA']
        solar_modulation_options = {'run1':self.solar_mod_run1, 'DIFF.BRK':self.solar_mod_brk, 'INJ.BRK+vA':self.solar_mod_brk}
        if propagation_model in model_options:
            self.propagation_model = propagation_model
        else:
            print()
            print('The propagation model "%s" is not provided in this tool. It will be set to default (DIFF.BRK).'%propagation_model)
            self.propagation_model = 'DIFF.BRK'
        self.dep_path = script_directory + '/dependencies/' + self.propagation_model + '/'
        if xsection_cov and self.propagation_model in ['DIFF.BRK', 'INJ.BRK+vA'] :
            xsection_cov = np.matrix(np.genfromtxt(self.dep_path+'CovMatrix_AMS02_pbar_CrossSection_'+str(self.propagation_model)+'.txt'))[14:,14:]
        elif not xsection_cov or self.propagation_model == 'run1' :
            xsection_cov = np.matrix(np.ones((44,44)))
        else :
            pass 
        self.cov_inv = (data_cov+xsection_cov).I
        self.solar_modulation = solar_modulation_options[self.propagation_model]
        self.load_deps()
        self.load_pp_data()
        self.preprocessing_prop_params(propagation_parameters)
        self.phi_CR_LIS = self.CR_sim()
        # print('phiCRLIS',self.phi_CR_LIS.shape)
        self.phi_CR   = self.solar_modulation(self.phi_CR_LIS) 
        self.chi2_CR_uncorr = self.chi2(self.phi_CR)
        # print('cuncorr',self.chi2_CR_uncorr)
        self.chi2_CR_corr = self.chi2_cov(self.phi_CR)
        self.chi2_CR_diff = np.clip(self.chi2_CR_uncorr - self.chi2_CR_corr,  -500,500)
        self.CR_marginalized_likelihood = np.sum(np.exp(self.chi2_CR_diff/2),axis=-1)
        print('\n The simulation tool has been initiated. \n')

    def load_deps(self):
        # DM antiprotons    
        self.DM_trafos = np.load(self.dep_path + 'DM_trafos_x.npy', allow_pickle = True)
        # Secondary antiprotons        
        self.S_trafos = np.load(self.dep_path + 'S_trafos.npy', allow_pickle = True)
    
    def load_pp_data(self):
        # Setting length of propagation parameters according to diffusion model
        N_pp = {'run1': 11, 'DIFF.BRK': 10, 'INJ.BRK+vA': 12}
        self.N_pp = N_pp[self.propagation_model]
        # Defining names of parameters
        self.strings = {'run1' : ['gamma 1,p', 'gamma 1', 'gamma 2,p', 'gamma 2', 'R_0', 's_0', 'D_0', 'delta', 'v_Alfven', 'v_0,c', 'z_h'],
                   'DIFF.BRK' : ['gamma 2,p', 'gamma 2', 'D0xx', 'delta_l', 'delta', 'delta_h - delta', 'R_D0', 's_D', 'R_D1', 'v_0'],
                   'INJ.BRK+vA': ['gamma 1,p', 'gamma 1', 'R_0,inj', 's', 'gamma 2,p', 'gamma 2', 'D_0', 'delta', 'delta_h - delta', 'R_1D', 'v_0', 'v_Alfven']} # TO DO: add
        # Loading priors
        mins_pp = {'run1' : np.array([1.63, 1.6, 2.38, 2.34, 5000, 0.32, 1.15e28, 0.375, 0, 0, 2]),
                   'DIFF.BRK' : np.array([2.249, 2.194, 3.411e+28, -9.66635e-01, 4.794e-01, -2.000e-01, 3.044e+03, 3.127e-01, 1.217e+0, 0]),
                   'INJ.BRK+vA': [1.59786, 1.60102, 4939.44, 0.221776, 2.41369, 2.36049,3.53e+28, 0.301255, -0.171395, 125612, 0, 14.3322]}
        maxs_pp = {'run1' : np.array([1.9, 1.88, 2.45, 2.4, 9500, 0.5, 5.2e28, 0.446, 11, 12.8, 7]),
                   'DIFF.BRK' : np.array([2.37e+00, 2.314e+00, 4.454e+28, -3.677e-01, 6.048e-01, -8.330e-02, 4.928e+03, 5.142e-01, 3.154e+05, 1.447e+01]),
                   'INJ.BRK+vA': [1.84643, 1.84721, 8765.77, 0.45543, 2.49947, 2.44248, 5.49E+28, 0.41704, -0.0398135, 413544, 8.61201, 29.206]}
        self.mins_pp = mins_pp[self.propagation_model]
        self.maxs_pp = maxs_pp[self.propagation_model]
        # Loading multinest sample for the corresponding diffusion model
        self.mns = (np.genfromtxt(self.dep_path + 'multinest_sample.dat'))
        self.mnpp = self.mns[:,:self.N_pp]

    # propagation_parameters - (n,N_pp) shaped array if flux is predicted for n DM_masses
    def preprocessing_prop_params(self, propagation_parameters):
        # Converting propagation parameters to 2D array. Required transformations will be done within simulations (DM_sim, CR_sim)
        if type(propagation_parameters) == list :
            propagation_parameters = np.array(propagation_parameters)[np.newaxis,:]
        if propagation_parameters.ndim == 1:
            propagation_parameters = propagation_parameters[np.newaxis,:]
        # Checking if given propagation parameters are custom or dummy
        if propagation_parameters.all == 0 : 
            self.pp = self.mnpp   
            self.marginalization = True  
        # Checking if correct number of propagation parameters are given
        elif propagation_parameters.shape[1] != self.N_pp :
            print('The number of propagation parameters is not consistent with the propagation model. The default multinest sample will be used for marginalization.')
            self.pp = self.mnpp  
            self.marginalization = True 
        # Checking if the given propagation parameters are inside the region in which the network is trained
        elif self.pe:
            for i in range(self.N_pp):
                if np.min(propagation_parameters[:, i]) <= self.mins_pp[i] or np.max(propagation_parameters[:, i]) >= self.maxs_pp[i]:
                    print("At least one of the inputs for %s is outside the trained parameter ranges. No output will be given. The default multinest sample will be used for marginalization. " % (self.strings[self.propagation_model])[i])
                    print(np.min(propagation_parameters[:, i]),np.max(propagation_parameters[:, i]))
                    self.pp = self.mnpp
                    self.marginalization = True
                    break
                else :
                    self.pp = propagation_parameters
                    self.marginalization = False
        else :
            self.pp = propagation_parameters
            self.marginalization = False   
        self.pp_ur = self.pp
        # print(np.shape(self.pp))  
    # DM_masses - 1D array of shape (n,)
    def preprocessing_DMparams(self, DM_mass, br_fr, sigma_v):
        # Preparing DM masses
        # Converting float DM mass to 1D array
        if type(DM_mass) == float or type(DM_mass)== np.float64 or type(DM_mass) == np.int64:
            DM_mass = np.array([DM_mass])
        elif type(DM_mass) == list :
            DM_mass = np.array(DM_mass)
        self.DM_mass = DM_mass
        # print('lenDMmass',len(self.DM_mass))
        self.DM_mass_r = np.repeat(DM_mass,len(self.pp),axis=0)
        # Min-max standardization of DM masses
        m_DM = ((np.log10(DM_mass)+3) - np.log10(5e3)) / (np.log10(5e6)-np.log10(5e3))
        # Repeating mass for the given number of propagation parameter points
        self.m_DM = np.repeat(m_DM,len(self.pp),axis=0)

        # Preparing branching fractions
        # Converting 1D bf array to 2D array
        if type(br_fr)==list:
            br_fr = np.array([br_fr])
        if br_fr.ndim == 1:
            br_fr = br_fr[np.newaxis,:] 
        # Replacing zeros by 1e-5 and renormalizing
        # br_fr = br_fr/np.sum(br_fr, axis = -1)[:,None] # initial normalization; already done in br_fr function in pbarlike.py
        masked_array = np.where(br_fr < 1e-5, 0, 1) # ones for every fs >= 1e-5
        masked_reversed = np.ones_like(masked_array) - masked_array # ones for every fs < 1e-5
        masked_rf = masked_array * br_fr # array with entries only >= 1e-5, else 0
        norm = np.sum(masked_rf, axis = -1)
        if norm==0.:
            norm=1
        scaling = (1-np.sum(masked_reversed, axis = -1)*1e-5)/norm # scaling for each >=1e-5 fs, while keeping relative fractions and normalizations
        bf_temp = masked_rf * scaling[:,None] + masked_reversed*1e-5 # scale fs >=1e-5 and set other to 1e-5
        # Processing braching fractions 
        bf = (np.log10(bf_temp) - np.array(self.DM_trafos[1,0])) / (np.array(self.DM_trafos[1,1])- np.array(self.DM_trafos[1,0])) 
        bf_r = np.repeat(bf,len(self.pp),axis=0)
        self.bf = bf_r

        # Repeating propagation parameters and branching fractions if necessary
        if len(self.DM_mass)>1:
            self.pp = np.tile(self.pp,(len(self.DM_mass),1))
            self.bf = np.repeat(bf_r,len(self.DM_mass),axis=0)
        
        # Preventing extrapolation
        self.stop_sim = False
        # Checking if the given parameter is inside the region in which the network is trained
        stop_DM = False
        #Checking DM parameters
        if np.min(self.DM_mass) < 5 or np.max(self.DM_mass) > 5e3:
            print('\n At least one of the given DM masses is outside of the provided range (5 GeV to 5 TeV). DM antiproton flux cannot be predicted.')
            stop_DM = True
        # Checking if the given parameter is inside the region in which the network is trained
        if np.min(bf) < 1e-5 or np.max(bf) > 1 :
            print(bf)
            print('The given branching fractions were not in the range of trained parameters or not normalized to one. Values below 1e-5 were mapped to 1e-5 and the remaining fractions normalized accordingly.')
            stop_DM = True 
        if stop_DM:
            self.stop_sim = True
            
        # Preparing thermally averaged annihilation cross-section (sigma_v)
        # Converting float sigma_v to 1D array
        if type(sigma_v) == float or type(sigma_v)== np.float64 or type(sigma_v) == np.int64:
            sigma_v = np.array([sigma_v])
        if type(sigma_v) == list:
            sigma_v = np.array(sigma_v)
        self.sv = sigma_v
        
    # Processing inputs, obtaining prediction from DMNet, and postprocessing prediction:
    def DM_sim(self): 
        # Transforming propagation parameters
        pp = ((self.pp - np.array(self.DM_trafos[0,0]))/np.array(self.DM_trafos[0,1]))
        # x - (n,40) array of x values at which network predicts output ; x = E/m_DM
        if self.propagation_model == 'run1':
            min_x = 0
        else: 
            min_x = -0.1 # Necessary for model without reacceleration (DM antiproton spectra diverge for E -> m_DM, x -> 0)
        x_temp = 10**(np.linspace(-3.7, min_x, 40))
        x = np.repeat([x_temp],len(self.DM_mass_r),axis=0)
        # E_dmnet - (n,40) array of kinetic energy(KE) per nucleon values at which network predicts output
        E_dmnet = x*self.DM_mass_r[:,None]
        # y_DM_x - (n,40) array of y values predicted by the DMNet for different x values; y(x) = log10(m^3 x phi(E))        
        # DM_model = tf.keras.models.load_model(self.DM_model)
        DM_model = tf.keras.models.load_model(self.dep_path + 'DM_model_x.h5')

        # print('Preprocessed mass:',self.m_DM)
        # print('Preprocessed branching fractions:',self.bf)
        # print('Preprocessed prop params for DMNet:',pp)

        y_DM_x = DM_model([self.m_DM,self.bf,pp])

        # print('DMNet Ouput:',y_DM_x)

        # Releasing memory
        tf.keras.backend.clear_session()
        del DM_model    
        gc.collect()
        # phi_dmnet - (n,40) array of flux values predicted by the DMNet for different KE per nucleon values
        phi_dmnet = 10**(y_DM_x) / (self.DM_mass_r[:,None]**3 * x)
        # phi_DM_LIS - (n,28) array of flux values interpolated to obtain fluxes at the same KE per nucleon values as in E_drn so that it can
        # be added to phi_CR_LIS. Only after solar modulation, the flux is to be interpolated to E_ams values.
        phi_DM_LIS = np.zeros((len(self.DM_mass_r),len(E_drn)))
        # Flux predicted by DMNet is only reliable in the allowed x range, i.e. for only those KE per nucleon in E_drn 
        # which for a given DM mass fall within the allowed x*m values. Thus we make a list of these allowed E_drn values, interpolate
        # the flux only at these values and set all other flux values at other E_drn to zero.
        for i in range(len(self.DM_mass_r)):
            E_drn_allowed = []
            indices = []
            for j in range(len(E_drn)):
                if E_drn[j]/self.DM_mass_r[i] >= 10**(-3.7) and E_drn[j]/self.DM_mass_r[i] <= 1:
                    E_drn_allowed.append(E_drn[j])
                    indices.append(j)
            phi_DM_LIS[i,indices]  = np.exp(np.interp(np.log(E_drn_allowed), np.log(E_dmnet[i]), np.log(phi_dmnet[i])))
        # phi_DM_LIS = np.reshape(phi_DM_LIS,(len(self.DM_mass),len(self.pp_ur),len(E_drn)))
        # print('Interpolated DMNet Output:',phi_DM_LIS)
        return phi_DM_LIS
        # Outputs 2D phi_DM_LIS - (n,58) array of flux values interpolated to obtain fluxes at the same KE per nucleon values as in E_drn

    def CR_sim(self):  
        # Preprocessing propagation parameters
        pp = ((self.pp - np.array(self.S_trafos[0]))/np.array(self.S_trafos[1]))
        # print('Preprocessed prop params for SNet:',pp)
        # y_CR - (28,) array of y values predicted by the sNet at different KE per nucleon values in E_drn ; y(E) = log10(E^2.7 phi(E))
        # S_model = tf.keras.models.load_model(self.S_model)
        S_model =tf.keras.models.load_model(self.dep_path + 'S_model.h5')
        y_CR = S_model(pp)
        # print('SNet Output:',y_CR)
        # Releasing memory
        tf.keras.backend.clear_session()
        del S_model
        gc.collect()
        # phi_CR_LIS - (28,) array of flux values predicted by the sNet at different KE per nucleon values in E_drn
        phi_CR_LIS = 10**(y_CR)/E_drn**2.7
        # print('E_drn:',E_drn,E_drn.dtype)
        # print('phi_CR_LIS:',phi_CR_LIS)
        return phi_CR_LIS
        # Outputs phi_CR_LIS - (28,) array of flux values predicted by the sNet at different KE per nucleon values in E_drn

    def LIS_sim(self):
        if self.stop_sim :
            print('The DM antiproton flux cannot be predicted by DRN due to atleast one parameter outside the region in which the network is trained.')
        else:
            DRN_output = self.DM_sim()
            phi_DM_LIS = np.array([self.sv[i]/10**(-25.5228)*DRN_output for i in range(len(self.sv))])
            # print('phidmlis:',phi_DM_LIS)
            phi_DM_LIS = np.reshape(phi_DM_LIS,(len(self.sv)*len(self.DM_mass),len(self.pp_ur),len(E_drn)))
            # Outputs 2D phi_DM_LIS - (n,28) array of flux values interpolated to obtain fluxes at the same KE per nucleon values as in E_drn
        return phi_DM_LIS

    # Calculates chi2 using ams data, ams errors and predicted flux
    def chi2(self,phi_pred):
        diff = self.data - phi_pred
        if phi_pred.ndim == 1:
            return np.sum((diff[14:])**2 / self.errors[14:]**2,axis = -1)
        else :
            return np.sum((diff[:,14:])**2 / self.errors[14:]**2,axis = -1)

    # Calculates chi2 using ams data, covariance matrix and predicted flux
    def chi2_cov(self,phi_pred):
        diff = self.data-phi_pred
        if phi_pred.ndim == 1:
            return np.diag((diff[14:]) @ self.cov_inv @ (diff[14:]).T)
        else :
            return np.diag((diff[:,14:]) @ self.cov_inv @ (diff[:,14:]).T)
    
    # Applying solar modulation to flux predicted at the local interstellar medium
    def solar_mod(self,phi_LIS, V, Z=-1., A=1., m=m_p ):
        # E_LIS_ams - (58,) array of KE per nucleon values at LIS which after solar modulation reduce to E_ams
        E_LIS_ams = E_ams + np.abs(Z)/A * V
        # phi_LIS_interp - (n,58) array of flux values interpolated to the above E values.
        phi_LIS_interp = np.exp(np.interp(np.log(E_LIS_ams),np.log(E_drn),np.log(phi_LIS)))
        # phi_earth - (n,58) array of flux values simulated as that which is measured by AMS 02, i.e., flux after solar modulation
        phi_earth = phi_LIS_interp * (E_ams**2 + 2*E_ams*m)/(E_LIS_ams**2 + 2*E_LIS_ams*m)
        return phi_earth
        # Outputs (n,58) array containing flux values that need to be compared with AMS 02 measurements

    def nuisance_estimation(self,phi_LIS):
        # Defining the function to be minimized to profile the nuisance parameters
        def lsq(nuisance_parameters):
            # # V - solar modulation potential
            V = nuisance_parameters[0]
            # # A - normalization constant
            A = nuisance_parameters[1]
            # phi_pred - (1,58) array containing values of normalized, solar modulated flux values 
            phi_pred = self.solar_mod(phi_LIS*A, V )
            # chi2_temp - a scalar value of weighted sum of squared difference between the normalized, solar modulated flux and flux measured by AMS02
            chi2_temp = self.chi2(phi_pred)
            return chi2_temp
        
        # Initiating whatever to minimize the chi squared function (iminuit<2)
        profiling = Minuit. from_array_func(fcn   = lsq ,#grad = jax.grad(lsq), # lsq - function to be minimized
                                            start = np.array([0.6, 1]), # starting values for the parameters to be estimated
                                            error = np.array([0.001, 0.001]), # step sizes for gradient descent for each of the parameters
                                            limit = np.array([[0.2, 0.9], [0.1, 5]]), # allowed interval for the parameters
                                            #name  = np.array(['phi_AMS-02_pbar', 'norm_AMS-02_pbar']), 
                                            errordef=1) # errordef = 1 for Least squares function; errordef = 0.5 for maximum likelihood function
        
        # migrad - algorithm for gradient descent (?)
        profiling.migrad()
        # V_profiled,A-profiled - values of estimated parameters
        V_profiled,A_profiled = profiling.np_values()
        return self.solar_mod(phi_LIS*A_profiled, V_profiled)
        # Outputs V_profiled,A-profiled - values of estimated parameters

        # # Initiating whatever to minimize the chi squared function (iminuit>2)
        # profiling = Minuit(fcn   = lsq ,#grad = jax.grad(lsq), # lsq - function to be minimized
        #                     V = 0.6,A = 1) # starting values for the parameters to be estimated
                            
        # profiling.errors ["V"] = 0.001 # step sizes for gradient descent for each of the parameters
        # profiling.errors ["A"] = 0.001
        # profiling.limits["V"] = (0.2,0.9) # allowed interval for the parameters
        # profiling.limits["A"] = (0.1, 5)
        # profiling.errordef = 0.5 # errordef = 1 for Least squares function; errordef = 0.5 for maximum likelihood function                                           
        
        # # migrad - algorithm for gradient descent (?)
        # profiling.migrad()
        # # V_profiled,A-profiled - values of estimated parameters
        # V_profiled = profiling.values["V"]
        # A_profiled = profiling.values["A"]
        # return self.solar_mod(phi_LIS*A_profiled, V_profiled)

    def nuisance_estimation_brk(self,phi_LIS):
        # Defining the function to be minimized to profile the nuisance parameters
        def lsq(V):
            # V - solar modulation potential
            # phi_pred - (1,58) array containing values of normalized, solar modulated flux values 
            phi_pred = self.solar_mod(phi_LIS, V )
            # chi2_temp - a scalar value of weighted sum of squared difference between the normalized, solar modulated flux and flux measured by AMS02
            chi2_temp = self.chi2(phi_pred)
            return chi2_temp
        
        # Initiating whatever to minimize the chi squared function (iminuit<2)
        profiling = Minuit.from_array_func(fcn   = lsq ,#grad = jax.grad(lsq), # lsq - function to be minimized
                                            start = np.array([0.6]), # starting values for the parameters to be estimated
                                            error = np.array([0.001]), # step sizes for gradient descent for each of the parameters
                                            limit = np.array([[0.2, 0.9]]), # allowed interval for the parameters
                                            #name  = np.array(['phi_AMS-02_pbar', 'norm_AMS-02_pbar']), 
                                            errordef=1) # errordef = 1 for Least squares function; errordef = 0.5 for maximum likelihood function
        
        # migrad - algorithm for gradient descent (?)
        profiling.migrad()
        # V_profiled,A-profiled - values of estimated parameters
        V_profiled = profiling.np_values()

        # # Initiating whatever to minimize the chi squared function (iminuit>2)
        # profiling = Minuit(fcn = lsq , #grad = jax.grad(lsq), # lsq - function to be minimized
        #                     V = np.array([0.6])) # starting values for the parameters to be estimated
        # profiling.errors = 0.001 # step sizes for gradient descent for each of the parameters
        # profiling.limits["V"] =  (0.2,0.9) # allowed interval for the parameters
        # profiling.errordef = 0.5 # errordef = 1 for Least squares function; errordef = 0.5 for maximum likelihood function
        
        # # migrad - algorithm for gradient descent (?)
        # profiling.migrad()
        # # V_profiled,A-profiled - values of estimated parameters
        # V_profiled = profiling.values
        # # print(V_profiled)
        return self.solar_mod(phi_LIS, V_profiled)
        # Outputs V_profiled,A-profiled - values of estimated parameters
    
    def solar_mod_run1(self,phi_LIS):
        return np.array([self.nuisance_estimation(phi_LIS[i]) for i in range(len(phi_LIS))])

    def solar_mod_brk(self, phi_LIS):
        phi = {'DIFF.BRK': 12, 'INJ.BRK+vA': 14}
        delta_phi_bar = {'DIFF.BRK': 13, 'INJ.BRK+vA': 15}
        V = self.mns[:,phi[self.propagation_model]] + self.mns[:,delta_phi_bar[self.propagation_model]]
        # print('Solar mod potential:',V)
        if self.marginalization:
            return np.array([self.solar_mod(phi_LIS[i],V[i]) for i in range(len(phi_LIS))])
        else:
            return np.array([self.nuisance_estimation_brk(phi_LIS[i]) for i in range(len(phi_LIS))])

    def TOA_sim(self, phi_DM_LIS):
        phi_LIS = self.phi_CR_LIS + phi_DM_LIS
        # print('phi_LIS:',phi_LIS)
        phi_DMCR = np.array([self.solar_modulation(phi_LIS[i]) for i in range(len(phi_LIS))])
        # print('phi_DMCR',phi_DMCR)
        return phi_DMCR
    
    def del_chi2(self, phi_DMCR):
        # print('phiDMCR after solmod:',phi_DMCR)
        del_chi2 = []
        for i in range(len(phi_DMCR)):
            # print('phiDMCRi:',phi_DMCR[i])
            chi2_DMCR = self.chi2(phi_DMCR[i])
            # print('chisquares:',chi2_DMCR)
            chi2_diff = np.clip(self.chi2_CR_uncorr - chi2_DMCR,  -500,500)
            # print('chi2_diff:',chi2_diff)
            del_chi2_t = -2*np.log( 1/len(self.pp_ur) * np.sum(np.exp(chi2_diff/2),axis=-1) )
            # print('Len pp:',len(self.pp_ur))
            # print('del_chi2_t:',del_chi2_t)
            del_chi2.append(del_chi2_t)
        return del_chi2

    def del_chi2_corr(self, phi_DMCR):
        delta_chi2_cov = []
        for i in range(len(phi_DMCR)):
            chi2_DMCR = self.chi2_cov(phi_DMCR[i])
            chi2_DM_diff = np.clip(self.chi2_CR_uncorr - chi2_DMCR,  -500,500)
            delta_chi2_t = -2*np.log(np.sum(np.exp(chi2_DM_diff/2),axis=-1) / self.CR_marginalized_likelihood)
            delta_chi2_cov.append(delta_chi2_t)
        return delta_chi2_cov

    # # theta_prop - (number_of_m_DM,11) array containing generally accepteed values of propagation parameters for run1
    # def usual_theta_prop(number_of_m_DM,gamma_1p = 1.80, gamma_1 = 1.79, gamma_2p = 2.405, gamma_2 = 2.357, R_0 = 7.92e3, s = 0.37, D_0 = 2.05e28, delta = 0.419, v_alfven = 8.84, v_0c = 0.09, z_h = 2.60):
    #     theta_prop_temp = np.array([gamma_1p, gamma_1, gamma_2p, gamma_2, R_0, s, D_0, delta, v_alfven, v_0c, z_h])
    #     theta_prop = np.repeat([theta_prop_temp],number_of_m_DM,axis=0)
    #     return theta_prop 