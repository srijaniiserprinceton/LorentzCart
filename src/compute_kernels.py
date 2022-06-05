import numpy as np
import scipy

class CartKerns:
    def __init__(self, z, n, n_, k_grid, kmin, kmax,
                 absq_range, fj, rho):
        self.z = z 
        self.rho = rho
        self.fj = fj
        
        self.n = n
        self.n_ = n_
        self.kx = k_grid
        self.ky = k_grid
        
        # the range of modes we want to use
        self.kmin = kmin
        self.kmax = kmax
        self.absq_range = absq_range
        
        # constructing the k meshgrid
        self.kxg, self.kyg = np.meshgrid(self.kx, self.ky, indexing='ij')
        self.abskg = np.sqrt(self.kxg**2 + self.kyg**2)
        
        # picking out the modes which are in kRmin < |k|R < kRmax
        self.k_mask = np.where((self.abskg < self.kmax)*\
                               (self.abskg > self.kmin), 1, 0)
        self.nmodes = np.where(self.k_mask == True)[0].size
        
        # indices of kx and ky for which 600 < |k|R < 1500 
        self.kx_ind, self.ky_ind = np.where(self.k_mask == 1)
        
        # load eigenfunctions of k
        eig = np.load('gyreEigenFunctions/eigs{:02d}.npz'.format(n))
        self.raw_H = eig['Xi_h']
        self.raw_V = eig['Xi_z']
        self.raw_k = eig['eig_k']
        
        # load eigenfunctions of k'
        eig_ = np.load('gyreEigenFunctions/eigs{:02d}.npz'.format(n_))
        self.raw_H_ = eig_['Xi_h']
        self.raw_V_ = eig_['Xi_z']
        self.raw_k_ = eig_['eig_k']

        # gradients needed
        self.raw_dzH, self.raw_dzV, self.raw_dzH_, self.raw_dzV_,\
        self.dzrho, self.dzfj = self.grad_z()

        # the miscellaneous interpolation functions needed
        self.ITP_H, self.ITP_V, self.ITP_H_,\
        self.ITP_V_, self.ITP_dzH, self.ITP_dzV,\
        self.ITP_dzH_, self.ITP_dzV_ = self.interpolate_eigs()        
        
    def grad_z(self):
        # get the derivatives                                                                
        dzH = np.gradient(self.raw_H, self.z, axis=1)
        dzV = np.gradient(self.raw_V, self.z, axis=1)
        dzH_ = np.gradient(self.raw_H_, self.z, axis=1)
        dzV_ = np.gradient(self.raw_V_, self.z, axis=1)
        dzrho = np.gradient(self.rho, self.z)
        dzfj = np.gradient(self.fj, self.z, axis=1)
        
        return dzH, dzV, dzH_, dzV_, dzrho, dzfj

    def interpolate_eigs(self):
        # interpolate because the quantities in the eigenfunction
        # files are not always defined for the same values of kx,ky as we need
        ITP_H   = scipy.interpolate.interp2d(self.raw_k, self.z, self.raw_H.T, 
                                             kind='linear', 
                                             bounds_error=False,fill_value=np.nan)
        ITP_V   = scipy.interpolate.interp2d(self.raw_k, self.z, self.raw_V.T, 
                                             kind='linear',
                                             bounds_error=False,fill_value=np.nan)
        
        ITP_H_   = scipy.interpolate.interp2d(self.raw_k_, self.z, self.raw_H_.T,
                                              kind='linear',
                                              bounds_error=False,fill_value=np.nan)
        ITP_V_   = scipy.interpolate.interp2d(self.raw_k_, self.z, self.raw_V_.T,
                                              kind='linear',
                                              bounds_error=False,fill_value=np.nan)
        
        ITP_dzH   = scipy.interpolate.interp2d(self.raw_k, self.z, self.raw_dzH.T,
                                               kind='linear',
                                               bounds_error=False,fill_value=np.nan)
        ITP_dzV   = scipy.interpolate.interp2d(self.raw_k, self.z, self.raw_dzV.T,
                                               kind='linear',
                                               bounds_error=False,fill_value=np.nan)

        ITP_dzH_   = scipy.interpolate.interp2d(self.raw_k_, self.z, self.raw_dzH_.T,
                                                kind='linear',
                                                bounds_error=False,fill_value=np.nan)
        ITP_dzV_   = scipy.interpolate.interp2d(self.raw_k_, self.z, self.raw_dzV_.T,
                                                kind='linear',
                                                bounds_error=False,fill_value=np.nan)

        return ITP_H, ITP_V, ITP_H_, ITP_V_, ITP_dzH, ITP_dzV, ITP_dzH_, ITP_dzV_
        
    
    # getting all the kernels for a certain q
    def compute_kernels(self, q_vec_ind):
        # kernels will be computed for this qx and qy
        qx_ind = q_vec_ind[0]
        qy_ind = q_vec_ind[1] 
                
        Poloidal_kernel = np.zeros((self.nmodes, self.fj.shape[0]))
        
        for idx, (kxi, kyi) in enumerate(zip(self.kx_ind, self.ky_ind)):
            # k vector, k unit vector, k norm
            k_vec = np.array([self.kx[kxi], self.ky[kyi]])
            k_hat = k_vec/np.linalg.norm(k_vec)
            abs_k = np.linalg.norm(k_vec)

            # compute |q|,q,k',|k'| and k'_hat
            # q = k' - k
            kxp_ind = kxi + qx_ind
            kyp_ind = kyi + qy_ind
            
            # if k' vector is outside of the grid
            if((kxp_ind > (len(self.kx)-1)) or (kyp_ind > (len(self.ky)-1))):
                continue

            # k' vector, k' unit vector, k' norm                                            
            kxp = self.kx[kxp_ind]
            kyp = self.ky[kyp_ind]
            kp_vec = np.array([kxp, kyp])
            abs_kp = np.sqrt(kxp**2 + kyp**2)
            kp_hat = kp_vec/abs_kp
        
            # q vector, q unit vector, q norm                                             
            qx = self.kx[kxp_ind] - self.kx[kxi]
            qy = self.ky[kyp_ind] - self.ky[kyi]
            q_vec = np.array([qx, qy])
            abs_q = np.sqrt(qx**2 + qy**2)
            q_hat = q_vec/abs_q

            # if abs_q is outside a desired range continue
            if((abs_q > self.absq_range[1]) or (abs_q < self.absq_range[0])):
                continue
                
            # dictionary of eig funcs
            kern_dict = {}
            
            # constructing the interpolated eigenfunctions for k
            kern_dict['H_k']  = self.ITP_H(abs_k, self.z).squeeze()
            kern_dict['V_k']  = self.ITP_V(abs_k, self.z).squeeze()
            kern_dict['dzH_k'] = self.ITP_dzH(abs_k, self.z).squeeze()
            kern_dict['dzV_k'] = self.ITP_dzV(abs_k, self.z).squeeze()
                                    
            # constructing the interpolated eigenfunctions for k'
            # these are real so conj is just for safety purposes if using complex efn later
            kern_dict['H_k_']  = np.conj(self.ITP_H_(abs_kp, self.z).squeeze())
            kern_dict['V_k_']  = np.conj(self.ITP_V_(abs_kp, self.z).squeeze())
            kern_dict['dzH_k_'] = np.conj(self.ITP_dzH_(abs_kp, self.z).squeeze())
            kern_dict['dzV_k_'] = np.conj(self.ITP_dzV_(abs_kp, self.z).squeeze())

            # other miscellaneous things needed
            kern_dict['abs_k'] = abs_k
            kern_dict['abs_k_'] = abs_kp
            kern_dict['abs_q'] = abs_q
            kern_dict['khdotk_h'] = np.dot(k_hat,kp_hat)
            kern_dict['kdotq'] = np.dot(k_vec,q_vec)

            # THIS CONDITION IS WRONG I THINK
            if np.sum(np.isnan(kp_hat)):
                continue
                            
            # WHY SHOULD THIS BE NEEDED IF THE INTERPOLATION WORKS FINE
            # If there is an interpolation error in the eigenfunctions, continue
            if np.sum(np.isnan([kern_dict['H_k_'], kern_dict['V_k_'],\
                                kern_dict['dzH_k_'], kern_dict['dzV_k_']])) != 0:
                continue
                        
            # the poloidal flow kernel
            Poloidal_kernel[idx] = self.flow_kern(kern_dict)       
            
        return Poloidal_kernel.astype(np.float32)
        
    
    def flow_kern(self, kern_dict):        
        '''Returns the poloidal component of flow kernels.
        '''
        
        scale_height = -self.rho / self.dzrho
        term1 = (kern_dict['abs_q']**2 * self.fj) *\
                (kern_dict['khdotk_h'] * kern_dict['dzH_k'] *\
                 kern_dict['H_k_'] +\
                 kern_dict['dzV_k'] * kern_dict['V_k_'])
        term2 = -1. * kern_dict['kdotq'] * (self.dzfj - self.fj/scale_height) *\
                (kern_dict['khdotk_h'] * kern_dict['H_k'] * kern_dict['H_k_'] +\
                 kern_dict['V_k'] * kern_dict['V_k_'])
        
        # compute integrand for P matrix                                                   
        integrand_P = -(term1 + term2)
        
        LZ = 1./kern_dict['abs_q'] if kern_dict['abs_q'] != 0 else 1
        return (np.trapz(integrand_P * self.rho[None, :], self.z, axis=1)) * LZ**2

    
    def Lorentz_kern(self, kern_dict):
        '''Calculates the different components of the Lorentz-stress kernels.
        '''
        Kxx, Kyy, Kzz, Kzk, Kzk_, Kkz, Kk_z, Kkk, Kk_k_, Kkk_, Kk_k =\
                                                    LorKern.computeKerns(kern_dict)
        return Kxx, Kyy, Kzz, Kzk, Kzk_, Kkz, Kk_z, Kkk, Kk_k_, Kkk_, Kk_k
