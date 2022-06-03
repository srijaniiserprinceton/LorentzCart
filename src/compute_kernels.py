import numpy as np
import scipy

def compute_kernels(QXY):
    #.....kernels will be computed for this qx and qy
    qx_ind = QXY[0] ; qy_ind = QXY[1] 
    
     #.....indices of kx and ky for which 600 < |k|R < 1500 
    kx_ind, ky_ind = np.where(k_mask==1)
    
    #.....load eigenfunctions of k
    quant = np.load('gyreEigenFunctions/eigs{:02d}.npz'.format(n))
    Xi_h = quant['Xi_h'] ; Xi_z = quant['Xi_z'] ; eig_k = quant['eig_k']
    
    #.....load eigenfunctions of k'
    quant_p = np.load('gyreEigenFunctions/eigs{:02d}.npz'.format(npr))
    Xi_hp = quant_p['Xi_h'] ; Xi_zp = quant_p['Xi_z'] ; eig_kp = quant_p['eig_k']
    
    #.....get the derivatives
    dzXi_h = np.gradient(Xi_h,z,axis=1) ; dzXi_z = np.gradient(Xi_z,z,axis=1)
    dzXi_hp = np.gradient(Xi_hp,z,axis=1) ; dzXi_zp = np.gradient(Xi_zp,z,axis=1)
    drho = np.gradient(rho,z) ; dfj = np.gradient(fj,z,axis=1)

    #.....interpolate because the quantities in the eigenfunction files are not always defined for the same values of kx,ky as we need
    ITP_Xi_h   = scipy.interpolate.interp2d(eig_k, z, Xi_h.T, 
                                            kind='linear', 
                                            bounds_error=False,fill_value=np.nan)
    ITP_Xi_z   = scipy.interpolate.interp2d(eig_k, z, Xi_z.T, 
                                            kind='linear',
                                            bounds_error=False,fill_value=np.nan)
    ITP_dzXi_h = scipy.interpolate.interp2d(eig_k, z, dzXi_h.T,
                                            kind='linear',
                                            bounds_error=False,fill_value=np.nan)
    ITP_dzXi_z = scipy.interpolate.interp2d(eig_k, z, dzXi_z.T,
                                            kind='linear',
                                            bounds_error=False,fill_value=np.nan)

    ITP_Xi_hp   = scipy.interpolate.interp2d(eig_kp, z, Xi_hp.T,
                                             kind='linear',
                                             bounds_error=False,fill_value=np.nan)
    ITP_Xi_zp   = scipy.interpolate.interp2d(eig_kp, z, Xi_zp.T
                                             ,kind='linear',
                                             bounds_error=False,fill_value=np.nan)
    ITP_dzXi_hp = scipy.interpolate.interp2d(eig_kp, z, dzXi_hp.T,
                                             kind='linear',
                                             bounds_error=False,fill_value=np.nan)
    ITP_dzXi_zp = scipy.interpolate.interp2d(eig_kp, z, dzXi_zp.T,
                                             kind='linear',
                                             bounds_error=False,fill_value=np.nan)
    
    Poloidal_kernel = np.zeros((nmodes,fj.shape[0]))
    
    for idx,(kxi,kyi) in enumerate(zip(kx_ind,ky_ind)):
        k_vec = np.array([kx[kxi],ky[kyi]])
        k_hat = k_vec/np.linalg.norm(k_vec)
        abs_k = np.linalg.norm(k_vec)
        
        Xi_h_k  = ITP_Xi_h (abs_k,z).squeeze()
        Xi_z_k  = ITP_Xi_z (abs_k,z).squeeze()
        dzXi_h_k = ITP_dzXi_h(abs_k,z).squeeze()
        dzXi_z_k = ITP_dzXi_z(abs_k,z).squeeze()
        
        if (kxi+qx_ind < 0) or (kxi+qx_ind > len(kx)) or (kyi+qy_ind < 0) or (kyi+qy_ind > len(ky)):
            sigma_out_ind += 1
            continue;

        # compute |q|,q,k',|k'| and k'_hat
        # q = k' - k
        kxp_ind = kxi + qx_ind
        kyp_ind = kyi + qy_ind

        if kxp_ind > (len(kx)-1) or kyp_ind > (len(ky)-1):
            # if qx is outside of the grid, continue
            continue;

        qx = kx[kxp_ind] - kx[kxi]
        qy = ky[kyp_ind] - ky[kyi]
        abs_q = np.sqrt(qx**2 + qy**2)
        q_vec = np.array([qx,qy])

        kxp = kx[kxp_ind]
        kyp = ky[kyp_ind]
        abs_kp = np.sqrt(kxp**2 + kyp**2)
        kp_vec = np.array([kxp,kyp])
        kp_hat = kp_vec/np.linalg.norm(kp_vec)

        if np.sum(np.isnan(kp_hat)):
            continue;

        # if abs_q is outside a desired range continue
        if abs_q > absq_range[1] or abs_q < absq_range[0] :
            continue;

        Xi_h_kp   = ITP_Xi_hp (abs_kp,z).squeeze()
        Xi_z_kp   = ITP_Xi_zp (abs_kp,z).squeeze()
        dzXi_h_kp = ITP_dzXi_hp(abs_kp,z).squeeze()
        dzXi_z_kp = ITP_dzXi_zp(abs_kp,z).squeeze()

        # If there is an interpolation error in the eigenfunctions, continue
        if np.sum(np.isnan([Xi_h_kp,Xi_z_kp,dzXi_h_kp,dzXi_z_kp])) != 0:
            continue;

        kdotkp_hat = np.dot(k_hat,kp_hat)
        kdotq      = np.dot(k_vec,q_vec)

        scale_height = -rho/drho
        term1 = (abs_q**2*fj)*(kdotkp_hat*dzXi_h_k*np.conj(Xi_h_kp) +\
                dzXi_z_k*np.conj(Xi_z_kp))
        term2 = -np.dot(k_vec,q_vec)*(dfj-fj/scale_height) *\
                (kdotkp_hat*Xi_h_k*np.conj(Xi_h_kp) + Xi_z_k*np.conj(Xi_z_kp))

        # compute integrand for P matrix
        integrand_P = -(term1 + term2)
        LZ = 1/abs_q if abs_q != 0 else 1
        
        Poloidal_kernel[idx] = (np.trapz(integrand_P*rho[None,:],z,axis=1))*LZ**2

    return Poloidal_kernel.astype(np.float32)
