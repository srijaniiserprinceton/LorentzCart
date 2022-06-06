import numpy as np

def computeKerns(kern_dict):
    Kxx = make_Kxx(kern_dict)
    Kyy = 1. * Kxx
    
    Kzz = make_Kzz(kern_dict)
    
    Kzk, Kzk_ = make_Kzk_Kzk_(kern_dict)
    
    Kkz, Kk_z = make_Kkz_Kk_z(kern_dict)
    
    Kkk, Kk_k_ = make_Kkk_Kk_k_(kern_dict)
    
    Kkk_, Kk_k = make_Kkk__Kk_k(kern_dict)
    
    return Kxx, Kyy, Kzz, Kzk, Kzk_, Kkz, Kk_z, Kkk, Kk_k_, Kkk_, Kk_k

    
def extract_eig1_eig2_dict(kern_dict, swap=False):
    if(swap == False):
        k1, k2 = kern_dict['abs_k'], kern_dict['abs_k_']
        H1, V1, dH1, dV1, d2H1, d2V1 = kern_dict['H_k'], kern_dict['V_k'],\
                                       kern_dict['dzH_k'], kern_dict['dzV_k'],\
                                       kern_dict['d2zH_k'], kern_dict['d2zV_k']
        H2, V2, dH2, dV2, d2H2, d2V2 = kern_dict['H_k_'], kern_dict['V_k_'],\
                                       kern_dict['dzH_k_'], kern_dict['dzV_k_'],\
                                       kern_dict['d2zH_k_'], kern_dict['d2zV_k_']
        
    else:
        k2, k1 = kern_dict['abs_k'], kern_dict['abs_k_']
        H2, V2, dH2, dV2, d2H2, d2V2 = kern_dict['H_k'], kern_dict['V_k'],\
                                       kern_dict['dzH_k'], kern_dict['dzV_k'],\
                                       kern_dict['d2zH_k'], kern_dict['d2zV_k']
        H1, V1, dH1, dV1, d2H1, d2V1 = kern_dict['H_k_'], kern_dict['V_k_'],\
                                       kern_dict['dzH_k_'], kern_dict['dzV_k_'],\
                                       kern_dict['d2zH_k_'], kern_dict['d2zV_k_']

    return k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2


def make_Kxx(kern_dict):
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    Kxx = dV1 * dV2 + k1 * k2 * H1 * H2 - k2 * dV1 * H2 - k1 * dV2 * H1
    
    return Kxx

    
def make_Kzz(kern_dict):
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    long_term1 = k1 * dV1 * H2 + k2 * dV2 * H1 + k1 * V1 * dH2 + k2 * V2 * dH1
                 
    long_term2 = k2 * dV1 * H2 + k1 * dV2 * H1 + k2 * V1 * dH2 + k1 * V2 * dH1

    Kzz = (dH1 * dH2 + 0.5 * long_term1) * kern_dict['khdotk_h'] +\
          k1 * k2 * H1 * H2 + 0.5 * long_term2
    
    return Kzz

def make_Kzk_Kzk_(kern_dict):
    def make_Kzk():
        kern = (k1 * H1 * dH2 + 0.5 * k1 * dH1 * H2) * kern_dict['khdotk_h'] +\
               0.5 * (k1 * V1 * dV2 + d2H1 * V2 - dH1 * dV2 + k2 * dH1 * H2 -\
                      k1 * V2 * dV1 + k1**2 * V2 * H1)
        return kern
        
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    Kzk = make_Kzk()
    
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                extract_eig1_eig2_dict(kern_dict, swap=True)
    Kzk_ = -1. *  make_Kzk()

    # need to be converted to imaginary later
    return Kzk, Kzk_

def make_Kkz_Kk_z(kern_dict):
    def make_Kkz():
        kern = 0.5 * (k1 * H1 * dH2 + k1**2 * V1 * H2 + k1 * k2 * V2 * H1) *\
               kern_dict['khdotk_h'] + 0.5 * (k1 * dV1 * V2 - k1 * dV2 * V1 +\
               3. * k1 * k2 * V1 * H2 + k2 * H1 * dH2 - H1 * d2V2)
        
        return kern
    
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    Kkz = make_Kkz()

    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                extract_eig1_eig2_dict(kern_dict, swap=True)
    Kk_z = -1. *  make_Kkz()

    # need to be converted to imaginary later
    return Kkz, Kk_z

def make_Kkk_Kk_k_(kern_dict):
    def make_Kkk():
        kern = 1.5 * (k1 * H1 * dV2 - k1 * k2 * H1 * H2) -\
               0.5 * (k1**2 * H1 * H2 * kern_dict['khdotk_h'] +\
                      k1 * dH1 * V2)

        return kern
        
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    Kkk = make_Kkk()

    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                extract_eig1_eig2_dict(kern_dict, swap=True)
    Kk_k_ = make_Kkk()

    return Kkk, Kk_k_
    
def make_Kkk__Kk_k(kern_dict):
    def make_Kkk_():
        kern = k1 * k2 * H1 * H2 * kern_dict['khdotk_h'] +\
               0.5 * (k1 * k2 * V1 * V2 - k2 * H1 * dV2 +\
                      k2**2 * H1 * H2 + k1 * dH2 * V1)
        
        return kern
    
    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                            extract_eig1_eig2_dict(kern_dict)
    Kkk_ = make_Kkk_()

    k1, k2, H1, V1, dH1, dV1, d2H1, d2V1, H2, V2, dH2, dV2, d2H2, d2V2 =\
                                extract_eig1_eig2_dict(kern_dict, swap=True)
    Kk_k = make_Kkk_()

    return Kkk_, Kk_k
