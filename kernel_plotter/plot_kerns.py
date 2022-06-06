import numpy as np
import matplotlib.pyplot as plt
plt.ion()

def plot_LorentzKern_z(z, Lorentz_kern):
    '''Plotting the 10 components (Kxx and Kyy are the same).
    '''

    del Lorentz_kern['Kyy']
    
    
    fig, ax = plt.subplots(2, 5, figsize=(15,10))
    
    # choosing 5 random kernels from all the ones in nmodes
    mode_ind_arr = np.random.randint(0, Lorentz_kern['Kxx'].shape[0], 1)

    for key_ind, key in enumerate(Lorentz_kern.keys()):
        row, col = key_ind//5, key_ind%5
        for mode_ind in mode_ind_arr:
            ax[row, col].plot(Lorentz_kern[f'{key}'][mode_ind], z)
        ax[row,col].set_title(f'{key}')
        ax[row, col].grid()
    
    #plt.figure()
    #plt.plot(Lorentz_kern['Kxx'][0], z)
    #plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('Lorentz_kernels.pdf')
