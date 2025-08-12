##This code tracks the Evolution of g* over time of the SM in the range T0 = 10 MeV to T0 = 0.01 MeV
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import module_definitions as defin

#%%
#entropy density before annihilation electron-positron.
target_value = 43 * np.pi**2 / 90 

#Entropy before annihilation- entropy after annihilation. For entropy conservation it should be zero
def entropy(T_nu, T):
    """
        Calculation of s2*a2^3 - s1*a1^3. 1 --> Before annihilation e+-. 2 --> After annihilation e+-

        Parameters
        ----------
            T:float
                Temperature in MeV of the plasma
            T_nu:float
                Temperature in MeV of the neutrinos

        Returns
        -------
            delta S:float
                s2*a2^3 - s1*a1^3
        """
    total= ((defin.rho_nu(T_nu) +  defin.P_nu(T_nu) )/T_nu + (defin.rho_e(T) + defin.rho_gamma(T) + defin.P_e(T) + defin.P_gamma(T))/T)/T_nu**3 #s2*a2^3/a1^3
    return total - target_value #s2a2^3 = a1a1^3

#%%Calculation of the temperature of the neutrinos and g* for sifferent T_gamma

def file_gstar(num_points = 1000, T_0 = 25, T_f = 0.01):
    
    """
Creates a file with the values of T_gamma, T_nu and g*.

Parameters
----------
    num_points : int, optional
        Number of points to calculate. Default: 1000

    T_0 : float, optional
        Initial temperature of gamma (max value) in MeV. Default: 25

    T_f : float, optional
        Final temperature of gamma (min value) in MeV. Default: 0.01

Returns
-------
    None
        Txt file with the following columns: T_gamma, T_nu, g*. 
        Name of the file: 'g_star_T_neutrino_{num_points}_T0_{T_0}_Tf_{T_f}.txt'
"""

    Temp =np.logspace(np.log10(T_0), np.log10(T_f), num=num_points) #Temperatures in which the g* is calculated.

    T_nu_f = [] #Value of the temperature of the neutrinos for different Tgamma
    g_star = []

    for T in Temp:
        sol = root_scalar(entropy, bracket=[T_f/10, T_0*2], method='brentq', args=(T)) #Finds the value of the temperature of the neutrinos that makes entropy (s0-sf) zero

        g_star.append(30/np.pi**2/T**
                      4*((defin.rho_nu(sol.root) + defin.rho_e(T))+ defin.rho_gamma(T))) #rho_T = g*T^4pi^2/30
        T_nu_f.append(sol.root)
    

    np.savetxt(rf'Final_file\g_star_T_neutrino_{num_points}_T0_{T_0}_Tf_{T_f}.txt', np.column_stack((Temp, T_nu_f, g_star)), 
        header="T_gamma(MeV)    T_nu(MeV)   gstar", 
        fmt="%.6e")

# %% 
if __name__ == "__main__":
    num_points = 1000
    T_0 = 25
    T_f = 0.01
    Temp =np.logspace(np.log10(T_0), np.log10(T_f), num=num_points) #Temperatures in which the g* is calculated.

    T_nu_f = [] #Value of the temperature of the neutrinos for different Tgamma
    g_star = []

    for T in Temp:
        sol = root_scalar(entropy, bracket=[T_f/10, T_0*2], method='brentq', args=(T)) #Finds the value of the temperature of the neutrinos that makes entropy (s0-sf) zero

        g_star.append(30/np.pi**2/T**
                      4*((defin.rho_nu(sol.root) + defin.rho_e(T))+ defin.rho_gamma(T))) #rho_T = g*T^4pi^2/30
        T_nu_f.append(sol.root)
    

    # Graphic Evolution of $g^*$ near $e^+e^-$ annihilation
    plt.plot(Temp,g_star)
    plt.title(r'Evolution of $g^*$ near $e^+e^-$ annihilation')
    plt.ylabel(r'$g^*$')
    plt.xlabel('T (MeV)')
    plt.gca().invert_xaxis()  
    plt.xscale('log')
    plt.show()

    # Graphic Evolution of $T_{\nu}$ near $e^+e^-$ annihilation
    T_aux = []#For low temperatures T_nu --> T_gamma*(4/11)**(1/3). We create T_aux to observe if the relation is being followed
    for T in Temp:
        T_aux.append(T*(4/11)**(1/3))
    plt.plot(Temp,T_nu_f, label = r'Real $T_{\nu}$ evolution')
    plt.plot(Temp,T_aux, linestyle = '--', label = r'$T_{\nu} = (4/11)^{1/3}T$')
    plt.title(r'Evolution of $T_{\nu}$ near $e^+e^-$ annihilation')
    plt.ylabel(r'$T_{\nu}$ (MeV)')
    plt.xlabel('T (MeV)')
    plt.gca().invert_xaxis()  
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()



# %%
