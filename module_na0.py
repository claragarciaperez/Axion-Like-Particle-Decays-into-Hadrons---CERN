#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import odeint
from scipy.optimize import root_scalar
import sys
import os
module_path = os.path.join(os.getcwd(), "Final_file")
if module_path not in sys.path:
    sys.path.append(module_path)
import module_definitions as defin
import module_Tnu as TNUEM
import pandas as pd
#%%
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G_F = 1.16638e-11 
G =1/M_pl**2
m_n = 939.57 #Neutron mass in MeV
m_p = 938.27 #Proton mass in MeV
m_e = 0.511 #Electron mass in MeV

mpi = 139.57 #MeV pion mass
mk = 493.67 #MeV kaon mass

alpha = 1/137
e = np.sqrt(4*np.pi*alpha)

zeta3   = 1.202056903159594  # ζ(3)
eta0    = 6.1e-10            # baryon-to-photon ratio today

MeV_to_m = 5.067730717e12    # 1 m = 5.0677e12 MeV^-1

#Upload the file to get data of g*,s in T

#%%
class initial_Abundance_ALP:
    """
        Class to calculate the initial abundance of alps

        Parameters
        ----------
            T_ini: float
                Initial temperature in which the ALPs start to appear. (MeV)
            T_0: float
                Initial decaying temperature (end of integration). (MeV)
            m: float
                Mass of the ALP (MeV)
            gstar_interp:
                Interpolator g*(T) (log-log interpolator with T in MeV)
            entropy_interp: 
                Interpolator s(T)/(T^3)
            prop_charged_interp: 
                Inerpolator g*_q(T)(log-log interpolator with T in MeV)
            Temperature_interp: 
                Interpolator T(t) (log-log interpolator with T in MeV and t in MeV^-1)
        """
    def __init__(self,T_ini, T_0,m,gstar_interp,entropy_interp,prop_charged_interp, Temperature_interp ):

        Temp_aux = np.logspace(np.log10(T_ini),np.log10(T_0),1000)
        n_axionseq = [defin.n_axion_eq(T,m) for T in Temp_aux]
        self.n_axionseq_interp = interp1d(Temp_aux,n_axionseq,kind='cubic',fill_value="extrapolate")
        self.entropy_interp = entropy_interp
        self.prop_charged_interp = prop_charged_interp
        self.Temperature_interp  = Temperature_interp 
        self.gstar_interp = gstar_interp

    def g_value(self,m,tau_a):
        """
        Calculates the axion-photon coupling. g = np.sqrt(64*np.pi/m**3/tau_a)

        Parameters
        ----------
            tau_a: float
                Lifetime of the ALP in MeV^-1
            m: float
                Mass of the ALP (MeV)
        """
        return np.sqrt(64*np.pi/m**3/tau_a)


    def Gamma_q(self,g,T,gq,m):
        """
        Calculates the axion production rate via scattering of a charged particle q + gamma <--> q + a
        Parameters
        ----------
            g: float
                Axion photon coupling in MeV^-2
            T: float
                Temperature (MeV)
            gq: float
                Effective number of relativistic charged particles
            m: float
                mass of the ALP (MeV)
        """
        m_gamma = (e*T*np.sqrt(gq))/6 #Cuidado si 3 o 6
        #return ((alpha*g**2*gq*T**3)/36)*(np.log(T**2/m_gamma**2) + 0.82)*np.exp(-m/T)
        return ((alpha*g**2*gq*T**3)/36)*(np.log(T**2/m_gamma**2) + 0.82)

    def dYa_dt(self,Ya,t,m,tau_a):
        """
        Differential equation for calcualting Ya(T). dYa/dt = Gamma_q(Y_aeq - Ya)

        Parameters
        ----------
            Ya: float
                na/s, being na number density of ALPS and s entropy density.
            t: float
                Time (MeV^-1)
            m: float
                Mass of the ALP (MeV)
            tau_a: float
                Lifetime of the ALP (MeV)
        """
        T = np.exp(self.Temperature_interp(np.log(t)))
        g = self.g_value(m,tau_a)
        #gq = np.exp(self.gstar_interp(np.log(T)))*np.exp(self.prop_charged_interp(np.log(T)))
        gq = np.exp(self.prop_charged_interp(np.log(T)))
        s =(self.entropy_interp(T))*T**3
        G_q  = self.Gamma_q(g,T,gq,m)
        #nq =(zeta3/np.pi**2)*gq*T**3
        #Yq = nq/s
        #return G_q*Yq*(1-Ya/(self.n_axionseq_interp(T)/s))
        return G_q*((self.n_axionseq_interp(T)/s) - Ya)

# %%
def solve_dYa_dt (m, taua,Tini, T_0, times,entropy_interp,prop_charged_interp,gstar_interp, Temperature_interp, Ya_0 = 0):
    """
        Calculates abundance of ALPs at a temperature T0.

        Parameters
        ----------
            m: float
                Mass of the ALP (MeV)
            taua: float
                Lifetime of the ALP (MeV)
            Tini: float
                Initial integrarion temperature or Reheating temperature (MeV)
            times: float
                List with times to integrate in MeV^-1
            entropy_interp: 
                Interpolator s(T)/(T^3)
            prop_charged_interp: 
                Inerpolator g*_q(T)(log-log interpolator with T in MeV)
            gstar_interp:
                Interpolator g*(T) (log-log interpolator with T in MeV)
            Temperature_interp: 
                Interpolator T(t) (log-log interpolator with T in MeV and t in MeV^-1)
            Y_a0: float
                Ya(Tini). Default = 0

        Return
        ----------
            rho_a(T_0)/rho_SM: float
                Proportion of energy density at T_0 of ALPs
            
    """
    
    ini_ALP = initial_Abundance_ALP(Tini, T_0,m,gstar_interp,entropy_interp,prop_charged_interp, Temperature_interp)

    sol =odeint(ini_ALP.dYa_dt, y0 =[Ya_0], t = times, args = (m,taua/h_bar),rtol = 1e-8, atol = 1e-10)

    return (sol[:,0][-1]*entropy_interp(T_0)*T_0**3*m/(defin.rho_SM(T_0,np.exp(gstar_interp(np.log(T_0))))))


