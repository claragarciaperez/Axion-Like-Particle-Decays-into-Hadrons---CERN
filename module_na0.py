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
    def __init__(self,T_ini, T_0,m,gstar_interp,entropy_interp,prop_charged_interp, Temperature_interp ):
        Temp_aux = np.logspace(np.log10(T_ini),np.log10(T_0),1000)
        n_axionseq = [defin.n_axion_eq(T,m) for T in Temp_aux]
        self.n_axionseq_interp = interp1d(Temp_aux,n_axionseq,kind='linear',fill_value="extrapolate")
        self.entropy_interp = entropy_interp
        self.prop_charged_interp = prop_charged_interp
        self.Temperature_interp  = Temperature_interp 
        self.gstar_interp = gstar_interp

    def g_value(self,m,tau_a):
        return np.sqrt(64*np.pi/m**3/tau_a)


    def Gamma_q(self,g,T,gq,m):
        m_gamma = (e*T*np.sqrt(gq))/3
        return ((alpha*g**2*gq*T**3)/36)*(np.log(T**2/m_gamma**2) + 0.82)*np.exp(-m/T)

    def dYa_dt(self,Ya,t,m,tau_a):
        T = np.exp(self.Temperature_interp(np.log(t)))
        g = self.g_value(m,tau_a)
        gq = np.exp(self.gstar_interp(np.log(T)))*np.exp(self.prop_charged_interp(np.log(T)))
        s =(self.entropy_interp(T))*T**3
        G_q  = self.Gamma_q(g,T,gq,m)
        nq =(zeta3/np.pi**2)*gq*T**3
        Yq = nq/s
        return G_q*Yq*(1-Ya/(self.n_axionseq_interp(T)/s))

# %%
def solve_dYa_dt (m, taua,Tini, T_0, times,entropy_interp,prop_charged_interp,gstar_interp, Temperature_interp, Ya_0 = 0):
    ini_ALP = initial_Abundance_ALP(Tini, T_0,m,gstar_interp,entropy_interp,prop_charged_interp, Temperature_interp)

    sol =odeint(ini_ALP.dYa_dt, y0 =[Ya_0], t = times, args = (m,taua/h_bar),rtol = 1e-8, atol = 1e-10)

    return (sol[:,0][-1]*entropy_interp(T_0)*T_0**3*m/(defin.rho_SM(T_0,np.exp(gstar_interp(np.log(T_0))))))


#%%
if __name__ == "__main__":
    Tini = 1e15
    T_0 = 20
    Ya_0 = 0
    m = 290
    taua = 0.01
    
    gstar_interp, entropy_interp,prop_charged_interp,Temperature_interp = defin.interpolate_na0(Tini,T_0)

    Y_a0 = defin.n_axion_eq(Tini,m)/(entropy_interp(Tini)*Tini**3)


    times_fin = np.logspace(np.log10(defin.time_RD(Tini, np.exp(gstar_interp(np.log(Tini))))), np.log10(defin.time_RD(T_0, np.exp(gstar_interp(np.log(T_0))))),10000)

    ini_ALP = initial_Abundance_ALP(Tini, T_0,m,gstar_interp,entropy_interp,prop_charged_interp, Temperature_interp)

    sol =odeint(ini_ALP.dYa_dt, y0 =[Y_a0], t = times_fin, args = (m,taua/h_bar))

    print(sol[:,0][-1]*entropy_interp(T_0)*T_0**3*m/(defin.rho_SM(T_0,np.exp(gstar_interp(np.log(T_0))))))

    T = []
    for t in times_fin:
        T.append(np.exp(Temperature_interp(np.log(t))))
    plt.plot(T,sol)
    plt.xscale('log')
    plt.ylabel(r'$Y_a$')
    plt.xlabel(r'T($MeV$)')
    plt.gca().invert_xaxis()

# %%
