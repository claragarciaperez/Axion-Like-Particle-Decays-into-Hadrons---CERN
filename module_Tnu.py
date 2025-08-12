#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import numdifftools as nd
from scipy.interpolate import interp1d
import sys
import os

module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final_file")
if module_path not in sys.path:
    sys.path.append(module_path)

import module_definitions as defin
#%% Constants
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G =1/M_pl**2
G_F = 1.16638e-11

#%% Interpolations of rho_e, P_e, P_nu, P_gamma, d_rho_e_dT from T0 = 25 MeV to Tf = 0.01 MeV
T_vals = np.logspace(np.log10(25),np.log10(0.01), 1000)
rho_e_vals = np.array([ defin.rho_e(T) for T in T_vals])
rho_e_interp = interp1d(T_vals, rho_e_vals, kind='cubic', fill_value="extrapolate")
P_nu_vals = np.array([defin.P_nu(T) for T in T_vals])
P_nu_interp = interp1d(T_vals, P_nu_vals, kind='cubic', fill_value="extrapolate")
P_gamma_vals = np.array([defin.P_gamma(T) for T in T_vals])
P_gamma_interp = interp1d(T_vals, P_gamma_vals, kind='cubic', fill_value="extrapolate")
P_e_vals = np.array([defin.P_e(T) for T in T_vals])
P_e_interp = interp1d(T_vals, P_e_vals, kind='cubic', fill_value="extrapolate")
d_rho_e_dT_vals = np.array([nd.Derivative(rho_e_interp, step = 1e-5)(T) for T in T_vals])
d_rho_e_dT_interp = interp1d(T_vals, d_rho_e_dT_vals, kind='cubic', fill_value='extrapolate')
    
#%%
def delta_rho_nu (T_gamma, T_nu):
    """
        Exchange of energy density/t between plasma and neutrinos due to neutron decoupling.

        G_F**2*T_gamma**9*(T_gamma - T_nu)/T_gamma

        Parameters
        ----------
            T_gamma:float
                Temperature in MeV of the plasma
            T_nu:float
                Temperature in MeV of the neutrinos

        Returns
        -------
            delta_rho_nu: float
        """
    return G_F**2*T_gamma**9*(T_gamma - T_nu)/T_gamma


def ec_dif_Tnu(vars,t ,tau_a):
    """
        System of differential equations of the evolution of rhoa, Tgamma, Tnu, aT

        Parameters
        ----------
            vars:[float, float, float, float]
                rho_a, Tgammae, Tnu, aT in units of MeV
            tau_a: float
                Lifetime of the ALP in MeV

        Returns
        -------
            diff equations: [float, float, float, float]
                drho_a_dt , (dT_gamma_dt), (dT_nu_dt), daT_dt
        """
    rho_a, Tgammae, Tnu, aT= vars 

    #Calculations of energy densities and pressure of plasma and neutrinos
    rg = defin.rho_gamma(Tgammae) + rho_e_interp(Tgammae)
    rnu = defin.rho_nu(Tnu)
    Pgammae = P_gamma_interp(Tgammae) + P_e_interp(Tgammae) 
    Pnu = P_nu_interp(Tnu)

    H = np.sqrt(8*np.pi*G/3*(rho_a + rg +rnu))

    dnu_dT = 4*rnu/Tnu
    drg_dT = 4*defin.rho_gamma(Tgammae)/Tgammae + d_rho_e_dT_interp(Tgammae)

    drho_a_dt = - rho_a/tau_a - 3*H*rho_a
    dT_gamma_dt = ((rho_a/tau_a - 3*(rg+ Pgammae)*H - delta_rho_nu(Tgammae, Tnu)))/drg_dT
    dT_nu_dt =  (-3*H*(rnu + Pnu) + delta_rho_nu(Tgammae, Tnu))/dnu_dT
    daT_dt = aT*H + aT/Tgammae*dT_gamma_dt 
    return [drho_a_dt , (dT_gamma_dt), (dT_nu_dt), daT_dt]

def solve_TEM_Tnu (t_list,T_0, g_star, p, tau_a):
    """
        Calculations of TEM and Tnu in the presence of ALPS

        Parameters
        ----------
            t_list: float
                list of times in MeV
            T_0: float
                Initial temperature in MeV
            g_star: float
                g*(T_0)
            p: float
                rho_a(0) = p*rho_SM(0)
            tau_a: float
                Lifetime of the ALP in seconds

        Returns
        -------
            sol:
                Solution of the differential equation with odeint. 
        """
    rho_a0 = defin.rho_SM(T_0, g_star)*p
    cond_ini = [rho_a0,T_0, T_0,T_0]
    sol = odeint(ec_dif_Tnu, cond_ini, t_list, args=(tau_a/h_bar,), rtol = 1e-8, atol = 1e-10)
    return sol

#%%

if __name__ == "__main__":
    T_0 = 20
    T_f = 0.01
    Temp = np.logspace(np.log10(T_0), np.log10(T_f), num=1000)

    data_gs = np.loadtxt(rf'g_star_T_neutrino_1000.txt', skiprows=1)
    gstar = data_gs[:,2]
    T_nudata = data_gs[:,1]
    T_data = data_gs[:,0]
    gstar_interpolate = interp1d(T_data, gstar, kind='cubic', fill_value="extrapolate")

    cond_ini = [0,T_0, T_0,T_0]

    t_list = []
    for i in Temp:
        t_list.append(defin.time_RD(i,gstar_interpolate(i)))

    time_seconds = []
    for t in t_list:
        time_seconds.append(t*h_bar)

    sol = solve_TEM_Tnu(t_list, T_0, gstar_interpolate(T_0),0,1)

    T_nu_0_teo_aux = []
        
    for i in  sol[:,1]:
        T_nu_0_teo_aux.append(i*(4/11)**(1/3))

    plt.title(r'Evolution of $T_{\gamma/e}$ and $T_{\nu}$ over time.')
    plt.plot(time_seconds, sol[:,1], label =r'$T_{\gamma/e}$')
    plt.plot(time_seconds, sol[:,2], label = r'$T_{\nu}$')
    plt.plot(time_seconds, T_nu_0_teo_aux, linestyle = '--', label =r'$T_{\gamma/e}(\frac{4}{11})^{1/3}$')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.xlabel('time (s)')
    plt.ylabel('T (MeV)')



# %%

