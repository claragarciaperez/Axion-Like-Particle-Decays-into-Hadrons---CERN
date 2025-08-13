#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.interpolate import interp1d
from scipy.integrate import odeint
import sys
import os
module_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "Final_file")
if module_path not in sys.path:
    sys.path.append(module_path)
import module_definitions as defin
import module_Tnu as TNUEM
#%%
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G_F = 1.16638e-11 
G =1/M_pl**2
m_n = 939.57 #Neutron mass in MeV
m_p = 938.27 #Proton mass in MeV
m_e = 0.511 #Electron mass in MeV
tau_n = 880.2 #Lifetime of the neutron in seconds
delta_m = 1.293 #mn-mp in MeV
A = 1/tau_n/0.0157527/delta_m**5
mpi = 139.57 #MeV pion mass
Gamma_pi = 3.8e7 #Decay rate of pions s^-1

zeta3   = 1.202056903159594  # ζ(3)
eta0    = 6.1e-10            # baryon-to-photon ratio today

MeV_to_m = 5.067730717e12    # 1 m = 5.0677e12 MeV^-1


#Upload the file to get data of g* in T
data_gs = np.loadtxt(rf'g_star_T_neutrino_1000_T0_25_Tf_0.01.txt', skiprows=1)
gstar = data_gs[:,2]
T_data = data_gs[:,0]

gsinterp= interp1d(T_data, gstar, kind='cubic', fill_value="extrapolate")

#%%
class deriv_n:
    """
        Class to calculate the evolution of Xn with temperature in presence of ALPS that can decay in pions

        Parameters
        ----------
            TEM: float
                List of temperature of the plasma in MeV
            Tnu: float
                List of temperature of neutrinos in MeV
            aT: float
                List of values of a*T for the different TEM
            time_T: float
                List of times in secondes for each value of temperature
            T_0: float
                Initial temperature in MeV
            T_f: float
                Final temperature in MeV
        """

    def __init__(self, TEM, Tnu, aT, time_T,T_0, T_f):
        self.Tnu_interpolate = interp1d(TEM, Tnu, kind='cubic', fill_value="extrapolate")
        self.time_interpolate = interp1d(TEM, time_T, kind='cubic', fill_value="extrapolate")
        self.aT_interpolate = interp1d(TEM, aT, kind='cubic', fill_value="extrapolate")

        T_vals = np.logspace(np.log10(T_0), np.log10(T_f), num=500)
        lambda_1a_vals = np.array([quad(defin.lambda_1a, 0, np.inf,  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l1a_interp = interp1d(T_vals, lambda_1a_vals, kind='cubic', fill_value="extrapolate")
        lambda_1b_vals = np.array([quad(defin.lambda_1b, 0, np.inf,  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l1b_interp = interp1d(T_vals, lambda_1b_vals, kind='cubic', fill_value="extrapolate")
        lambda_1c_vals = np.array([quad(defin.lambda_1c, 0, np.sqrt(delta_m**2 - m_e**2),  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l1c_interp = interp1d(T_vals, lambda_1c_vals, kind='cubic', fill_value="extrapolate")
        lambda_3a_vals = np.array([quad(defin.lambda_3a, np.sqrt(delta_m**2 - m_e**2), np.inf,  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l3a_interp = interp1d(T_vals, lambda_3a_vals, kind='cubic', fill_value="extrapolate")
        lambda_3b_vals = np.array([quad(defin.lambda_3b, delta_m + m_e, np.inf,  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l3b_interp = interp1d(T_vals, lambda_3b_vals, kind='cubic', fill_value="extrapolate")
        lambda_3c_vals = np.array([quad(defin.lambda_3c, 0, np.sqrt(delta_m**2 - m_e**2),  args=(T, self.Tnu_interpolate(T)))[0] for T in T_vals])
        self.l3c_interp = interp1d(T_vals, lambda_3c_vals, kind='cubic', fill_value="extrapolate")
        dt_dT = np.gradient(time_T, TEM)
        self.dt_dT_interp = interp1d(TEM, dt_dT, kind='cubic', fill_value="extrapolate")
        rho_e_vals = np.array([ defin.rho_e(T) for T in T_vals])
        self.rho_e_interp = interp1d(T_vals, rho_e_vals, kind='cubic', fill_value="extrapolate")

    def deriv_Xn(self, Xn,T):
        """
    Differential equation of the evolution of Xn with temperature without the effect of hadrons
    dXn/dT = (lpn(1-Xn) - lnpXn)dt/dT

    """
        l_pn = self.l3a_interp(T) + self.l3b_interp(T) + self.l3c_interp(T)
        l_np = self.l1a_interp(T) + self.l1b_interp(T) + self.l1c_interp(T)
        dXn_dt = l_pn*(1-Xn) - l_np*Xn
        der_t_dT = self.dt_dT_interp(T)
        dXn_dT = dXn_dt*der_t_dT
        return dXn_dT

    def n_b(self,T):
        """
        Baryon number density.
        n_b = eta0*ngamma*(a(Tf)Tf/aT)^3

        Creates a file with the values of T_gamma, T_nu and g*.

        Parameters
        ----------
            T:float
                Temperature in MeV

        Returns
        -------
            n_b:float
                Baryon numer density in m^-3
        """
        n_gamma = 2 * zeta3 / np.pi**2 * T**3
        return eta0 * n_gamma *(self.aT_interpolate(0.02)/ self.aT_interpolate(T))**3*(MeV_to_m**3)#Ultimo termino cambiar de Mev^-3 a m^-3

    def na(self,t, n0, t0, tau_a):
        """
        number density of alps in radiation domination
        na = n0(t/t0)^(-3/2)*exp(-(t/t0)/taua)

        Parameters
        ----------
            t: float
                Time in which we want to evaluate na(t) in seconds
            n0: float
                Initial number density in m^-3
            t0: float
                Initial time in seconds
            tau_a: float
                Lifetime of the alp

        Returns
        -------
            n_b: float
                Baryon numer density in m^-3
        """
        return n0*(t/t0)**(-3/2)*np.exp(-(t-t0)/(tau_a))
    
    def Fcpi (self,T):
        """
        Sommerfeld enhancement. Fcpi = x/(1 - np.exp(-x)) with x = 2*np.pi*alpha_EM/ve.

        Parameters
        ----------
            T: float
                Temperature in MeV
        Returns
        -------
            Fcpi: float
                Sommerfeld enhancement
                
        """
    
        alpha_EM= 1/137
        ve = np.sqrt(T/mpi) +np.sqrt(T/m_p)
        x = 2*np.pi*alpha_EM/ve
        return x/(1 - np.exp(-x))

    def deriv_Xn_pi(self,Xn, T, t0, na0,tau_a, BR_pi):
        """
        Differential equation for the evolution of Xn taking in consideration the pions and with radiation domination
        dXn_dT =((l_pn + n_pimin*cspimin)*(1-Xn) - (l_np + n_piplus*cspiplus)*Xn)(dt/dT)

        n_pimin = na*BR_pi/(tau_a*(Gamma_pi + cspimin*(1-Xn)*nB))
        n_piplus = na*BR_pi/(tau_a*(Gamma_pi + cspiplus*(Xn)*nB))

        Parameters
        ----------
            Xn: float
                nn/(np + nn)
            T: float
                Temperature in MeV
            t0: float
                Initial time in seconds
            na0: float
                Initial number density of alps
            tau_a: float
                Lifetime of the alp in seconds
            Br_pi: float
                Branching ratio of ALPS in pions
        Returns
        -------
            dXn/dT: float
                
        """
        
        l_pn = self.l3a_interp(T) + self.l3b_interp(T) + self.l3c_interp(T) #s
        l_np = self.l1a_interp(T) + self.l1b_interp(T) + self.l1c_interp(T) #s
        cspimin= 4.3e-23*self.Fcpi(T)#m^3/s
        cspiplus=4.3e-23/0.9 #m^3/s
        nB = self.n_b(T)
        n_pimin = self.na(self.time_interpolate(T),na0,t0,tau_a)*BR_pi/(tau_a*(Gamma_pi + cspimin*(1-Xn)*nB))
        n_piplus = self.na(self.time_interpolate(T),na0,t0,tau_a)*BR_pi/(tau_a*(Gamma_pi + cspiplus*(Xn)*nB))
        dXn_dt = (l_pn + n_pimin*cspimin)*(1-Xn) - (l_np + n_piplus*cspiplus)*Xn   
        der_t_dT = self.dt_dT_interp(T)
        dXn_dT = dXn_dt*der_t_dT
        return (dXn_dT)
    
    def deriv_Xn_pi_noRD(self,vars, T,tau_a, BR_pi, ma):
        """
        Differential equation for the evolution of Xn taking in consideration the pions and without assuming radiation domination
        dXn_dT =((l_pn + n_pimin*cspimin)*(1-Xn) - (l_np + n_piplus*cspiplus)*Xn)(dt/dT)
        dna_dT = (-na/tau_a - 3*na*H)*(dt/dT)

        n_pimin = na*BR_pi/(tau_a*(Gamma_pi + cspimin*(1-Xn)*nB))
        n_piplus = na*BR_pi/(tau_a*(Gamma_pi + cspiplus*(Xn)*nB))

        Parameters
        ----------
            vars: [float, float]
                [Xn, na]
            T: float
                Temperature in MeV
            tau_a: float
                Lifetime of the alp in seconds
            Br_pi: float
                Branching ratio of ALPS in pions
            ma: float
                Mass of the ALP in MeV
        Returns
        -------
            dXn/dT: float
                Derivative of the evolution of Xn with temperature
            dna/dT: float
                Derivative of the evolution of number density of ALPS with temperature

                
        """
        Xn, na = vars
        l_pn = self.l3a_interp(T) + self.l3b_interp(T) + self.l3c_interp(T) #s
        l_np = self.l1a_interp(T) + self.l1b_interp(T) + self.l1c_interp(T) #s
        cspimin= 4.3e-23*self.Fcpi(T)#m^3/s
        #cspimin= 4.3e-23
        cspiplus=4.3e-23/0.9 #m^3/s
        nB = self.n_b(T)
        n_pimin =na*BR_pi/(tau_a*(Gamma_pi + cspimin*(1-Xn)*nB))
        n_piplus = na*BR_pi/(tau_a*(Gamma_pi + cspiplus*(Xn)*nB))
        dXn_dt = (l_pn + n_pimin*cspimin)*(1-Xn) - (l_np + n_piplus*cspiplus)*Xn   
        H = np.sqrt(8*np.pi*G/3*(na*ma/(MeV_to_m**3)+ defin.rho_gamma(T) +defin.rho_nu(self.Tnu_interpolate(T)) + self.rho_e_interp(T)))/h_bar
        dna_dt = -na/tau_a - 3*na*H
        der_t_dT = self.dt_dT_interp(T)
        dXn_dT = dXn_dt*der_t_dT
        dna_dt = dna_dt*der_t_dT
        return (dXn_dT, dna_dt)
    
#%%
def He_neff(T_0, T_f, t_list,time_sec, p, tau_a, ma, Br, neff = True, Ya = True, Yapi =True ):
    """
        Calculation of Ye and Neff in the presence of ALPS

        Parameters
        ----------
            T_0: float
                Initial temperature un MeV
            T_f: float
                Final temperature un MeV
            t_list: float
                List of times to evaluate in MeV
            time_sec: float
                List of times to evaluate in MeV
            p: float
                rhoa_0 = p*rho_SM
            tau_a: float
                Lifetime of the alp in seconds
            ma: float
                ALP mass
            Br: float
                Branching ratio of ALPS in pions
            neff: bool
                If True the value of Neff will be given as a return. Default = True
            Ya: bool
                It True (Xna-XnSM)/XnSM (no pions considered) will be given as a return. Default = True
            Yapi: bool
                It True (Xn(a+pi)-XnSM)/XnSM (pions are considered) will be given as a return. Default = True

        Returns
        -------
            Ye_pi: float
                Proportion of He abundance at the BBN considering pion decay
            Ye_pi: float
                Proportion of He abundance at the BBN without considering pion decay
            Neff: float
                3*(11/4)^(4/3)*(Tnu/Tgamma)^4
                
        """
    result = []
    sol = TNUEM.solve_TEM_Tnu(t_list, T_0,gsinterp(T_0),p,tau_a)
    Tgamma = sol[:,1]
    Tnu = sol[:,2]
    aT = sol[:,3]

    obj = deriv_n(Tgamma,Tnu, aT, time_sec, T_0, T_f)

    na0 = p*defin.rho_SM(T_0,gsinterp(T_0) )/(ma)*(MeV_to_m**3)
    ini_cond = [1/(1+np.exp(delta_m/T_0)), na0]

    if Yapi:
        sol_Xn = odeint(obj.deriv_Xn_pi_noRD,ini_cond, Tgamma, args=(tau_a, Br,ma), rtol = 1e-8, atol = 1e-10)
        Xn_interp = interp1d(Tgamma, sol_Xn[:,0], kind='cubic', fill_value="extrapolate")
        result.append(Xn_interp(0.078)*2)

    if Ya:
        sol_Xn2 =odeint(obj.deriv_Xn, [1/(1+np.exp(delta_m/T_0))], Tgamma, rtol = 1e-8, atol = 1e-10)
        Xn_nopi_interp = interp1d(Tgamma, sol_Xn2[:,0], kind='cubic', fill_value="extrapolate")
        result.append(Xn_nopi_interp(0.078)*2)
    if neff:
        Neff = 3*(11/4)**(4/3)*(Tnu[-1]/Tgamma[-1])**4
        result.append(Neff)

    #sol_Xn3 =odeint(obj.deriv_Xn_pi, [1/(1+np.exp(delta_m/T_0))], Tgamma,args=(defin.time_RD(T_0, gsinterp(T_0))*h_bar,na0,tau_a, Br), rtol = 1e-8, atol = 1e-10)

    #plt.plot(Temperature_values,sol_Xn[:,0])
    #plt.plot(Temperature_values,sol_Xn2[:,0])
    #plt.gca().invert_xaxis()    #In order to go from higher to lower temperature
    #plt.xlabel('T(MeV)')
    #plt.ylabel(fr'$X_n$')
    #plt.legend()
    #plt.xscale('log')
    #plt.yscale('log')
    #plt.xlim(20,0.07)
    #plt.ylim(0.01,1)

    #F = defin.F(p*defin.rho_SM(T_0,gsinterp(T_0)),t_list[0],tau_a/h_bar)

    return result

if __name__ == "__main__":
    T_0 = 20
    T_f = 0.01
    p =0.0
    tau_a = 0.03
    ma = 400
    Br= 0.3
    Temperature_values = np.logspace(np.log10(T_0), np.log10(T_f), num=1000)

    t_list = []
    for i in Temperature_values:
        t_list.append(defin.time_RD(i,gsinterp(i)))
    time_seconds = []
    for t in t_list:
        time_seconds.append(t*h_bar)

    print(He_neff(T_0,T_f,t_list, time_seconds, p,tau_a,ma,Br))

