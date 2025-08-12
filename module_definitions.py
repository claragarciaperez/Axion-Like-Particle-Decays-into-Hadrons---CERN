#%% 
from scipy.integrate import quad
import numpy as np

#%% Constants
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G_F = 1.16638e-11
m_n = 939.57 #Neutron mass in MeV
m_p = 938.27 #Proton mass in MeV
m_e = 0.511 #Electron mass in MeV
tau_n = 880.2 #Lifetime of neutrons in seconds
delta_m = 1.293 #mn - mp in MeC
A = 1/tau_n/0.0157527/delta_m**5
mpi = 139.57 #MeV pion mass
BR_pi = 0.1 #Branching ratio
Gamma_pi = 3.8e7 #s^-1

zeta3   = 1.202056903159594  # ζ(3)
eta0    = 6.1e-10            # baryon-to-photon ratio todat


#%%
 
def rho_fermion(x, g, m, T):
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3 * E / (np.exp(E/T) + 1) * x**2

def rho_boson(x, g, m, T):
    E = np.sqrt(x**2 + m**2)
    
    return 4*np.pi*g/(2*np.pi)**3 *E / (np.exp(E/T) - 1) * x**2

def P_fermion(x, g, m, T):
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3 / (np.exp(E/T) + 1) * x**4 / (3*E)

def P_boson(x, g, m, T):
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3/ (np.exp(E/T) - 1) * x**4 / (3*E)


def rho_nu(T_nu): 
    return 3*7/8*np.pi**2/15*T_nu**4

def P_nu(T_nu): 
    g_nu = 6
    m_nu = 0 #It is approximate that the neutrino mass is zero
    return quad(P_fermion, 0, np.inf,  args=(g_nu, m_nu, T_nu))[0]

def rho_gamma(T):
    return np.pi**2/15*T**4

def P_gamma(T):
    g_gamma = 2
    m_gamma = 0
    return quad(P_boson, 0, np.inf,  args=(g_gamma, m_gamma, T))[0]

def rho_e (T):
    g_e = 4
    m_e = 0.511 #MeV
    return quad(rho_fermion,  0, np.inf, args=(g_e, m_e, T))[0]

def P_e (T):
    g_e = 4
    m_e = 0.511 #MeV
    return quad(P_fermion,  0, np.inf, args=(g_e, m_e, T))[0]


def rhoa_RD(t, rho_0, t0, tau_a):
    return rho_0*(t/t0)**(-3/2)*np.exp(-(t-t0)/(tau_a))

def rho_SM(T, g_star): #rho_SM in eq
    return (g_star*np.pi**2/30*T**4)

def time_RD(T, g_star): #In rad domination
    return(M_pl/2/np.sqrt(g_star)/1.66/T/T)
# %%
def lambda_1a (k_nu,T, T_nu):
    E_e = k_nu + delta_m
    k_e = np.sqrt(E_e**2-m_e**2)
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e*E_e*f_nu*(1-f_e)

def lambda_1b (k_e,T, T_nu):
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu = E_e + delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*(1-f_nu)

def lambda_1c (k_e, T, T_nu):
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu = delta_m - E_e
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*(1-f_nu)*(1-f_e)

def lambda_3a (k_e, T, T_nu):
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu =  E_e - delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*(1-f_nu)

def lambda_3b(k_nu, T, T_nu):
    E_e = k_nu - delta_m
    k_e = np.sqrt(E_e**2-m_e**2)
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e*E_e*f_nu*(1-f_e)

def lambda_3c (k_e, T, T_nu):
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu =  -E_e + delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*f_nu