#%% 
from scipy.integrate import quad
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
import pandas as pd


#%% Constants
M_pl = 1.22e22 #MeV
h_bar = 6.582e-22 #MeV
G_F = 1.16638e-11
zeta3   = 1.202056903159594  # ζ(3)
eta0    = 6.1e-10            # baryon-to-photon ratio todat

m_n = 939.57 #Neutron mass in MeV
m_p = 938.27 #Proton mass in MeV
m_e = 0.511 #Electron mass in MeV
mpi = 139.57 #MeV pion mass
delta_m = 1.293 #mn - mp in MeV
m_mu = 105.66
m_tau = 1.777e3
mu = 3
md = 5
ms = 95
mc = 1.27e3
mb = 4.18e3
mt = 173e3
mW = 80.4e3

qu = 2/3
qd = -1/3
qs = -1/3
qc = 2/3
qb = -1/3
qt = 2/3


tau_n = 880.2 #Lifetime of neutrons in seconds

A = 1/tau_n/0.0157527/delta_m**5

BR_pi = 0.1 #Branching ratio
Gamma_pi = 3.8e7 #s^-1


#%%
 
def rho_fermion(x, g, m, T):
    """
    Energy density of a fermion

    Parameters
    ----------
        x:float
            Momentum p
        g: float
            Degrees of freedom
        m: float
            mass (MeV)
        T: float
            Temperature (MeV)
    """
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3 * E / (np.exp(E/T) + 1) * x**2

def rho_boson(x, g, m, T):
    """
    Energy density of a boson

    Parameters
    ----------
        x:float
            Momentum p
        g: float
            Degrees of freedom
        m: float
            mass (MeV)
        T: float
            Temperature (MeV)
    """
    E = np.sqrt(x**2 + m**2)
    
    return 4*np.pi*g/(2*np.pi)**3 *E / (np.exp(E/T) - 1) * x**2

def P_fermion(x, g, m, T):
    """
    Pressure of a fermion

    Parameters
    ----------
        x:float
            Momentum p
        g: int
            Degrees of freedom
        m: float
            mass (MeV)
        T: float
            Temperature (MeV)
    """
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3 / (np.exp(E/T) + 1) * x**4 / (3*E)

def P_boson(x, g, m, T):
    """
    Pressure of a fermion

    Parameters
    ----------
        x:float
            Momentum p
        g: int
            Degrees of freedom
        m: float
            mass (MeV)
        T: float
            Temperature (MeV)
    """
    E = np.sqrt(x**2 + m**2)
    return 4*np.pi*g/(2*np.pi)**3/ (np.exp(E/T) - 1) * x**4 / (3*E)


def n_fermion (x,m,g,Nc,T):
    """
    Number density of a fermion 

    Parameters
    ----------
        x:float
            p/T
        m: float
            mass (MeV)
        g: int
            Degrees of freedom
        Nc: int
            Color number
        T: float
            Temperature (MeV)
    """
    E = np.sqrt(x**2 + (m/T)**2)
    return Nc*g*x**2*T**3/(2*np.pi**2*(np.exp(E) + 1))
    #E = np.sqrt(p**2 + (m)**2)
    #return Nc*g*p**2/(2*np.pi**2*(np.exp(E/T) + 1))


def rho_nu(T_nu):
    """
    Energy density of neutrinos

    Parameters
    ----------
        T_nu: float
            Temperature (MeV)
    """ 
    return 3*7/8*np.pi**2/15*T_nu**4

def P_nu(T_nu): 
    """
    Pressure of neutrinos (integral)

    Parameters
    ----------
        T_nu: float
            Temperature (MeV)
    """ 
    g_nu = 6
    m_nu = 0 #It is approximate that the neutrino mass is zero
    return quad(P_fermion, 0, np.inf,  args=(g_nu, m_nu, T_nu))[0]

def rho_gamma(T):
    """
    Energy density of photons

    Parameters
    ----------
        T: float
            Temperature (MeV)
    """ 
    return np.pi**2/15*T**4

def P_gamma(T):
    """
    Pressure of photons (integral)

    Parameters
    ----------
        T: float
            Temperature (MeV)
    """     
    g_gamma = 2
    m_gamma = 0
    return quad(P_boson, 0, np.inf,  args=(g_gamma, m_gamma, T))[0]

def rho_e (T):
    """
    Energy density of electrons and positrons

    Parameters
    ----------
        T: float
            Temperature (MeV)
    """ 
    g_e = 4
    m_e = 0.511 #MeV
    return quad(rho_fermion,  0, np.inf, args=(g_e, m_e, T))[0]

def P_e (T):
    """
    Pressure of electrons and positrons

    Parameters
    ----------
        T: float
            Temperature (MeV)
    """ 
    g_e = 4
    m_e = 0.511 #MeV
    return quad(P_fermion,  0, np.inf, args=(g_e, m_e, T))[0]


def rhoa_RD(t, rho_0, t0, tau_a):
    """
    Energy density of a non relativistic particle decaying under assumption of radiation domination

    Parameters
    ----------
        t: float
            time
        rho_0: float
            Initial energy density
        t0: float
            Initial time
        tau_a: float
            Lifetime of the particle
            
    """ 
    return rho_0*(t/t0)**(-3/2)*np.exp(-(t-t0)/(tau_a))

def rho_SM(T, g_star): 
    """
    Energy density in the SM in equilibrium

    Parameters
    ----------
        T: float
            Temperature (MeV)
        g_star: float
            Effective relativistic degrees of freedom at temperature T
    """ 
    return (g_star*np.pi**2/30*T**4)

def time_RD(T, g_star):
    """
    Conversion from temperature to time (MeV^-1) in Radiation domination

    Parameters
    ----------
        T: float
            Temperature (MeV)
    """ 
    return(M_pl/2/np.sqrt(g_star)/1.66/T/T)
# %%
def lambda_1a (k_nu,T, T_nu):
    """
    Rate n + nu_e --> p + e- (s^-1)

    Parameters
    ----------
        k_nu: float
            Momentum of nu
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = k_nu + delta_m
    k_e = np.sqrt(E_e**2-m_e**2)
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e*E_e*f_nu*(1-f_e)

def lambda_1b (k_e,T, T_nu):
    """
    Rate n + e+ --> p + nue (s^-1)

    Parameters
    ----------
        k_e: float
            Momentum of e
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu = E_e + delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*(1-f_nu)

def lambda_1c (k_e, T, T_nu):
    """
    Rate n --> p + e- + nue (s^-1)

    Parameters
    ----------
        k_e: float
            Momentum of e
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu = delta_m - E_e
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*(1-f_nu)*(1-f_e)

def lambda_3a (k_e, T, T_nu):
    """
    Rate p + e- --> n + nue (s^-1)

    Parameters
    ----------
        k_e: float
            Momentum of e
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu =  E_e - delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*(1-f_nu)

def lambda_3b(k_nu, T, T_nu):
    """
    Rate p + nu_e --> p + e+ (s^-1)

    Parameters
    ----------
        k_nu: float
            Momentum of nu
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = k_nu - delta_m
    k_e = np.sqrt(E_e**2-m_e**2)
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e*E_e*f_nu*(1-f_e)

def lambda_3c (k_e, T, T_nu):
    """
    Rate p + e- + nue --> n (s^-1)

    Parameters
    ----------
        k_e: float
            Momentum of e
        T: float
            Temperature plasma (MeV)
        T_nu:float
            Temperature neutrinos (MeV)
    """ 
    E_e = np.sqrt(k_e**2 + m_e**2)
    k_nu =  -E_e + delta_m
    f_nu = 1/(np.exp(k_nu/T_nu) + 1)
    f_e = 1/(np.exp(E_e/T) + 1)
    return A*k_nu**2*k_e**2*f_e*f_nu

#%%
def open_rates(filename):
    """
    Opens the file with the rates of ma Gamma_phot, Gamma_a

    Parameters
    ----------
        filename: string
            Name of the file with the rates
    
    Return
    ---------
        m_a: [float]
            List with masses (GeV)
        Gamma_phot: [float]
            List with the values of Gamma_phot (s-1)
        Gamma_pions: [float]
            List with the values of Gamma_pions (s-1)
        
    """ 
    file = np.loadtxt(filename)
    ma = file[:,0]
    Gamma_phot = file[:,1]
    Gamma_pions = file[:,2]
    return ma, Gamma_phot, Gamma_pions

def Br (m_a):
    """
    Calculates the Branching ratios of a in photons and pions for a given mass

    Parameters
    ----------
        m_a: float
            Mass of the ALP in MeV
    
    Return
    ---------
        Br_pi: float
            Branching ratio of a --> pions
        Br_gamma: float
            Branching ratio of a --> photons
        
    """ 
    ma, gp, gpi= open_rates('DecayWidthsALP.txt')
    Gammaphotons_interp= interp1d(ma, gp, kind='cubic', fill_value="extrapolate")
    Gammapi_interp= interp1d(ma, gpi, kind='cubic', fill_value="extrapolate")
    Br_pi = Gammapi_interp(m_a/1000)/(Gammapi_interp(m_a/1000) + Gammaphotons_interp(m_a/1000))
    Br_gamma = Gammaphotons_interp(m_a/1000)/(Gammapi_interp(m_a/1000) + Gammaphotons_interp(m_a/1000))
    return Br_pi, Br_gamma

# %%
def F(rho_0, t_0, tau_a):
    return rho_0*(tau_a/t_0)**(-3/2)/(3*M_pl**2/(32*np.pi*tau_a**2))
#%%
def tau_max(Tdec, gsinterp, p, m, T0min = 1.61):
    """
    Calculates the maximum value of tau_a physically possible, given an initial quantity and a mass

    Parameters
    ----------
        Tdec: float
            Decoupling temperature of ALPS (MeV). (Equal to T0)
        gsinterp: function
            Interpolator of g*
        p: float
            Initial ratio of energy density of the ALP (p = rho_a/rho_SM(T0))
        m: float
            Mass of the ALP in MeV
        T0min: float
            Temperatura corresponding to the % of essos we are accepting in Yp. Default: 1.61
    
    Return
    ---------
        tau_m: float
            Maximum possible value of tau_a
        
        
    """ 
    Br_pi = Br(m)[0]
    Pconv = 2.5*10**-2*(1.5)**3
    na = p*rho_SM(Tdec,gsinterp(Tdec))/(m)
    ngammadec =  2 * zeta3 / np.pi**2 * Tdec**3
    diva =1
    tau_m = 0.023*(1.5/T0min)**2/(1 + 0.07*np.log(Pconv/0.1*Br_pi/0.4*2*na/3/ngammadec*24*(diva)**3))
    return tau_m

#%%
def nq_int_fermion (g,Nc,m,T):
    """
    Integral to calculate number density of fermions

    Parameters
    -------------
        g: int
            Degrees of freedom
        Nc: int
            Color number
        m: float
            ALP mass in MeV
        T: float
            Temperature en MeV

    """
    return quad(n_fermion, 0, np.inf,  args=(m,g,Nc,T),limit=500, epsabs=1e-10, epsrel=1e-10)[0]


def nq_charged(T):
    """
    Calculation of sum n_iQ_i^2

    Parameters
    ------------
        T: float
            Temperature in MeV
    """
    nq = nq_int_fermion(4,1, m_e, T) + nq_int_fermion(4,1, m_mu, T) + nq_int_fermion(4, 1,m_tau, T)
    
    if (T>300):  
        nq_quarks = (nq_int_fermion(4,3, mu, T)*qu**2 + nq_int_fermion(4,3, md, T)*qd**2 + nq_int_fermion(4,3, ms, T)*qs**2 +
                  nq_int_fermion(4,3, mc, T)*qc**2 + nq_int_fermion(4,3, mb, T)*qb**2 + nq_int_fermion(4,3, mt, T)*qt**2)
        nq += nq_quarks
    return nq

def n_axion_eq(T, m, g=1.0):
    """
    Calculation of number density of axions at temperature T. It T>100m a relativistic approx is performed

    Parameters
    ---------
        T: float
            Temperature in MeV
        m: float
            Mass in MeV
        g: int
            Degrees of freedom. Default = 1
    """
    if T/m >100:  
        return g * zeta3 / np.pi**2 * T**3
    else:
        def integrand(p):
            E = np.sqrt(p**2 + m**2)
            return p**2 / np.expm1(E/T)
    val, err = quad(integrand, 0, np.inf, limit=200)
    return g / (2*np.pi**2) * val

def interpolate_na0 (Tini,T_0):
    """
    Performs interpolations of gstar_interp, entropy_interp,prop_charged_interp,Temperature_interp.
    All of them are log-log except for entropy.

    Parameters
    -----------
        Tini: float
            Initial temperature (Treheating) in MeV
        T_0: float
            Final temperature in MeV. (Initial decay temperature)
    """
    df = pd.read_csv("standardmodel.dat", sep="\s+", skiprows=0)
    list_SM = df.values
    Temperatures = list_SM[:,0]
    gstar = list_SM[:,7]
    entropy = list_SM[:,3]

    gstar_interp = interp1d(np.log(Temperatures),np.log(gstar), kind='cubic')
    entropy_interp = interp1d((Temperatures),(entropy), kind='cubic')
    

    Temp_aux = np.logspace(np.log10(Tini),np.log10(T_0),1000)
    #list_prop_charged = [rho_charged_aux(T)/rho_SM(T,np.exp(gstar_interp(np.log(T)))) for T in Temp_aux]
    list_prop_charged = [nq_charged(T)/(zeta3*T**3/(np.pi**2)) for T in Temp_aux]
    prop_charged_interp = interp1d(np.log(Temp_aux),np.log(list_prop_charged), kind='cubic', fill_value="extrapolate")



    times = [time_RD(T, np.exp(gstar_interp(np.log(T)))) for T in Temp_aux]
    Temperature_interp = interp1d(np.log(times),np.log(Temp_aux), kind='cubic', fill_value="extrapolate")

    return gstar_interp, entropy_interp,prop_charged_interp,Temperature_interp
