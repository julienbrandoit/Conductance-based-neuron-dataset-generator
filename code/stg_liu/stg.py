import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import gsigmoid
from utils import get_w_factors, get_w_factors_constant_tau
from utils import d_gsigmoid
from utils import find_first_decreasing_zero_bisection
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from utils import gamma_uniform_mean_std_matching

# == simulation functions == #

import warnings

warnings.filterwarnings(
    "ignore",
    category=RuntimeWarning,
    message="divide by zero encountered in scalar divide"
)

def simulate_individual(args):
    """
    Simulate a single STG neuron from t=0 to T_final with uniform time step dt.

    args : (u0, individual, T_final, dt, params)

    Returns a (2, N) array: row 0 is time, row 1 is voltage.
    """
    u0, individual, T_final, dt, params = args
    t_eval = np.arange(0, T_final, dt)
    return simulate_individual_t_eval((u0, individual, t_eval, params))


def simulate_individual_t_eval(args, ode_func=None, sigma_noise=0.0, cutoff_freq=1000.0):
    """
    Simulate a single STG neuron at specified time points.

    args : (u0, individual, t_eval, params)
    ode_func : ODE function to use; defaults to ODEs. Use ODEs_with_noisy_current for noisy injection.
    sigma_noise : std of Gaussian noise (only used with ODEs_with_noisy_current).
    cutoff_freq : lowpass cutoff frequency in Hz for noise filtering.

    Returns a (2, N) array: row 0 is time, row 1 is voltage.
    """
    u0, individual, t_eval, params = args

    # Default to standard ODEs if not specified
    if ode_func is None:
        ode_func = ODEs

    # Prepare arguments for the ODE function
    ode_args = (
        individual[0], individual[1], individual[2], individual[3],
        individual[4], individual[5], individual[6], individual[7],
        params['E_Na'], params['E_K'], params['E_H'], params['E_leak'],
        params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca']
    )

    # Generate filtered noise and add to args if using noisy current ODE
    if ode_func == ODEs_with_noisy_current:
        filtered_noise = generate_filtered_noise(t_eval, sigma_noise, cutoff_freq)
        ode_args = ode_args + (t_eval, filtered_noise)

    sol = solve_ivp(
        ode_func,
        [0, t_eval[-1]],
        u0,
        t_eval=t_eval,
        args=ode_args,
        method='BDF',
        dense_output=False,
        rtol=1e-2,
        atol=1e-4
    )
    return np.array((sol.t, sol.y[0]))


def get_u0(V0, Ca0):
    """
    Build the initial state vector for the STG model at membrane potential V0 and calcium Ca0.

    Returns an array of 13 state variables initialised at steady state.
    """
    u0 = np.zeros(13)

    u0[0] = V0
    u0[1] = m_inf_Na(V0)
    u0[2] = h_inf_Na(V0)
    u0[3] = m_inf_Kd(V0)
    u0[4] = m_inf_CaT(V0)
    u0[5] = h_inf_CaT(V0)
    u0[6] = m_inf_CaS(V0)
    u0[7] = h_inf_CaS(V0)
    u0[8] = m_inf_KCa(V0, Ca0)
    u0[9] = m_inf_A(V0)
    u0[10] = h_inf_A(V0)
    u0[11] = m_inf_H(V0)
    u0[12] = Ca0

    return u0


def get_default_parameters():
    """
    Return the default reversal potentials and calcium dynamics parameters for the STG model.

    Returns a dict with keys E_leak, E_Na, E_K, E_H, E_Ca, tau_Ca, alpha_Ca, beta_Ca.
    """
    params = {}

    params['E_leak'] = -50  # Leak reversal potential
    params['E_Na'] = 50    # Sodium reversal potential
    params['E_K'] = -80    # Potassium reversal potential
    params['E_H'] = -20    # H-current reversal potential
    params['E_Ca'] = 80    # Calcium reversal potential

    params['tau_Ca'] = 20          # Calcium decay time constant
    params['alpha_Ca'] = 0.94
    params['beta_Ca'] = 0.05

    return params


def generate_filtered_noise(t_eval, sigma_noise, cutoff_freq=1000.0, dt=None):
    """
    Generate Gaussian noise lowpass-filtered with a 4th-order Butterworth filter.

    t_eval : time points (ms); used to infer sampling rate if dt is None.
    sigma_noise : std of the noise before filtering.
    cutoff_freq : filter cutoff in Hz.

    Returns a 1-D array of filtered noise the same length as t_eval.
    """
    if dt is None:
        dt = np.mean(np.diff(t_eval))
    
    fs = 1.0 / dt * 1000  # dt is in ms, convert to Hz
    noise = np.random.normal(0, sigma_noise, len(t_eval))
    
    nyquist = 0.5 * fs
    normal_cutoff = cutoff_freq / nyquist
    b, a = butter(4, normal_cutoff, btype='low', analog=False)
    filtered_noise = filtfilt(b, a, noise)
    
    return filtered_noise


def get_default_u0():
    """
    Return the default initial state vector for the STG model (V0=-70 mV, Ca0=0.5 uM).

    Returns an array of 13 state variables.
    """
    V0 = -70  # Resting membrane potential
    Ca0 = 0.5 # Initial calcium concentration
    u0 = get_u0(V0, Ca0)
    return u0


def get_best_set(g_s, g_u):
    """
    Return the pair of conductances to neuromodulate based on (g_s, g_u).

    Returns ['CaS', 'A'] if g_u < 0, ['A', 'H'] if g_s >= 0, else ['CaS', 'H'].
    """
    if g_u < 0:
        #print('Cautious, g_u is negative!')
        return ['CaS', 'A']
    
    if g_s >= 0:
        return ['A', 'H']
    if g_s < 0:
        return ['CaS', 'H']

    
# == Gating variables functions == #

def m_inf_Na(V):
    """Steady-state activation (m) for the Na channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=-5.29, D=25.5)


def h_inf_Na(V):
    """Steady-state inactivation (h) for the Na channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=5.18, D=48.9)


def tau_m_Na(V):
    """Time constant for Na activation (m). Returns float or array."""
    return gsigmoid(V, A=1.32, B=-1.26, C=-25, D=120)


def tau_h_Na(V):
    """Time constant for Na inactivation (h). Returns float or array."""
    return gsigmoid(V, A=0, B=0.67, C=-10, D=62.9) * gsigmoid(V, A=1.5, B=1, C=3.6, D=34.9)


def m_inf_Kd(V):
    """Steady-state activation (m) for the Kd channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=-11.8, D=12.3)


def tau_m_Kd(V):
    """Time constant for Kd activation (m). Returns float or array."""
    return gsigmoid(V, A=7.2, B=-6.4, C=-19.2, D=28.3)


def m_inf_CaT(V):
    """Steady-state activation (m) for the CaT channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=-7.2, D=27.1)


def h_inf_CaT(V):
    """Steady-state inactivation (h) for the CaT channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=5.5, D=32.1)


def tau_m_CaT(V):
    """Time constant for CaT activation (m). Returns float or array."""
    return gsigmoid(V, A=21.7, B=-21.3, C=-20.5, D=68.1)


def tau_h_CaT(V):
    """Time constant for CaT inactivation (h). Returns float or array."""
    return gsigmoid(V, A=105, B=-89.8, C=-16.9, D=55)

def m_inf_CaS(V):
    """Steady-state activation (m) for the CaS channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=-8.1, D=33)


def h_inf_CaS(V):
    """Steady-state inactivation (h) for the CaS channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=6.2, D=60)


def tau_m_CaS(V):
    """Time constant for CaS activation (m). Returns float or array."""
    return 1.4 + 7/(np.exp((V+27)/10) + np.exp((V+70)/-13))


def tau_h_CaS(V):
    """Time constant for CaS inactivation (h). Returns float or array."""
    return 60 + 150/(np.exp((V+55)/9) + np.exp((V+65)/-16))


def m_inf_KCa(V, Ca):
    """Steady-state activation (m) for the KCa channel. Returns float or array."""
    return Ca/(Ca + 3) * gsigmoid(V, A=0, B=1, C=-12.6, D=28.3)


def tau_m_KCa(V):
    """Time constant for KCa activation (m). Returns float or array."""
    return gsigmoid(V, A=90.3, B=-75.1, C=-22.7, D=46)


def m_inf_A(V):
    """Steady-state activation (m) for the A-type K channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=-8.7, D=27.2)


def h_inf_A(V):
    """Steady-state inactivation (h) for the A-type K channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=4.9, D=56.9)


def tau_m_A(V):
    """Time constant for A-type K activation (m). Returns float or array."""
    return gsigmoid(V, A=11.6, B=-10.4, C=-15.2, D=32.9)


def tau_h_A(V):
    """Time constant for A-type K inactivation (h). Returns float or array."""
    return gsigmoid(V, A=38.6, B=-29.2, C=-26.5, D=38.9)


def m_inf_H(V):
    """Steady-state activation (m) for the H channel. Returns float or array."""
    return gsigmoid(V, A=0, B=1, C=6, D=70)


def tau_m_H(V):
    """Time constant for H activation (m). Returns float or array."""
    return gsigmoid(V, A=272, B=1499, C=-8.73, D=42.2)

# == Derivatives == #

def d_m_inf_Na(V):
    """Derivative of m_inf_Na with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = -5.29, D = 25.5)

def d_h_inf_Na(V):
    """Derivative of h_inf_Na with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = 5.18, D = 48.9)

def d_m_inf_Kd(V):
    """Derivative of m_inf_Kd with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = -11.8, D = 12.3)

def d_m_inf_CaT(V):
    """Derivative of m_inf_CaT with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = -7.2, D = 27.1)

def d_h_inf_CaT(V):
    """Derivative of h_inf_CaT with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = 5.5, D = 32.1)

def d_m_inf_CaS(V):
    """Derivative of m_inf_CaS with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = -8.1, D = 33)

def d_h_inf_CaS(V):
    """Derivative of h_inf_CaS with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = 6.2, D = 60)

def d_m_inf_KCa_dV(V, Ca):  
    """Derivative of m_inf_KCa with respect to V. Returns float or array."""
    return Ca/(Ca + 3) * d_gsigmoid(V, A = 0, B = 1, C = -12.6, D = 28.3)

def d_m_inf_KCa_dCa(V, Ca):  
    """Derivative of m_inf_KCa with respect to Ca. Returns float or array."""
    return 3 * gsigmoid(V, A = 0, B = 1, C = -12.6, D = 28.3) / (Ca + 3)**2

def d_Ca_inf_dV(V, alpha, E_Ca, g_CaT, g_CaS, m_inf_CaT_values, m_inf_CaS_values, h_inf_CaT_values, h_inf_CaS_values, d_m_inf_CaT_values, d_m_inf_CaS_values, d_h_inf_CaT_values, d_h_inf_CaS_values): 
    """Derivative of the equilibrium Ca concentration with respect to V. Returns float or array."""
    d = np.zeros_like(V)
    d = g_CaT * 3 * m_inf_CaT_values**2 * h_inf_CaT_values * d_m_inf_CaT_values * (V - E_Ca) +\
        g_CaT * m_inf_CaT_values**3 * d_h_inf_CaT_values * (V - E_Ca) +\
        g_CaT * m_inf_CaT_values**3 * h_inf_CaT_values +\
        g_CaS * 3 * m_inf_CaS_values**2 * h_inf_CaS_values * d_m_inf_CaS_values * (V - E_Ca) +\
        g_CaS * m_inf_CaS_values**3 * d_h_inf_CaS_values * (V - E_Ca) +\
        g_CaS * m_inf_CaS_values**3 * h_inf_CaS_values
    return - alpha * d

def d_m_inf_A(V):
    """Derivative of m_inf_A with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = -8.7, D = 27.2)

def d_h_inf_A(V):
    """Derivative of h_inf_A with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = 4.9, D = 56.9)

def d_m_inf_H(V):
    """Derivative of m_inf_H with respect to V. Returns float or array."""
    return d_gsigmoid(V, A = 0, B = 1, C = 6, D = 70)

# == UTILS == #

def compute_equilibrium_Ca(alpha, I_Ca, beta):
    """Compute the steady-state Ca concentration from dCa/dt=0: returns -alpha*I_Ca + beta."""
    return -alpha * I_Ca + beta


def find_V_th_DICs(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
             E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
             tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, get_I_static = False, normalize = True, y_tol = 1e-6, x_tol=1e-6, max_iter = 1000, verbose=True):
    """
    Find the threshold voltage V_th where the total DIC (g_t) first decreases through zero.

    Uses bisection over the array V. Returns (V_th, (g_f, g_s, g_u, g_t)) evaluated at V_th.
    Returns (V_th, (nan, nan, nan, nan)) if no zero is found.
    """
    g_t = lambda V_scalar : DICs(np.asarray([V_scalar,]), g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg, False, normalize)[3]

    V_th = find_first_decreasing_zero_bisection(V, g_t, y_tol = y_tol, x_tol=x_tol, max_iter = max_iter, verbose=verbose)
    V_th = np.asarray([V_th,], dtype=np.float64)

    if V_th is None or np.isnan(V_th):
        return V_th, (np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan))
        
    values = DICs(V_th, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg, get_I_static, normalize)
    
    return V_th, values


# == ODEs == #
def ODEs(t, u, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca):        
    """
    Compute time derivatives of the STG neuron state vector.

    u[0..12]: V, m_Na, h_Na, m_Kd, m_CaT, h_CaT, m_CaS, h_CaS, m_KCa, m_A, h_A, m_H, Ca.
    Returns du of the same shape as u.
    """
    du = np.zeros_like(u)
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    m_Kd = u[3]
    m_CaT, h_CaT = u[4], u[5]
    m_CaS, h_CaS = u[6], u[7]
    m_KCa = u[8]
    m_A, h_A = u[9], u[10]
    m_H = u[11]
    Ca = u[12]

    I_Na = g_Na * m_Na**3 * h_Na * (V - E_Na)
    I_Kd = g_Kd * m_Kd**4 * (V - E_K)
    I_CaT = g_CaT * m_CaT**3 * h_CaT * (V - E_Ca)
    I_CaS = g_CaS * m_CaS**3 * h_CaS * (V - E_Ca)
    I_KCa = g_KCa * m_KCa**4 * (V - E_K)
    I_A = g_A * m_A**3 * h_A * (V - E_K)
    I_H = g_H * m_H * (V - E_H)
    I_leak = g_leak * (V - E_leak)

    du[0] = -(I_Na + I_Kd + I_CaT + I_CaS + I_KCa + I_A + I_H + I_leak)

    I_Ca = I_CaT + I_CaS
    du[12] = (-alpha_Ca * I_Ca - Ca + beta_Ca) / tau_Ca

    du[1] = (m_inf_Na(V) - m_Na) / tau_m_Na(V)
    du[2] = (h_inf_Na(V) - h_Na) / tau_h_Na(V)
    du[3] = (m_inf_Kd(V) - m_Kd) / tau_m_Kd(V)
    du[4] = (m_inf_CaT(V) - m_CaT) / tau_m_CaT(V)
    du[5] = (h_inf_CaT(V) - h_CaT) / tau_h_CaT(V)
    du[6] = (m_inf_CaS(V) - m_CaS) / tau_m_CaS(V)
    du[7] = (h_inf_CaS(V) - h_CaS) / tau_h_CaS(V)
    du[8] = (m_inf_KCa(V, Ca) - m_KCa) / tau_m_KCa(V)
    du[9] = (m_inf_A(V) - m_A) / tau_m_A(V)
    du[10] = (h_inf_A(V) - h_A) / tau_h_A(V)
    du[11] = (m_inf_H(V) - m_H) / tau_m_H(V)

    return du


def ODEs_with_noisy_current(t, u, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, t_eval, filtered_noise):        
    """
    Same as ODEs but with a pre-generated lowpass-filtered noisy current injected into the voltage equation.

    t_eval and filtered_noise define the noise signal, interpolated at time t.
    Returns du of the same shape as u.
    """
    du = np.zeros_like(u)
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    m_Kd = u[3]
    m_CaT, h_CaT = u[4], u[5]
    m_CaS, h_CaS = u[6], u[7]
    m_KCa = u[8]
    m_A, h_A = u[9], u[10]
    m_H = u[11]
    Ca = u[12]

    I_Na = g_Na * m_Na**3 * h_Na * (V - E_Na)
    I_Kd = g_Kd * m_Kd**4 * (V - E_K)
    I_CaT = g_CaT * m_CaT**3 * h_CaT * (V - E_Ca)
    I_CaS = g_CaS * m_CaS**3 * h_CaS * (V - E_Ca)
    I_KCa = g_KCa * m_KCa**4 * (V - E_K)
    I_A = g_A * m_A**3 * h_A * (V - E_K)
    I_H = g_H * m_H * (V - E_H)
    I_leak = g_leak * (V - E_leak)
    
    I_noise = np.interp(t, t_eval, filtered_noise)

    du[0] = -(I_Na + I_Kd + I_CaT + I_CaS + I_KCa + I_A + I_H + I_leak) + I_noise

    I_Ca = I_CaT + I_CaS
    du[12] = (-alpha_Ca * I_Ca - Ca + beta_Ca) / tau_Ca

    du[1] = (m_inf_Na(V) - m_Na) / tau_m_Na(V)
    du[2] = (h_inf_Na(V) - h_Na) / tau_h_Na(V)
    du[3] = (m_inf_Kd(V) - m_Kd) / tau_m_Kd(V)
    du[4] = (m_inf_CaT(V) - m_CaT) / tau_m_CaT(V)
    du[5] = (h_inf_CaT(V) - h_CaT) / tau_h_CaT(V)
    du[6] = (m_inf_CaS(V) - m_CaS) / tau_m_CaS(V)
    du[7] = (h_inf_CaS(V) - h_CaS) / tau_h_CaS(V)
    du[8] = (m_inf_KCa(V, Ca) - m_KCa) / tau_m_KCa(V)
    du[9] = (m_inf_A(V) - m_A) / tau_m_A(V)
    du[10] = (h_inf_A(V) - h_A) / tau_h_A(V)
    du[11] = (m_inf_H(V) - m_H) / tau_m_H(V)

    return du


# == DICs related functions == #

def DICs(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
         E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
         tau_f_stg=tau_m_Na, tau_s_stg=tau_m_Kd, tau_u_stg=tau_m_H, get_I_static=False, normalize=True):
    """
    Compute the fast (g_f), slow (g_s), ultra-slow (g_u), and total (g_t) dynamic input conductances.

    V and conductances can be scalars or arrays. If get_I_static is True, also returns I_static.
    """

    # get the S matrix
    if not get_I_static:
        S = sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
                               E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
                               tau_f_stg, tau_s_stg, tau_u_stg, normalize, get_I_static)
    else:
        S, S_static = sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
                                         E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
                                         tau_f_stg, tau_s_stg, tau_u_stg, normalize, get_I_static)

    if S.ndim == 3:
        S = S[np.newaxis, :, :, :]

    m = S.shape[0]

    g_vec = np.array([g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak]).T
    g_vec = np.atleast_2d(g_vec)
    g_vec = g_vec[:, :, np.newaxis, np.newaxis].transpose(0, 2, 1, 3)

    S_mul = np.sum(S * g_vec, axis=2)

    g_f = S_mul[:, 0, :]
    g_s = S_mul[:, 1, :]
    g_u = S_mul[:, 2, :]
    g_t = g_f + g_s + g_u

    if get_I_static:
        V_Na = V - E_Na
        V_K = V - E_K
        V_Ca = V - E_Ca
        V_H = V - E_H
        V_leak = V - E_leak

        V_vec = np.array([V_Na, V_K, V_Ca, V_Ca, V_K, V_K, V_H, V_leak])
        g_vec = g_vec[:, 0, :, 0][:, :, np.newaxis]
        
        I_static = np.sum(S_static * g_vec * V_vec, axis=1)

        if m == 1:
            I_static = I_static[0]
            g_f = g_f[0]
            g_s = g_s[0]
            g_u = g_u[0]
            g_t = g_t[0]

        return g_f, g_s, g_u, g_t, I_static
    
    if m == 1:
        g_f = g_f[0]
        g_s = g_s[0]
        g_u = g_u[0]
        g_t = g_t[0]

    return g_f, g_s, g_u, g_t

def sensitivity_matrix(V, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, g_leak,
             E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca,
             tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, normalize = True, get_I_static = False):
    """
    Compute the DIC sensitivity matrix S of shape (3, 8, n) for m=1 or (m, 3, 8, n) for m>1.

    Each entry S[f/s/u, channel, v] gives the contribution of that channel to a given DIC timescale
    at voltage v. If get_I_static is True, also returns the static current matrix S_static.
    """
    
    V = np.atleast_1d(V)
    g_Na = np.atleast_1d(g_Na)
    g_Kd = np.atleast_1d(g_Kd)
    g_CaT = np.atleast_1d(g_CaT)
    g_CaS = np.atleast_1d(g_CaS)
    g_KCa = np.atleast_1d(g_KCa)
    g_A = np.atleast_1d(g_A)
    g_H = np.atleast_1d(g_H)
    g_leak = np.atleast_1d(g_leak)

    # m represents the number of conductances, it will be 1 for scalars, or match the size of arrays
    m = g_Na.size # we assume all conductances have the same size ... should be the case with a proper call to the function !
    n = V.size  # n is the size of V

    # S will be a 3xN matrix with N = 7+1. Each row will correspond to a different variable (f, s, u) and each column to a different channel.
    S_Na = np.zeros((m, 3, n))
    S_Kd = np.zeros((m, 3, n))
    S_CaT = np.zeros((m, 3, n))
    S_CaS = np.zeros((m, 3, n))
    S_KCa = np.zeros((m, 3, n))
    S_A = np.zeros((m, 3, n))
    S_H = np.zeros((m, 3, n))
    S_leak = np.zeros((m, 3, n))

    V = np.atleast_2d(V).T

    m_inf_Na_values = m_inf_Na(V)
    h_inf_Na_values = h_inf_Na(V)
    m_inf_Kd_values = m_inf_Kd(V)
    m_inf_CaT_values = m_inf_CaT(V)
    h_inf_CaT_values = h_inf_CaT(V)
    m_inf_CaS_values = m_inf_CaS(V)
    h_inf_CaS_values = h_inf_CaS(V)

    I_CaT = g_CaT * m_inf_CaT_values**3 * h_inf_CaT_values * (V - E_Ca)
    I_CaS = g_CaS * m_inf_CaS_values**3 * h_inf_CaS_values * (V - E_Ca)

    I_Ca = I_CaT + I_CaS
    Ca = compute_equilibrium_Ca(alpha_Ca, I_Ca, beta_Ca)
    
    m_inf_KCa_values = m_inf_KCa(V, Ca)
    m_inf_A_values = m_inf_A(V)
    h_inf_A_values = h_inf_A(V)
    m_inf_H_values = m_inf_H(V)

    d_m_inf_Na_values = d_m_inf_Na(V)
    d_h_inf_Na_values = d_h_inf_Na(V)
    d_m_inf_Kd_values = d_m_inf_Kd(V)
    d_m_inf_CaT_values = d_m_inf_CaT(V)
    d_h_inf_CaT_values = d_h_inf_CaT(V)
    d_m_inf_CaS_values = d_m_inf_CaS(V)
    d_h_inf_CaS_values = d_h_inf_CaS(V)
    d_m_inf_A_values = d_m_inf_A(V)
    d_h_inf_A_values = d_h_inf_A(V)
    d_m_inf_H_values = d_m_inf_H(V)
    d_m_inf_KCa_dCa_values = d_m_inf_KCa_dCa(V, Ca)
    d_m_inf_KCa_dV_values = d_m_inf_KCa_dV(V, Ca)
    d_Ca_inf_dV_values = d_Ca_inf_dV(V, alpha_Ca, E_Ca, g_CaT, g_CaS, m_inf_CaT_values, m_inf_CaS_values, h_inf_CaT_values, h_inf_CaS_values, d_m_inf_CaT_values, d_m_inf_CaS_values, d_h_inf_CaT_values, d_h_inf_CaS_values)

    S_Na[:, 0, :] += (m_inf_Na_values**3 * h_inf_Na_values).T
    S_Kd[:,0,:] += (m_inf_Kd_values**4).T
    S_CaT[:,0,:] += (m_inf_CaT_values**3 * h_inf_CaT_values).T
    S_CaS[:,0,:] += (m_inf_CaS_values**3 * h_inf_CaS_values).T
    S_KCa[:,0,:] += (m_inf_KCa_values**4).T
    S_A[:,0,:] += (m_inf_A_values**3 * h_inf_A_values).T
    S_H[:,0,:] += (m_inf_H_values).T
    S_leak[:,0,:] += 1.0

    if get_I_static:
        S_Na_static = S_Na[:, 0, :].copy()
        S_Kd_static = S_Kd[:, 0, :].copy()
        S_CaT_static = S_CaT[:, 0, :].copy()
        S_CaS_static = S_CaS[:, 0, :].copy()
        S_KCa_static = S_KCa[:, 0, :].copy()
        S_A_static = S_A[:, 0, :].copy()
        S_H_static = S_H[:, 0, :].copy()
        S_leak_static = S_leak[:, 0, :].copy()

    dV_dot_dm_Na = - 3 * m_inf_Na_values**2 * h_inf_Na_values * d_m_inf_Na_values * (V - E_Na)
    dV_dot_dh_Na = - m_inf_Na_values**3 * d_h_inf_Na_values * (V - E_Na)
    dV_dot_dm_Kd = - 4 * m_inf_Kd_values**3 * d_m_inf_Kd_values * (V - E_K)
    dV_dot_dm_CaT = - 3 * m_inf_CaT_values**2 * h_inf_CaT_values * d_m_inf_CaT_values * (V - E_Ca)
    dV_dot_dh_CaT = - m_inf_CaT_values**3 * d_h_inf_CaT_values * (V - E_Ca) 
    dV_dot_dm_CaS = - 3 * m_inf_CaS_values**2 * h_inf_CaS_values * d_m_inf_CaS_values * (V - E_Ca)
    dV_dot_dh_CaS = - m_inf_CaS_values**3 * d_h_inf_CaS_values * (V - E_Ca)
    dV_dot_dm_KCa = - 4 * m_inf_KCa_values**3 * d_m_inf_KCa_dV_values * (V - E_K) 
    dV_dot_dCa_KCa = - 4 * m_inf_KCa_values**3 * d_m_inf_KCa_dCa_values * (V - E_K) * d_Ca_inf_dV_values 
    dV_dot_dm_A = - 3 * m_inf_A_values**2 * h_inf_A_values * d_m_inf_A_values * (V - E_K)
    dV_dot_dh_A = - m_inf_A_values**3 * d_h_inf_A_values * (V - E_K)
    dV_dot_dm_H = - d_m_inf_H_values * (V - E_H)

    w_fs_m_Na, w_su_m_Na = get_w_factors(V, tau_m_Na, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_Na, w_su_h_Na = get_w_factors(V, tau_h_Na, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_Kd, w_su_m_Kd = get_w_factors(V, tau_m_Kd, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_CaT, w_su_m_CaT = get_w_factors(V, tau_m_CaT, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_CaT, w_su_h_CaT = get_w_factors(V, tau_h_CaT, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_CaS, w_su_m_CaS = get_w_factors(V, tau_m_CaS, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_CaS, w_su_h_CaS = get_w_factors(V, tau_h_CaS, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_KCa, w_su_m_KCa = get_w_factors(V, tau_m_KCa, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_KCa2, w_su_m_KCa2 = get_w_factors_constant_tau(V, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg) # TO BE CHANGED
    w_fs_m_A, w_su_m_A = get_w_factors(V, tau_m_A, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_h_A, w_su_h_A = get_w_factors(V, tau_h_A, tau_f_stg, tau_s_stg, tau_u_stg)
    w_fs_m_H, w_su_m_H = get_w_factors(V, tau_m_H, tau_f_stg, tau_s_stg, tau_u_stg)

    S_Na[:,0,:] += (- w_fs_m_Na * dV_dot_dm_Na - w_fs_h_Na * dV_dot_dh_Na).T
    S_Kd[:,0,:] += (- w_fs_m_Kd * dV_dot_dm_Kd).T
    S_CaT[:,0,:] += (- w_fs_m_CaT * dV_dot_dm_CaT - w_fs_h_CaT * dV_dot_dh_CaT).T
    S_CaS[:,0,:] += (- w_fs_m_CaS * dV_dot_dm_CaS - w_fs_h_CaS * dV_dot_dh_CaS).T
    S_KCa[:,0,:] += (- w_fs_m_KCa * dV_dot_dm_KCa - w_fs_m_KCa2 * dV_dot_dCa_KCa).T
    S_A[:,0,:] += (- w_fs_m_A * dV_dot_dm_A - w_fs_h_A * dV_dot_dh_A).T
    S_H[:,0,:] += (- w_fs_m_H * dV_dot_dm_H).T

    S_Na[:,1,:] += (- (w_su_m_Na - w_fs_m_Na) * dV_dot_dm_Na - (w_su_h_Na - w_fs_h_Na) * dV_dot_dh_Na).T
    S_Kd[:,1,:] += (- (w_su_m_Kd - w_fs_m_Kd) * dV_dot_dm_Kd).T
    S_CaT[:,1,:] += (- (w_su_m_CaT - w_fs_m_CaT) * dV_dot_dm_CaT - (w_su_h_CaT - w_fs_h_CaT) * dV_dot_dh_CaT).T
    S_CaS[:,1,:] += (- (w_su_m_CaS - w_fs_m_CaS) * dV_dot_dm_CaS - (w_su_h_CaS - w_fs_h_CaS) * dV_dot_dh_CaS).T
    S_KCa[:,1,:] += (- (w_su_m_KCa - w_fs_m_KCa) * dV_dot_dm_KCa - (w_su_m_KCa2 - w_fs_m_KCa2) * dV_dot_dCa_KCa).T
    S_A[:,1,:] += (- (w_su_m_A - w_fs_m_A) * dV_dot_dm_A - (w_su_h_A - w_fs_h_A) * dV_dot_dh_A).T
    S_H[:,1,:] += (- (w_su_m_H - w_fs_m_H) * dV_dot_dm_H).T

    S_Na[:,2,:] += (- (1 - w_su_m_Na) * dV_dot_dm_Na - (1 - w_su_h_Na) * dV_dot_dh_Na).T
    S_Kd[:,2,:] += (- (1 - w_su_m_Kd) * dV_dot_dm_Kd).T
    S_CaT[:,2,:] += (- (1 - w_su_m_CaT) * dV_dot_dm_CaT - (1 - w_su_h_CaT) * dV_dot_dh_CaT).T
    S_CaS[:,2,:] += (- (1 - w_su_m_CaS) * dV_dot_dm_CaS - (1 - w_su_h_CaS) * dV_dot_dh_CaS).T
    S_KCa[:,2,:] += (- (1 - w_su_m_KCa) * dV_dot_dm_KCa - (1 - w_su_m_KCa2) * dV_dot_dCa_KCa).T
    S_A[:,2,:] += (- (1 - w_su_m_A) * dV_dot_dm_A - (1 - w_su_h_A) * dV_dot_dh_A).T
    S_H[:,2,:] += (- (1 - w_su_m_H) * dV_dot_dm_H).T

    if normalize:
        S_Na /= g_leak[:, np.newaxis, np.newaxis]
        S_Kd /= g_leak[:, np.newaxis, np.newaxis]
        S_CaT /= g_leak[:, np.newaxis, np.newaxis]
        S_CaS /= g_leak[:, np.newaxis, np.newaxis]
        S_KCa /= g_leak[:, np.newaxis, np.newaxis]
        S_A /= g_leak[:, np.newaxis, np.newaxis]
        S_H /= g_leak[:, np.newaxis, np.newaxis]
        S_leak /= g_leak[:, np.newaxis, np.newaxis]

    S_Na = S_Na[:, :, np.newaxis, :]
    S_Kd = S_Kd[:, :, np.newaxis, :]
    S_CaT = S_CaT[:, :, np.newaxis, :]
    S_CaS = S_CaS[:, :, np.newaxis, :]
    S_KCa = S_KCa[:, :, np.newaxis, :]
    S_A = S_A[:, :, np.newaxis, :]
    S_H = S_H[:, :, np.newaxis, :]
    S_leak = S_leak[:, :, np.newaxis, :]

    if not get_I_static:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaT[0], S_CaS[0], S_KCa[0], S_A[0], S_H[0], S_leak[0]), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaT, S_CaS, S_KCa, S_A, S_H, S_leak), axis=2)
    else:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaT[0], S_CaS[0], S_KCa[0], S_A[0], S_H[0], S_leak[0]), axis=1), np.stack((S_Na_static, S_Kd_static, S_CaT_static, S_CaS_static, S_KCa_static, S_A_static, S_H_static, S_leak_static), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaT, S_CaS, S_KCa, S_A, S_H, S_leak), axis=2), np.stack((S_Na_static, S_Kd_static, S_CaT_static, S_CaS_static, S_KCa_static, S_A_static, S_H_static, S_leak_static), axis=1)

# == Compensation algorithms and generation functions == #

def generate_population(n_cells, V_th,  g_f_target, g_s_target, g_u_target, g_bar_range_leak, g_bar_range_Na, g_bar_range_Kd, g_bar_range_CaT, g_bar_range_CaS, g_bar_range_KCa, g_bar_range_A, g_bar_range_H, params, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0, distribution='uniform', normalize_by_leak=True):
    """
    Generate n_cells neurons whose DICs match (g_f_target, g_s_target, g_u_target) at V_th.

    Conductances for None ranges are compensated; others are drawn from the given ranges using
    the specified distribution ('uniform' or 'gamma'). Returns an (n_cells, 8) array of conductances.
    """
    g_Na = np.random.uniform(g_bar_range_Na[0], g_bar_range_Na[1], n_cells) if g_bar_range_Na is not None else np.full(n_cells, np.nan)
    g_Kd = np.random.uniform(g_bar_range_Kd[0], g_bar_range_Kd[1], n_cells) if g_bar_range_Kd is not None else np.full(n_cells, np.nan)
    g_CaT = np.random.uniform(g_bar_range_CaT[0], g_bar_range_CaT[1], n_cells) if g_bar_range_CaT is not None else np.full(n_cells, np.nan)
    g_CaS = np.random.uniform(g_bar_range_CaS[0], g_bar_range_CaS[1], n_cells) if g_bar_range_CaS is not None else np.full(n_cells, np.nan)
    g_KCa = np.random.uniform(g_bar_range_KCa[0], g_bar_range_KCa[1], n_cells) if g_bar_range_KCa is not None else np.full(n_cells, np.nan)
    g_A = np.random.uniform(g_bar_range_A[0], g_bar_range_A[1], n_cells) if g_bar_range_A is not None else np.full(n_cells, np.nan)
    g_H = np.random.uniform(g_bar_range_H[0], g_bar_range_H[1], n_cells) if g_bar_range_H is not None else np.full(n_cells, np.nan)
    
    if distribution == 'uniform':
        # here we assume that the ranges are the min and max values for the uniform distribution
        mean_leak = (g_bar_range_leak[0] + g_bar_range_leak[1]) / 2
        g_leak = np.random.uniform(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
        
    elif distribution == 'gamma':
        g_bar_range_leak = gamma_uniform_mean_std_matching(*g_bar_range_leak)
        mean_leak = g_bar_range_leak[0] * g_bar_range_leak[1]
        g_leak = np.random.gamma(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
    else:
        raise ValueError('Invalid distribution type ! Please use either "uniform" or "gamma".')

    if normalize_by_leak:
        f = g_leak/mean_leak
        g_Na *= f
        g_Kd *= f
        g_CaT *= f
        g_CaS *= f
        g_KCa *= f
        g_A *= f
        g_H *= f

    x = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], g_leak, g_Na, g_Kd, g_CaT, g_CaS, g_KCa, g_A, g_H, params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca'], default_g_CaS_for_Ca=default_g_CaS_for_Ca, default_g_CaT_for_Ca=default_g_CaT_for_Ca)
    
    return x

def modulate_population(population, V_th, g_f_target, g_s_target, g_u_target, params, set_to_compensate, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0, iterations=0):
    """
    Adjust the conductances in population so that the DICs match the given targets at V_th.

    set_to_compensate lists the conductance names to be solved for (e.g. ['Na', 'A', 'H']).
    The number of entries must equal the number of non-None targets.
    Returns an (n_cells, 8) array of modulated conductances.
    """
    population = np.asarray(population)
    
    number_none_dics = 0
    if g_f_target is None or np.isnan(g_f_target):
        number_none_dics += 1
    if g_s_target is None or np.isnan(g_s_target):
        number_none_dics += 1
    if g_u_target is None or np.isnan(g_u_target):
        number_none_dics += 1

    if 3 - number_none_dics != len(set_to_compensate):
        raise ValueError('Number of conductances to compensate should be equal to the number of target DICS')

    while iterations >= 0:
        if 'Na' in set_to_compensate:
            population[:, 0] = np.nan
        if 'Kd' in set_to_compensate:
            population[:, 1] = np.nan
        if 'CaT' in set_to_compensate:
            population[:, 2] = np.nan
        if 'CaS' in set_to_compensate:
            population[:, 3] = np.nan
        if 'KCa' in set_to_compensate:
            population[:, 4] = np.nan
        if 'A' in set_to_compensate:
            population[:, 5] = np.nan
        if 'H' in set_to_compensate:
            population[:, 6] = np.nan

        if 'CaS' not in set_to_compensate and 'CaT' not in set_to_compensate:
            iterations = 0

        population = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], population[:, 7], population[:, 0], population[:, 1], population[:, 2], population[:, 3], population[:, 4], population[:, 5], population[:, 6], params['E_Na'], params['E_K'], params['E_H'], params['E_leak'], params['E_Ca'], params['alpha_Ca'], params['beta_Ca'], params['tau_Ca'], default_g_CaS_for_Ca=default_g_CaS_for_Ca, default_g_CaT_for_Ca=default_g_CaT_for_Ca)
        iterations -= 1
        default_g_CaT_for_Ca = population[:, 2].copy()
        default_g_CaS_for_Ca = population[:, 3].copy()

    return population

def general_compensation_algorithm(V_th, target_DICs, new_g_leak, new_g_Na, new_g_Kd, new_g_CaT, new_g_CaS, new_g_KCa, new_g_A, new_g_H, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg = tau_m_Na, tau_s_stg = tau_m_Kd, tau_u_stg = tau_m_H, default_g_CaS_for_Ca = 10., default_g_CaT_for_Ca = 6.0):
    """
    Solve for the conductances left as None/NaN so that the DICs match target_DICs at V_th.

    The number of NaN conductances must equal the number of non-NaN entries in target_DICs.
    If CaT or CaS are among those solved, default values are used as linearization points
    (approximate). Returns an (n_cells, 8) array with all conductances filled in.
    """
    none_index = []

    new_g_dict = {
        'Na': new_g_Na,
        'Kd': new_g_Kd,
        'CaT': new_g_CaT,
        'CaS': new_g_CaS,
        'KCa': new_g_KCa,
        'A': new_g_A,
        'H': new_g_H,
        'leak': new_g_leak
    }

    # Handle None and NaN values
    for idx, (key, g_value) in enumerate(new_g_dict.items()):
        g_value = np.atleast_1d(g_value)
        if g_value is None or np.isnan(g_value).all():
            none_index.append(idx)
            new_g_dict[key] = np.zeros_like(g_value) if key not in ['CaT', 'CaS'] else np.full_like(g_value, default_g_CaT_for_Ca if key == 'CaT' else default_g_CaS_for_Ca)

    if len(none_index) == 0 or len(none_index) > 3:
        raise ValueError('Number of conductances to compensate should be between 1 and 3')

    target_DICs = np.array(target_DICs, dtype=np.float64)
    not_none_index_target = np.where(~np.isnan(target_DICs))[0]
    target_DICs = target_DICs[not_none_index_target]

    # verify if the number of none index is equal to the number of none in target_DICs
    if len(none_index) != len(not_none_index_target):
        raise ValueError('Number of None in target_DICs should be equal to the number of None in the conductances')

    new_gs = np.asarray([new_g_dict[key] for key in new_g_dict.keys()])
    copy_new_gs = new_gs.copy().T

    if not isinstance(V_th, np.ndarray):
        V_th = np.array([V_th,])

    S_full = sensitivity_matrix(V_th, *new_gs, E_Na, E_K, E_H, E_leak, E_Ca, alpha_Ca, beta_Ca, tau_Ca, tau_f_stg, tau_s_stg, tau_u_stg)

    S_full = S_full.squeeze()
    if S_full.ndim == 2:
        S_full = S_full[np.newaxis, :, :]

    not_none_index = [i for i in range(len(new_g_dict)) if i not in none_index]

    S_random = S_full[:, not_none_index_target, :][:, :, not_none_index]
    S_compensated = S_full[:, not_none_index_target, :][:, :, none_index]

    new_gs = new_gs[not_none_index]

    new_gs = new_gs.T[:, np.newaxis, :]
    A = S_compensated
    result_dot_product = np.sum(S_random * new_gs, axis=2)
    b = target_DICs[np.newaxis, :] - result_dot_product

    # add a dimension to b - I have to introduce this because of the new version of numpy ...
    b = b[:, :, np.newaxis]

    try:
        #print the condition number of the matrix
        #print(np.linalg.cond(A))
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.full((1, len(none_index)), -np.inf)

    x = x.squeeze()
    
    # refill new_gs with the new values
    copy_new_gs[:, none_index] = x
    return copy_new_gs

def generate_spiking_population(n_cells, V_th=-51.0):
    """
    Generate n_cells spiking STG neurons with DICs g_f=-6.2, g_s=4.0, g_u=5.0 at V_th.

    Conductance ranges (gamma distribution): leak [0.007, 0.014], Kd [70, 140],
    CaT [3, 7], CaS [6, 22], KCa [140, 180]; Na, A, H compensated.
    Returns an (n_cells, 8) array of conductances.
    """

    # FROM Fyon et al. 2024
    g_bar_range_Kd2 = [70, 140]
    g_bar_range_CaT2 = [3, 7]
    g_bar_range_CaS2 = [6, 22]
    g_bar_range_KCa2 = [140, 180]
    g_bar_range_leak2 = [0.007, 0.014]

    # g_Na is compensated
    g_bar_range_Kd = [g_bar_range_Kd2[0], g_bar_range_Kd2[1]]
    g_bar_range_CaT = [g_bar_range_CaT2[0], g_bar_range_CaT2[1]]
    g_bar_range_CaS = [g_bar_range_CaS2[0], g_bar_range_CaS2[1]]
    g_bar_range_KCa = [g_bar_range_KCa2[0], g_bar_range_KCa2[1]]
    # g_A is compensated
    # g_H is compensated
    g_bar_range_leak = [g_bar_range_leak2[0], g_bar_range_leak2[1]]

    g_s_spiking = 4.
    g_u_spiking = 5.
    g_f_spiking = -g_s_spiking - 2.2

    PARAMS = get_default_parameters()
    spiking_population = generate_population(n_cells, V_th, g_f_spiking, g_s_spiking, g_u_spiking, g_bar_range_leak, None, g_bar_range_Kd, g_bar_range_CaT, g_bar_range_CaS, g_bar_range_KCa, None, None, params=PARAMS, distribution="gamma")

    return spiking_population

def generate_neuromodulated_population(n_cells, V_th_target, g_s_target, g_u_target, set_to_compensate=None, clean=True, use_fitted_gCaS = lambda g_s, g_u : 34.12021074772369 -2.3296612301271464*g_s, use_fitted_gCaT = lambda g_s, g_u : 24.6 - 5.14 * g_s, iterations=5, d_gCaS=10., d_gCaT=6.0):
    """
    Generate n_cells neurons with target DICs (g_s_target, g_u_target) via neuromodulation of a spiking population.

    set_to_compensate is determined automatically if None. Fitted functions for default CaS/CaT
    linearization points can be overridden. Returns an array of shape (<=n_cells, 8).
    """
    g_f_target = None
    spiking_population = generate_spiking_population(n_cells, V_th_target)

    if set_to_compensate is None:
        set_to_compensate = get_best_set(g_s_target, g_u_target)

    if use_fitted_gCaS:
        d_gCaS = use_fitted_gCaS(g_s_target, g_u_target)

    if use_fitted_gCaT:
        d_gCaT = use_fitted_gCaT(g_s_target, g_u_target)

    neuromodulated_population = modulate_population(spiking_population, V_th_target, g_f_target, g_s_target, g_u_target, get_default_parameters(), set_to_compensate, default_g_CaS_for_Ca=d_gCaS, default_g_CaT_for_Ca=d_gCaT, iterations=iterations)

    if clean:
        neuromodulated_population = neuromodulated_population[np.all(neuromodulated_population >= 0, axis=1)]

    return neuromodulated_population