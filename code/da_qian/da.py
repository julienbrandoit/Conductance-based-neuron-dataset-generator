import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from utils import gsigmoid, d_gsigmoid, get_w_factors, get_w_factors_constant_tau
from utils import find_first_decreasing_zero_bisection
from scipy.integrate import solve_ivp
from scipy.signal import butter, filtfilt
from utils import gamma_uniform_mean_std_matching

# == simulation functions == #


def simulate_individual(args):
    """
    Simulate a single DA neuron from t=0 to T_final with uniform time step dt.

    args : (u0, individual, T_final, dt, params)

    Returns a (2, N) array: row 0 is time, row 1 is voltage.
    """
    u0, individual, T_final, dt, params = args
    t_eval = np.arange(0, T_final, dt)
    return simulate_individual_t_eval((u0, individual, t_eval, params))

def simulate_individual_t_eval(args, ode_func=None, sigma_noise=0.0, cutoff_freq=1000.0):
    """
    Simulate a single DA neuron at specified time points.

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
        individual[4], individual[5], individual[6],
        params['E_Na'], params['E_K'], params['E_Ca'], params['E_leak'],
        params['E_NMDA'], params['Mg']
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

def get_u0(V0):
    """
    Build the initial state vector for the DA model at membrane potential V0.

    Returns an array of 8 state variables initialised at steady state.
    """
    u0 = np.zeros(8)
    u0[0] = V0
    u0[1] = m_inf_Na(V0)
    u0[2] = h_inf_Na(V0)
    u0[3] = n_inf_Kd(V0)
    u0[4] = m_inf_CaL(V0)
    u0[5] = m_inf_CaN(V0)
    u0[6] = a0_ERG(V0)
    u0[7] = 0
    return u0

def get_default_parameters():
    """
    Return the default reversal potentials and Mg concentration for the DA model.

    Returns a dict with keys E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg.
    """
    return {
        'E_Na': 60.,
        'E_K': -85.,
        'E_Ca': 60.,
        'E_leak': -50.,
        'E_NMDA': 0.,
        'Mg': 1.4
    }

def get_default_u0():
    """
    Return the default initial state vector for the DA model (V0 = -90 mV).

    Returns an array of 8 state variables.
    """
    return get_u0(-90.)


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

def get_best_set(g_s, g_u):
    """
    Return the pair of conductances to neuromodulate based on g_s.

    Returns ('ERG', 'CaL') if g_s < 0, else ('ERG', 'Kd').
    """

# == Gating variables functions == #

def m_inf_Na(V):
    """Steady-state activation (m) for the Na channel. Returns float or array."""
    return gsigmoid(V, 0., 1., -9.7264, 30.0907)

def h_inf_Na(V):
    """Steady-state inactivation (h) for the Na channel. Returns float or array."""
    return gsigmoid(V, 0., 1., 10.7665, 54.0289)

def tau_m_Na(V):
    """Time constant for Na activation (m). Returns float or array."""
    return 0.01 + 1.0 / ((-(15.6504 + 0.4043*V)/(np.exp(-19.565 -0.5052*V)-1.0)) + 3.0212*np.exp(-7.4630e-3*V))

def tau_h_Na(V):
    """Time constant for Na inactivation (h). Returns float or array."""
    return 0.4 + 1.0 / ((5.0754e-4*np.exp(-6.3213e-2*V)) + 9.7529*np.exp(0.13442*V))

def n_inf_Kd(V):
    """Steady-state activation (n) for the Kd channel. Returns float or array."""
    return gsigmoid(V, 0., 1., -12., 25.)

def tau_n_Kd(V):
    """Time constant for Kd activation (n). Returns float or array."""
    return gsigmoid(V, 20., -18., -10., 38.)

def m_inf_CaL(V):
    """Steady-state activation (m) for the CaL channel. Returns float or array."""
    return gsigmoid(V, 0., 1., -2., 50.)

def tau_m_CaL(V):
    """Time constant for CaL activation (m). Returns float or array."""
    return gsigmoid(V, 30., -28., -3., 45.)

def m_inf_CaN(V):
    """Steady-state activation (m) for the CaN channel. Returns float or array."""
    return gsigmoid(V, 0., 1., -7., 30.)

def tau_m_CaN(V):
    """Time constant for CaN activation (m). Returns float or array."""
    return gsigmoid(V, 30., -25., -6., 55.)

def a0_ERG(V):
    """Opening rate constant for the ERG channel. Returns float or array."""
    return 0.0036 * np.exp(0.0759*V)

def b0_ERG(V):
    """Closing rate constant for the ERG channel. Returns float or array."""
    return 1.2523e-5 * np.exp(-0.0671*V)

def ai_ERG(V):
    """Inactivation rate constant for the ERG channel. Returns float or array."""
    return 0.1 * np.exp(0.1189*V)

def bi_ERG(V):
    """Recovery-from-inactivation rate constant for the ERG channel. Returns float or array."""
    return 0.003 * np.exp(-0.0733*V)

def NMDA_inf(V, Mg=1.4):
    """Steady-state Mg2+ block factor for the NMDA channel. Returns float or array."""
    return 1 / (1 + Mg * np.exp(-0.08*V) / 10.)

def tau_ERG_constant_function(V, tau=100.):
    """Return a constant array of value tau, same shape as V. Used as the ultra-slow time-scale boundary for DICs."""
    return np.full_like(V, tau)

def o_inf_ERG(V):
    """Steady-state open probability for the ERG channel. Returns float or array."""
    return a0_ERG(V) * bi_ERG(V) / (a0_ERG(V)*(ai_ERG(V) + bi_ERG(V)) + b0_ERG(V)*bi_ERG(V))

# == Derivatives == #

def d_m_inf_Na(V):
    """Derivative of m_inf_Na with respect to V. Returns float or array."""
    return d_gsigmoid(V, 0., 1., -9.7264, 30.0907)

def d_h_inf_Na(V):
    """Derivative of h_inf_Na with respect to V. Returns float or array."""
    return d_gsigmoid(V, 0., 1., 10.7665, 54.0289)

def d_n_inf_Kd(V):
    """Derivative of n_inf_Kd with respect to V. Returns float or array."""
    return d_gsigmoid(V, 0., 1., -12., 25.)

def d_m_inf_CaL(V):
    """Derivative of m_inf_CaL with respect to V. Returns float or array."""
    return d_gsigmoid(V, 0., 1., -2., 50.)

def d_m_inf_CaN(V):
    """Derivative of m_inf_CaN with respect to V. Returns float or array."""
    return d_gsigmoid(V, 0., 1., -7., 30.)

def d_NMDA_inf_dV(V, Mg=1.4):
    """Derivative of NMDA_inf with respect to V. Returns float or array."""
    return 0.08 * Mg * np.exp(-0.08*V) * NMDA_inf(V, Mg)**2 / (10.)

def d_a0_ERG(V):
    """Derivative of a0_ERG with respect to V. Returns float or array."""
    return 0.0036 * 0.0759 * np.exp(0.0759*V)

def d_b0_ERG(V):
    """Derivative of b0_ERG with respect to V. Returns float or array."""
    return 1.2523e-5 * -0.0671 * np.exp(-0.0671*V)

def d_ai_ERG(V):
    """Derivative of ai_ERG with respect to V. Returns float or array."""
    return 0.1 * 0.1189 * np.exp(0.1189*V)

def d_bi_ERG(V):
    """Derivative of bi_ERG with respect to V. Returns float or array."""
    return 0.003 * -0.0733 * np.exp(-0.0733*V)

def d_o_inf_ERG(V):
    """Derivative of o_inf_ERG with respect to V (quotient rule). Returns float or array."""
    f_prime = d_a0_ERG(V) * bi_ERG(V) + a0_ERG(V) * d_bi_ERG(V)
    f = a0_ERG(V) * bi_ERG(V)
    g = a0_ERG(V) * (ai_ERG(V) + bi_ERG(V)) + b0_ERG(V) * bi_ERG(V)
    g_prime = d_a0_ERG(V) * (ai_ERG(V) + bi_ERG(V)) + a0_ERG(V) * (d_ai_ERG(V) + d_bi_ERG(V)) + d_b0_ERG(V) * bi_ERG(V) + b0_ERG(V) * d_bi_ERG(V)
    return (f_prime * g - f * g_prime) / g**2

# == utils == #

def find_V_th_DICs(V, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak,
                    E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg,
                    tau_f_da=tau_m_Na, tau_s_da=tau_m_CaN, tau_u_da=tau_ERG_constant_function, get_I_static=False, normalize=True, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=True):
    """
    Find the threshold voltage V_th where the total DIC (g_t) first decreases through zero.

    Uses bisection over the array V. Returns (V_th, (g_f, g_s, g_u, g_t)) evaluated at V_th.
    Returns (V_th, (nan, nan, nan, nan)) if no zero is found.
    """

    g_t = lambda V_scalar: DICs(np.asarray([V_scalar]), g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, 
                                E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, tau_f_da, tau_s_da, tau_u_da, False, normalize)[3]

    V_th = find_first_decreasing_zero_bisection(V, g_t, y_tol=y_tol, x_tol=x_tol, max_iter=max_iter, verbose=verbose)
    V_th = np.asarray([V_th], dtype=np.float64)

    if V_th is None or np.isnan(V_th):
        return V_th, (np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan), np.atleast_1d(np.nan))

    values = DICs(V_th, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, 
                    E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, tau_f_da, tau_s_da, tau_u_da, get_I_static, normalize)

    return V_th, values
# == ODEs == #

def ODEs(t, u, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg):
    """
    Compute time derivatives of the DA neuron state vector.

    u[0..7]: V, m_Na, h_Na, n_Kd, m_CaL, m_CaN, o_ERG, i_ERG.
    Returns du of the same shape as u.
    """
    du = np.zeros_like(u)
    
    # Extract variables
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    n_Kd = u[3]
    m_CaL = u[4]
    m_CaN = u[5]
    o_ERG, i_ERG = u[6], u[7]
    
    # Compute ionic currents
    I_Na = g_Na * m_Na**3 * h_Na * (V - E_Na)
    I_Kd = g_Kd * n_Kd**3 * (V - E_K)
    I_CaL = g_CaL * m_CaL**2 * (V - E_Ca)
    I_CaN = g_CaN * m_CaN * (V - E_Ca)
    I_ERG = g_ERG * o_ERG * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    I_NMDA = g_NMDA * (V - E_NMDA) * NMDA_inf(V, Mg)
    
    # Voltage equation
    du[0] = (-I_Na - I_Kd - I_CaL - I_CaN - I_ERG - I_leak - I_NMDA)
    
    # Gating variable equations
    du[1] = (m_inf_Na(V) - m_Na) / tau_m_Na(V)
    du[2] = (h_inf_Na(V) - h_Na) / tau_h_Na(V)
    du[3] = (n_inf_Kd(V) - n_Kd) / tau_n_Kd(V)
    du[4] = (m_inf_CaL(V) - m_CaL) / tau_m_CaL(V)
    du[5] = (m_inf_CaN(V) - m_CaN) / tau_m_CaN(V)
    du[6] = a0_ERG(V) * (1 - o_ERG - i_ERG) + bi_ERG(V) * i_ERG - o_ERG * (ai_ERG(V) + b0_ERG(V))
    du[7] = ai_ERG(V) * o_ERG - bi_ERG(V) * i_ERG
    
    return du


def ODEs_with_noisy_current(t, u, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, t_eval, filtered_noise):
    """
    Same as ODEs but with a pre-generated, lowpass-filtered noisy current injected into the voltage equation.

    t_eval and filtered_noise define the noise signal, interpolated at time t.
    Returns du of the same shape as u.
    """
    du = np.zeros_like(u)
    
    # Extract variables
    V = u[0]
    m_Na, h_Na = u[1], u[2]
    n_Kd = u[3]
    m_CaL = u[4]
    m_CaN = u[5]
    o_ERG, i_ERG = u[6], u[7]
    
    # Compute ionic currents
    I_Na = g_Na * m_Na**3 * h_Na * (V - E_Na)
    I_Kd = g_Kd * n_Kd**3 * (V - E_K)
    I_CaL = g_CaL * m_CaL**2 * (V - E_Ca)
    I_CaN = g_CaN * m_CaN * (V - E_Ca)
    I_ERG = g_ERG * o_ERG * (V - E_K)
    I_leak = g_leak * (V - E_leak)
    I_NMDA = g_NMDA * (V - E_NMDA) * NMDA_inf(V, Mg)
    
    # Interpolate pre-generated noise at current time
    I_noise = np.interp(t, t_eval, filtered_noise)
    
    # Voltage equation with noisy current
    du[0] = (-I_Na - I_Kd - I_CaL - I_CaN - I_ERG - I_leak - I_NMDA) + I_noise
    
    # Gating variable equations
    du[1] = (m_inf_Na(V) - m_Na) / tau_m_Na(V)
    du[2] = (h_inf_Na(V) - h_Na) / tau_h_Na(V)
    du[3] = (n_inf_Kd(V) - n_Kd) / tau_n_Kd(V)
    du[4] = (m_inf_CaL(V) - m_CaL) / tau_m_CaL(V)
    du[5] = (m_inf_CaN(V) - m_CaN) / tau_m_CaN(V)
    du[6] = a0_ERG(V) * (1 - o_ERG - i_ERG) + bi_ERG(V) * i_ERG - o_ERG * (ai_ERG(V) + b0_ERG(V))
    du[7] = ai_ERG(V) * o_ERG - bi_ERG(V) * i_ERG
    
    return du

# == DICs related functions == #


def DICs(V, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, 
        E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg,
        tau_f_da=tau_m_Na, tau_s_da=tau_n_Kd, tau_u_da=tau_ERG_constant_function, get_I_static=False, normalize=True):
    """
    Compute the fast (g_f), slow (g_s), ultra-slow (g_u), and total (g_t) dynamic input conductances.

    V and conductances can be scalars or arrays; the output shape follows the same broadcasting rules
    as sensitivity_matrix. If get_I_static is True, also returns I_static.
    """

    if not get_I_static:
        S = sensitivity_matrix(V, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak, 
                               E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, 
                               tau_f_da, tau_s_da, tau_u_da, normalize, get_I_static)
    else:
        S, S_static = sensitivity_matrix(V, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak,
                                            E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg,
                                            tau_f_da, tau_s_da, tau_u_da, normalize, get_I_static)
        
    if S.ndim == 3:
        S = S[np.newaxis, :, :, :]

    m = S.shape[0]

    g_vec = np.array([g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak]).T
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
        V_leak = V - E_leak
        V_NMDA = V - E_NMDA

        V_vec = np.array([V_Na, V_K, V_Ca, V_Ca, V_K, V_NMDA, V_leak])
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

def sensitivity_matrix(V, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, g_leak,
                          E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg,
                          tau_f_da=tau_m_Na, tau_s_da=tau_n_Kd, tau_u_da=tau_ERG_constant_function, normalize=True, get_I_static=False):
    """
    Compute the DIC sensitivity matrix S of shape (3, 7, n) for m=1 or (m, 3, 7, n) for m>1.

    Each entry S[f/s/u, channel, v] gives the contribution of that channel to a given DIC timescale
    at voltage v. If get_I_static is True, also returns the static current matrix S_static.
    """
    V = np.atleast_1d(V)
    g_Na = np.atleast_1d(g_Na)
    g_Kd = np.atleast_1d(g_Kd)
    g_CaL = np.atleast_1d(g_CaL)
    g_CaN = np.atleast_1d(g_CaN)
    g_ERG = np.atleast_1d(g_ERG)
    g_NMDA = np.atleast_1d(g_NMDA)
    g_leak = np.atleast_1d(g_leak)

    m = g_Na.size
    n = V.size

    S_Na = np.zeros((m, 3, n))
    S_Kd = np.zeros((m, 3, n))
    S_CaL = np.zeros((m, 3, n))
    S_CaN = np.zeros((m, 3, n))
    S_ERG = np.zeros((m, 3, n))
    S_NMDA = np.zeros((m, 3, n))
    S_leak = np.zeros((m, 3, n))

    V = np.atleast_2d(V).T

    m_inf_Na_values = m_inf_Na(V)
    h_inf_Na_values = h_inf_Na(V)
    n_inf_Kd_values = n_inf_Kd(V)
    m_inf_CaL_values = m_inf_CaL(V)
    m_inf_CaN_values = m_inf_CaN(V)
    NMAD_inf_values = NMDA_inf(V, Mg)
    o_inf_ERG_values = o_inf_ERG(V)

    d_m_inf_Na_values = d_m_inf_Na(V)
    d_h_inf_Na_values = d_h_inf_Na(V)
    d_n_inf_Kd_values = d_n_inf_Kd(V)
    d_m_inf_CaL_values = d_m_inf_CaL(V)
    d_m_inf_CaN_values = d_m_inf_CaN(V)
    d_NMDA_inf_dV_values = d_NMDA_inf_dV(V, Mg)
    d_o_inf_ERG_values = d_o_inf_ERG(V)

    S_Na[:, 0, :] += (m_inf_Na_values**3 * h_inf_Na_values).T
    S_Kd[:, 0, :] += (n_inf_Kd_values**3).T
    S_CaL[:, 0, :] += (m_inf_CaL_values**2).T
    S_CaN[:, 0, :] += m_inf_CaN_values.T
    S_ERG[:, 0, :] += o_inf_ERG_values.T
    S_NMDA[:, 0, :] += NMAD_inf_values.T
    S_leak[:, 0, :] += 1.

    if get_I_static:
        S_Na_static = S_Na[:, 0, :].copy()
        S_Kd_static = S_Kd[:, 0, :].copy()
        S_CaL_static = S_CaL[:, 0, :].copy()
        S_CaN_static = S_CaN[:, 0, :].copy()
        S_ERG_static = S_ERG[:, 0, :].copy()
        S_NMDA_static = S_NMDA[:, 0, :].copy()
        S_leak_static = S_leak[:, 0, :].copy()

    dV_dot_dm_Na = -3 * m_inf_Na_values**2 * h_inf_Na_values * d_m_inf_Na_values * (V - E_Na)
    dV_dot_dh_Na = - m_inf_Na_values**3 * d_h_inf_Na_values * (V - E_Na)
    dV_dot_dn_Kd = -3 * n_inf_Kd_values**2 * d_n_inf_Kd_values * (V - E_K)
    dV_dot_dm_CaL = -2 * m_inf_CaL_values * d_m_inf_CaL_values * (V - E_Ca)
    dV_dot_dm_CaN = -1 * d_m_inf_CaN_values * (V - E_Ca)
    dV_dot_do_ERG = -1 * d_o_inf_ERG_values * (V - E_K)
    dV_dot_dNMDA = -1 * d_NMDA_inf_dV_values * (V - E_NMDA)

    w_fs_m_Na, w_su_m_Na = get_w_factors(V, tau_m_Na, tau_f_da, tau_s_da, tau_u_da)
    w_fs_h_Na, w_su_h_Na = get_w_factors(V, tau_h_Na, tau_f_da, tau_s_da, tau_u_da)
    w_fs_n_Kd, w_su_n_Kd = get_w_factors(V, tau_n_Kd, tau_f_da, tau_s_da, tau_u_da)
    w_fs_m_CaL, w_su_m_CaL = get_w_factors(V, tau_m_CaL, tau_f_da, tau_s_da, tau_u_da)
    w_fs_m_CaN, w_su_m_CaN = get_w_factors(V, tau_m_CaN, tau_f_da, tau_s_da, tau_u_da)
    w_fs_o_ERG, w_su_o_ERG = get_w_factors(V, tau_ERG_constant_function, tau_f_da, tau_s_da, tau_u_da)
    #w_fs_NMDA, w_su_NMDA = get_w_factors_constant_tau(V, 1e-12, tau_f_da, tau_s_da, tau_u_da) # TO BE CHANGED ?

    w_fs_o_ERG = 0.
    w_su_o_ERG = 0.

    w_fs_NMDA = 1. # fast time scale
    w_su_NMDA = 1.

    S_Na[:, 0, :] += (- w_fs_m_Na * dV_dot_dm_Na - w_fs_h_Na * dV_dot_dh_Na).T
    S_Kd[:, 0, :] += (- w_fs_n_Kd * dV_dot_dn_Kd).T
    S_CaL[:, 0, :] += (- w_fs_m_CaL * dV_dot_dm_CaL).T
    S_CaN[:, 0, :] += (- w_fs_m_CaN * dV_dot_dm_CaN).T
    S_ERG[:, 0, :] += (- w_fs_o_ERG * dV_dot_do_ERG).T
    S_NMDA[:, 0, :] += (- w_fs_NMDA * dV_dot_dNMDA).T

    S_Na[:, 1, :] += (- (w_su_m_Na - w_fs_m_Na) * dV_dot_dm_Na - (w_su_h_Na - w_fs_h_Na) * dV_dot_dh_Na).T
    S_Kd[:, 1, :] += (- (w_su_n_Kd - w_fs_n_Kd) * dV_dot_dn_Kd).T
    S_CaL[:, 1, :] += (- (w_su_m_CaL - w_fs_m_CaL) * dV_dot_dm_CaL).T
    S_CaN[:, 1, :] += (- (w_su_m_CaN - w_fs_m_CaN) * dV_dot_dm_CaN).T
    S_ERG[:, 1, :] += (- (w_su_o_ERG - w_fs_o_ERG) * dV_dot_do_ERG).T
    S_NMDA[:, 1, :] += (- (w_su_NMDA - w_fs_NMDA) * dV_dot_dNMDA).T

    S_Na[:, 2, :] += (- (1 - w_su_m_Na) * dV_dot_dm_Na - (1 - w_su_h_Na) * dV_dot_dh_Na).T
    S_Kd[:, 2, :] += (- (1 - w_su_n_Kd) * dV_dot_dn_Kd).T
    S_CaL[:, 2, :] += (- (1 - w_su_m_CaL) * dV_dot_dm_CaL).T
    S_CaN[:, 2, :] += (- (1 - w_su_m_CaN) * dV_dot_dm_CaN).T
    S_ERG[:, 2, :] += (- (1 - w_su_o_ERG) * dV_dot_do_ERG).T
    S_NMDA[:, 2, :] += (- (1 - w_su_NMDA) * dV_dot_dNMDA).T

    if normalize:
        S_Na /= g_leak[:, np.newaxis, np.newaxis]
        S_Kd /= g_leak[:, np.newaxis, np.newaxis]
        S_CaL /= g_leak[:, np.newaxis, np.newaxis]
        S_CaN /= g_leak[:, np.newaxis, np.newaxis]
        S_ERG /= g_leak[:, np.newaxis, np.newaxis]
        S_NMDA /= g_leak[:, np.newaxis, np.newaxis]
        S_leak /= g_leak[:, np.newaxis, np.newaxis]

    S_Na = S_Na[:, :, np.newaxis, :]
    S_Kd = S_Kd[:, :, np.newaxis, :]
    S_CaL = S_CaL[:, :, np.newaxis, :]
    S_CaN = S_CaN[:, :, np.newaxis, :]
    S_ERG = S_ERG[:, :, np.newaxis, :]
    S_NMDA = S_NMDA[:, :, np.newaxis, :]
    S_leak = S_leak[:, :, np.newaxis, :]

    if not get_I_static:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaL[0], S_CaN[0], S_ERG[0], S_NMDA[0], S_leak[0]), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaL, S_CaN, S_ERG, S_NMDA, S_leak), axis=2)
    else:
        if m == 1:
            return np.concatenate((S_Na[0], S_Kd[0], S_CaL[0], S_CaN[0], S_ERG[0], S_NMDA[0], S_leak[0]), axis=1), np.stack((S_Na_static[0], S_Kd_static[0], S_CaL_static[0], S_CaN_static[0], S_ERG_static[0], S_NMDA_static[0], S_leak_static[0]), axis=1)
        else:
            return np.concatenate((S_Na, S_Kd, S_CaL, S_CaN, S_ERG, S_NMDA, S_leak), axis=2), np.stack((S_Na_static, S_Kd_static, S_CaL_static, S_CaN_static, S_ERG_static, S_NMDA_static, S_leak_static), axis=1)
        
# == Compensation algorithms and generation functions == #
def generate_population(n_cells, V_th, g_f_target, g_s_target, g_u_target, g_bar_range_leak, g_bar_range_Na, g_bar_range_Kd, g_bar_range_CaL, g_bar_range_CaN, g_bar_range_ERG, g_bar_range_NMDA, params, distribution='uniform', normalize_by_leak=True):
    """
    Generate n_cells neurons whose DICs (g_f, g_s, g_u) match the given targets at V_th.

    Conductance ranges can be None to mark that channel as compensated.
    distribution: 'uniform' or 'gamma'. If normalize_by_leak, all conductances are scaled
    proportionally to the sampled leak value.

    Returns an (n_cells, 7) array of conductances.
    """
    g_Na = np.random.uniform(g_bar_range_Na[0], g_bar_range_Na[1], n_cells) if g_bar_range_Na is not None else np.full(n_cells, np.nan)
    g_Kd = np.random.uniform(g_bar_range_Kd[0], g_bar_range_Kd[1], n_cells) if g_bar_range_Kd is not None else np.full(n_cells, np.nan)
    g_CaL = np.random.uniform(g_bar_range_CaL[0], g_bar_range_CaL[1], n_cells) if g_bar_range_CaL is not None else np.full(n_cells, np.nan)
    g_CaN = np.random.uniform(g_bar_range_CaN[0], g_bar_range_CaN[1], n_cells) if g_bar_range_CaN is not None else np.full(n_cells, np.nan)
    g_ERG = np.random.uniform(g_bar_range_ERG[0], g_bar_range_ERG[1], n_cells) if g_bar_range_ERG is not None else np.full(n_cells, np.nan)
    g_NMDA = np.random.uniform(g_bar_range_NMDA[0], g_bar_range_NMDA[1], n_cells) if g_bar_range_NMDA is not None else np.full(n_cells, np.nan)
    
    if distribution == 'uniform':
        mean_leak = (g_bar_range_leak[0] + g_bar_range_leak[1]) / 2
        g_leak = np.random.uniform(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
    elif distribution == 'gamma':
        g_bar_range_leak = gamma_uniform_mean_std_matching(*g_bar_range_leak)
        mean_leak = g_bar_range_leak[0] * g_bar_range_leak[1]
        g_leak = np.random.gamma(g_bar_range_leak[0], g_bar_range_leak[1], n_cells)
    else:
        raise ValueError('Invalid distribution type! Please use either "uniform" or "gamma".')

    if normalize_by_leak:
        f = g_leak / mean_leak
        g_Na *= f
        g_Kd *= f
        g_CaL *= f
        g_CaN *= f
        g_ERG *= f
        g_NMDA *= f

    x = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], g_leak, g_Na, g_Kd, g_CaL, g_CaN, g_ERG, g_NMDA, params['E_Na'], params['E_K'], params['E_Ca'], params['E_leak'], params['E_NMDA'], params['Mg'])
    
    return x

def modulate_population(population, V_th, g_f_target, g_s_target, g_u_target, params, set_to_compensate):
    """
    Adjust the conductances in population so that the DICs match the given targets at V_th.

    Targets that are None/NaN are left free; the corresponding conductances listed in
    set_to_compensate are solved for. The number of free targets must equal len(set_to_compensate).

    Returns an (n_cells, 7) array of modulated conductances.
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

    if 'Na' in set_to_compensate:
        population[:, 0] = np.nan
    if 'Kd' in set_to_compensate:
        population[:, 1] = np.nan
    if 'CaL' in set_to_compensate:
        population[:, 2] = np.nan
    if 'CaN' in set_to_compensate:
        population[:, 3] = np.nan
    if 'ERG' in set_to_compensate:
        population[:, 4] = np.nan
    if 'NMDA' in set_to_compensate:
        population[:, 5] = np.nan

    population = general_compensation_algorithm(V_th, [g_f_target, g_s_target, g_u_target], population[:, 6], population[:, 0], population[:, 1], population[:, 2], population[:, 3], population[:, 4], population[:, 5], params['E_Na'], params['E_K'], params['E_Ca'], params['E_leak'], params['E_NMDA'], params['Mg'])

    return population

def general_compensation_algorithm(V_th, target_DICs, new_g_leak, new_g_Na, new_g_Kd, new_g_CaL, new_g_CaN, new_g_ERG, new_g_NMDA, E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, tau_f_da=tau_m_Na, tau_s_da=tau_n_Kd, tau_u_da=tau_ERG_constant_function):
    """
    Solve for the conductances left as None/NaN so that the DICs match target_DICs at V_th.

    target_DICs is a length-3 list [g_f, g_s, g_u]; entries that are None/NaN are the target DICs
    to reach, while the corresponding conductances in new_g_* must also be None/NaN.
    The number of free conductances must equal the number of non-NaN target DICs.

    Returns an (n_cells, 7) array with the compensated conductances filled in.
    """
    none_index = []

    new_g_dict = {
        'Na': new_g_Na,
        'Kd': new_g_Kd,
        'CaL': new_g_CaL,
        'CaN': new_g_CaN,
        'ERG': new_g_ERG,
        'NMDA': new_g_NMDA,
        'leak': new_g_leak
    }

    # Handle None and NaN values
    for idx, (key, g_value) in enumerate(new_g_dict.items()):
        g_value = np.atleast_1d(g_value)
        if g_value is None or np.isnan(g_value).all():
            none_index.append(idx)
            new_g_dict[key] = np.zeros_like(g_value)

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

    S_full = sensitivity_matrix(V_th, *new_gs, E_Na, E_K, E_Ca, E_leak, E_NMDA, Mg, tau_f_da, tau_s_da, tau_u_da)

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
        x = np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        x = np.full((1, len(none_index)), -np.inf)

    x = x.squeeze()
    
    # refill new_gs with the new values
    copy_new_gs[:, none_index] = x
    return copy_new_gs


def generate_spiking_population(n_cells, V_th=-55.5):
    """
    Generate n_cells spiking DA neurons with DICs g_f=-12.95, g_s=0.5, g_u=5.0 at V_th.

    Conductance ranges (gamma distribution): leak [0.0087, 0.017], Kd [6, 10],
    CaL [0.015, 0.075], NMDA [0.012, 0.012]; Na, CaN, ERG compensated.

    Returns an (n_cells, 7) array of conductances.
    """

    g_bar_range_Kd = [6, 10]
    g_bar_range_CaL = [0.015, 0.075]
    g_bar_range_NMDA = [0.012, 0.012]
    g_bar_range_leak = [0.0087, 0.017]

    g_s_spiking = 0.5
    g_u_spiking = 5.
    g_f_spiking = -3.9*g_s_spiking - 11.

    PARAMS = get_default_parameters()
    spiking_population = generate_population(n_cells, V_th, g_f_spiking, g_s_spiking, g_u_spiking, g_bar_range_leak, None, g_bar_range_Kd, g_bar_range_CaL, None, None, g_bar_range_NMDA, PARAMS, distribution='gamma', normalize_by_leak=True)

    return spiking_population

def generate_neuromodulated_population(n_cells, V_th_target, g_s_target, g_u_target, set_to_compensate=None, clean=True):
    """
    Generate n_cells neurons with DICs (g_s, g_u) via neuromodulation of a spiking population.

    Starts from generate_spiking_population, then calls modulate_population.
    set_to_compensate defaults to get_best_set(g_s_target, g_u_target).
    If clean is True, rows with any negative conductance are removed.

    Returns an array of shape (<=n_cells, 7).
    """
    g_f_target = None
    spiking_population = generate_spiking_population(n_cells, V_th_target)

    if set_to_compensate is None:
        set_to_compensate = get_best_set(g_s_target, g_u_target)

    neuromodulated_population = modulate_population(spiking_population, V_th_target, g_f_target, g_s_target, g_u_target, get_default_parameters(), set_to_compensate)

    if clean:
        neuromodulated_population = neuromodulated_population[np.all(neuromodulated_population >= 0, axis=1)]

    return neuromodulated_population
