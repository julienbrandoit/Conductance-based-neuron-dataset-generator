import numpy as np


def gsigmoid(V, A, B, C, D):
    """
    Generalized sigmoid: A + B / (1 + exp((V + D) / C)).

    Returns float or np.ndarray.
    """
    return A + B / (1 + np.exp((V + D) / C))


def d_gsigmoid(V, A, B, C, D):
    """
    Derivative of gsigmoid with respect to V.

    Returns float or np.ndarray.
    """
    return -B * np.exp((V + D) / C) / (C * (1 + np.exp((V + D) / C)) ** 2)


def gamma_uniform_mean_std_matching(uniform_a, uniform_b):
    """
    Return the shape (k) and scale (theta) of the gamma distribution whose
    mean and variance match those of Uniform(uniform_a, uniform_b).

    Returns (k, theta).
    """
    p = uniform_a + uniform_b
    q_sq = (uniform_b - uniform_a) ** 2
    k = 3 * p ** 2 / q_sq
    theta = q_sq / (6 * p)
    return k, theta


# == DICs weighting factors ==

def w_factor(V, tau_x, tau_1, tau_2, default=1):
    """
    Compute the log-interpolated weighting factor for a channel with time
    constant tau_x(V) between boundary functions tau_1(V) and tau_2(V).

    Returns np.ndarray the same shape as V.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x(V) > tau_1(V)) & (tau_x(V) <= tau_2(V))
    mask_2 = tau_x(V) > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x(V[mask_1]))) / \
                     (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def w_factor_constant_tau(V, tau_x, tau_1, tau_2, default=1):
    """
    Same as w_factor but tau_x is a scalar constant instead of a callable.

    Returns np.ndarray the same shape as V.
    """
    V = np.asarray(V)
    result = np.ones_like(V) * default
    mask_1 = (tau_x > tau_1(V)) & (tau_x <= tau_2(V))
    mask_2 = tau_x > tau_2(V)
    result[mask_1] = (np.log(tau_2(V[mask_1])) - np.log(tau_x)) / \
                     (np.log(tau_2(V[mask_1])) - np.log(tau_1(V[mask_1])))
    result[mask_2] = 0
    return result


def get_w_factors(V, tau_x, tau_f, tau_s, tau_u):
    """
    Return the weighting factor pair (w_fs, w_su) for a channel whose time
    constant is tau_x(V), relative to boundary functions tau_f, tau_s, tau_u.

    Returns (w_fs, w_su) as np.ndarrays.
    """
    return (w_factor(V, tau_x, tau_f, tau_s, default=1),
            w_factor(V, tau_x, tau_s, tau_u, default=1))


def get_w_factors_constant_tau(V, tau_x, tau_f, tau_s, tau_u):
    """
    Same as get_w_factors but tau_x is a scalar constant.

    Returns (w_fs, w_su) as np.ndarrays.
    """
    return (w_factor_constant_tau(V, tau_x, tau_f, tau_s),
            w_factor_constant_tau(V, tau_x, tau_s, tau_u))

# == Spike detection ==

def find_first_decreasing_zero_bisection(V, f, y_tol=1e-6, x_tol=1e-6, max_iter=1000, verbose=True):
    """
    Find the first x in V where a decreasing function f(x) crosses zero from above.

    Scans consecutive pairs in V for a sign change (f > 0 then f <= 0), then
    refines the crossing with bisection. Returns the crossing voltage as a
    scalar float, or np.nan if no decreasing zero crossing is found.
    """
    V = np.asarray(V, dtype=float)
    f_values = np.array([f(v) for v in V])

    bracket_lo, bracket_hi = None, None
    for i in range(len(V) - 1):
        if f_values[i] > 0 and f_values[i + 1] <= 0:
            bracket_lo, bracket_hi = V[i], V[i + 1]
            f_lo, f_hi = f_values[i], f_values[i + 1]
            break

    if bracket_lo is None:
        if verbose:
            print("find_first_decreasing_zero_bisection: no decreasing zero crossing found.")
        return np.nan

    for _ in range(max_iter):
        mid = (bracket_lo + bracket_hi) / 2.0
        f_mid = f(mid)
        if abs(f_mid) < y_tol or (bracket_hi - bracket_lo) < x_tol:
            return mid
        if f_mid > 0:
            bracket_lo, f_lo = mid, f_mid
        else:
            bracket_hi, f_hi = mid, f_mid

    return (bracket_lo + bracket_hi) / 2.0


def get_spiking_times(t, V, spike_high_threshold=10, spike_low_threshold=0):
    """
    Extract spike onset times from a voltage trace.

    A spike is counted when V crosses above spike_high_threshold and later
    falls below spike_low_threshold. Only onsets that have a corresponding
    end are returned.

    Returns (indices, times) as np.ndarrays; both empty if no spikes found.
    """
    above_threshold = V > spike_high_threshold
    below_threshold = V < spike_low_threshold

    spike_starts = np.where(np.diff(above_threshold.astype(int)) == 1)[0] + 1
    spike_ends = np.where(np.diff(below_threshold.astype(int)) == 1)[0] + 1

    if len(spike_starts) == 0 or len(spike_ends) == 0:
        return np.array([]), np.array([])

    valid_starts = spike_starts[spike_starts < spike_ends[-1]]
    spike_times = t[valid_starts]

    return valid_starts, spike_times
