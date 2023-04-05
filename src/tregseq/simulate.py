import numpy as np
import sympy as sym
from .seq_utils import seq_to_list


def fix_wt(matrix, seq):
    '''
    fix the energy matrix such that the binding energy for the wild type base
    identity at each position is 0

    Args:
        matrix (arr): energy matrix
        seq (str): wild type sequence of binding site

    Returns:
        arr: fixed energy matrix
    '''     

    seq = seq_to_list(seq)
    mat_fixed = []
    for i, row in enumerate(matrix):
        mat_fixed.append([val - row[seq[i]] for val in row])

    return np.asarray(mat_fixed)
    

def get_d_energy(seq, energy_mat, e_wt=0):
    '''
    given an energy matrix and the sequence of the binding site, calculate
    the total binding energy. if the energy matrix is not fixed, we add
    the binding energy to the wild type sequence.

    Args:
        seq (str): sequence of binding site
        energy_mat (arr): energy matrix
        e_wt (int, optional): total binding energy to the wild type binding
        site. Defaults to 0.

    Returns:
        float: total binding energy in kBT units
    '''    

    indices = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    if energy_mat.shape[0] == 4:
        energy_mat = energy_mat.T
    d_energy = 0
    for i,AA in enumerate(seq):
        d_energy += energy_mat[i][indices[AA]]
        
    return d_energy + e_wt


def get_weight(seq, energy_mat, e_wt=0):
    '''
    compute the Boltzmann weight, exp(- d_energy / kBT). note that here
    d_energy is already in kBT units and so we don't have to perform the division.

    Args:
        seq (str): sequence of binding site
        energy_mat (arr): energy matrix
        e_wt (int, optional): total binding energy to the wild type binding
        site. Defaults to 0.

    Returns:
        float: Boltzmann weight.
    '''    

    d_energy = get_d_energy(seq, energy_mat, e_wt=e_wt)
    
    return np.exp(- d_energy)


## computing pbound and fold change using canonical ensemble

def constitutive_pbound(p_seq, p_emat):

    d_energy = get_d_energy(p_seq, p_emat)
    _pbound = np.exp(-d_energy)

    return _pbound / (1 + _pbound)


def simrep_pbound(p_seq, r_seq, p_emat, r_emat, n_p, n_r, n_NS,
                  ep_wt=0, er_wt=0):
    '''
    calculate the probability of binding for a gene with the simple repression
    regulatory architecture

    Args:
        p_seq (str): sequence of the RNAP binding site
        r_seq (str): sequence of the repressor binding site
        p_emat (arr): energy matrix for RNAP
        r_emat (arr): energy matrix for the repressor
        n_p (int): number of RNAPs
        n_r (int): number of repressors
        n_NS (int): number of non-specific binding sites
        ep_wt (int, optional): binding energy to the wild type RNAP binding
        site. Defaults to 0.
        er_wt (int, optional): binding energy to the wild type repressor binding
        site. Defaults to 0.

    Returns:
        float: probability of RNAP binding
    '''    

    w_p = get_weight(p_seq, p_emat, e_wt=ep_wt)
    w_r = get_weight(r_seq, r_emat, e_wt=er_wt)

    return (n_p / n_NS * w_p) / (1 + n_p / n_NS * w_p + n_r / n_NS * w_r)


def simrep_fc(r_seq, r_emat, n_r, n_NS, e_wt=0):
    '''
    calculate fold change in expression levesl when repressors are introduced
    into the system

    Args:
        r_seq (_type_): sequence of the repressor binding site
        r_emat (_type_): energy matrix for the repressor
        n_r (_type_): number of repressors
        n_NS (_type_): number of non-specific binding sites
        e_wt (int, optional): binding energy to the wild type repressor binding
        site. Defaults to 0.

    Returns:
        float: fold change
    '''    

    w_r = get_weight(r_seq, r_emat, e_wt=e_wt)

    return 1 / (1 + n_r / n_NS * w_r)


def simact_pbound(p_seq, a_seq, p_emat, a_emat, e_ap, n_p, n_a, n_NS):
    
    a_weight = get_weight(a_seq, a_emat, n_a, n_NS)
    _f_reg = 1 + a_weight
    f_reg = (1 + a_weight * np.exp(- e_ap)) / _f_reg
    _pbound = 1 + n_NS / (n_p * f_reg) * np.exp(get_d_energy(p_seq, p_emat))
    pbound = 1 / _pbound
    return pbound


## computing occupancy using chemical potential

# constitutive promoter

def constitutive_fugacity(P, M, N, e_S, e_NS, beta=1):
    # final expression is given by the quadratic formula
    x = np.exp(-beta * e_S)
    y = np.exp(-beta * e_NS)

    a = (P - M - N) * x * y
    b = P * (x + y) - M * x - N * y
    l = (- b - np.sqrt(b**2 - 4 * a * P)) / (2 * a)
    
    return l


def constitutive_occupancy(P, M, N, e_S, e_NS, beta=1):

    x = np.exp(-beta * e_S)
    y = np.exp(-beta * e_NS)
    l = constitutive_fugacity(P, M, N, e_S, e_NS, beta=beta)
    occS = l * x / (1 + l * x)
    occNS = l * y / (1 + l * y)

    return occS, occNS

# simple repression

def coeffs_eq1(P, M, N, x_p, y_p, x_r, y_r):
    a = (P - M - N) * x_p * y_p
    b = P * x_p + P * y_p - M * x_p - N * y_p
    c = P * x_r * y_r
    d = P * (x_r + y_r)
    e = P * x_p * y_r + P * x_r * y_p - M * x_p * y_r - N * x_r * y_p
    f = P
    return a, b, c, d, e, f


def coeffs_eq2(R, M, N, x_p, y_p, x_r, y_r):
    A = R * x_p * y_p
    B = R * (x_p + y_p)
    C = (R - M - N) * x_r * y_r
    D = R * x_r + R * y_r - M * x_r - N * y_r
    E = R * x_p * y_r + R * x_r * y_p - M * x_r * y_p - N * x_p * y_r
    F = R
    return A, B, C, D, E, F


def coeffs_combined(a, b, c, d, e, f, A, B, C, D, E, F):
    _a = (e**2 - 2 * a * c) * A - a * e * E + 2 * a**2 * C
    _b = 2 * A * (b * e - a * d) - a * (e * B + b * E) + 2 * a**2 * D
    _c = (b**2 - 2 * a * f) * A - a * b * B + 2 * a** 2 * F
    _d = a * E - A * e
    _e = a * B - A * b
    _f = e**2 - 4 * a * c
    _g = 2 * b * e - 4 * a * d 
    _h = b ** 2 - 4 * a * f
    return _a, _b, _c, _d, _e, _f, _g, _h


def get_final_coeffs(a, b, c, d, e, f, g, h):
    coeff4 = a**2 - d**2 * f
    coeff3 = 2 * a * b - d**2 * g - 2 * d * e * f
    coeff2 = 2 * a * c + b**2 - d**2 * h - 2 * d * e * g - e**2 * f
    coeff1 = 2 * b * c - 2 * d * e * h - e**2 * g
    coeff0 = c**2 - e**2 * h
    return [coeff4, coeff3, coeff2, coeff1, coeff0]


def find_x(y, a, b, c, d, e, f):
    x1 = - b - e * y - np.sqrt((b + e*y)**2 - 4 * a * (c * y**2 + d * y + f))
    x1 = x1 / (2 * a)

    x2 = - b - e * y + np.sqrt((b + e*y)**2 - 4 * a * (c * y**2 + d * y + f))
    x2 = x2 / (2 * a)
    
    x = x1 if x1 >= 0 else x2
    return x


def satisfy_equations(x, y, P, R, M, N, x_p, y_p, x_r, y_r):
    term1 = 1 + x * x_p + y * x_r
    term2 = 1 + x * y_p + y * y_r
    diff1 = abs((term1 * term2 * P) - (M * x_p * x * term2 + N * y_p * x * term1))
    diff2 = abs((term1 * term2 * R) - (M * x_r * y * term2 + N * y_r * y * term1))
    return (diff1 <= 1e-6 and diff2 <= 1e-6)


def simrep_fugacity_hardsolve(P, R, M, N, x_p, y_p, x_r, y_r, beta=1):
    l_p, l_r = sym.symbols('l_p, l_r')

    a = 1 + l_p * x_p + l_r * x_r
    b = 1 + l_p * y_p + l_r * y_r
    eq1 = sym.Eq(a * b * P, M * x_p * l_p * b + N * y_p * l_p * a)
    eq2 = sym.Eq(a * b * R, M * x_r * l_r * b + N * y_r * l_r * a)
    results = sym.solve([eq1, eq2],(l_p, l_r))

    posreal_res = []
    for res in results:
        if sym.im(res[0]) < 1e-6 and sym.im(res[1]) < 1e-6:
            if sym.re(res[0]) > 0 and sym.re(res[1]) > 0:
                posreal_res.append((sym.re(res[0]), sym.re(res[1])))
    if len(posreal_res) == 1:
        return posreal_res[0]
    else:
        print(P, R, M, N, x_p, y_p, x_r, y_r)
        print('No unique solution')


def solve_biquadratic(P, R, M, N, x_p, y_p, x_r, y_r):
    a, b, c, d, e, f = coeffs_eq1(P, M, N, x_p, y_p, x_r, y_r)
    A, B, C, D, E, F = coeffs_eq2(R, M, N, x_p, y_p, x_r, y_r)
    _a, _b, _c, _d, _e, _f, _g, _h = coeffs_combined(a, b, c, d, e, f, A, B, C, D, E, F)
    coeffs = get_final_coeffs(_a, _b, _c, _d, _e, _f, _g, _h)
    roots = np.roots(coeffs)
    
    l_y = []
    has_roots = False
    for r in roots:
        if r >= 0:
            y = r 
            x = find_x(y, a, b, c, d, e, f)
            if satisfy_equations(x, y, P, R, M, N, x_p, y_p, x_r, y_r):
                sol_x = x
                sol_y = y
                has_roots = True
                break

    if not has_roots:
        sol_x, sol_y = simrep_fugacity_hardsolve(P, R, M, N, x_p, y_p, x_r, y_r, beta=1)             

    return sol_x, sol_y


def simrep_occupancy(P, R, M, N, ep_S, ep_NS, er_S, er_NS, beta=1):
    x_p = np.exp(-beta * ep_S)
    y_p = np.exp(-beta * ep_NS)
    x_r = np.exp(-beta * er_S)
    y_r = np.exp(-beta * er_NS)
    l_p, l_r = solve_biquadratic(P, R, M, N, x_p, y_p, x_r, y_r)

    occS_rnap = l_p * x_p / (1 + l_p * x_p + l_r * x_r)
    occNS_rnap = l_p * y_p / (1 + l_p * y_p + l_r * y_r)
    occS_rep = l_r * x_r / (1 + l_p * x_p + l_r * x_r)
    occNS_rep = l_r * y_r / (1 + l_p * y_p + l_r * y_r)
    return l_p, l_r, occS_rnap, occNS_rnap, occS_rep, occNS_rep

