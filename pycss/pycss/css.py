import numpy as np
from numba import njit, prange
from time import time
from datetime import timedelta
import pycss.sim as sim


def css_varpro(imDataParams, options):

    imDataParams = imDataParams.copy()
    options = options.copy()

    verbose = options.get('verbose', True)
    # if verbose:
    #     print('Varpro: option =\n', options)

    mask = options['mask']

    TE_s = imDataParams['TE_s'].ravel()
    signal = imDataParams['signal'].astype(np.complex128)
    sz = signal.shape[:3]
    Sig = signal.reshape((np.prod(sz), signal.shape[-1]))[mask.ravel(), :]
    assert Sig.shape[0] == options['Pm0'].shape[0]

    C_rho = options['Cm']
    C_phi = options['Cp']
    C_w = options['Cf']
    C_R = options['Cr']
    pos_constraint = options['pos_constraint']

    if verbose:
        print('Varpro: Start loop over all voxels.', end=' ')
    t = time()
    Pme, Resnorm, Iterations = map_varpro(TE_s,
                                          Sig,
                                          options['Pm0'],
                                          C_rho, C_phi, C_w, C_R,
                                          options['tol'],
                                          int(options['itermax']),
                                          pos_constraint)

    if pos_constraint:
        Pme[:, 3:] = np.log(Pme[:, 3:])

    elapsed_s = time() - t
    if verbose:
        print(f'Done. Elapsed time: {str(timedelta(seconds=elapsed_s)):.10}')

    maps = construct_param_maps(Pme, [C_rho, C_phi, C_w, C_R], mask)

    output_dict = {'params_matrices': Pme,
                   'param_maps': construct_param_maps(Pme,
                                                      [C_rho, C_phi, C_w, C_R], mask),
                   'resnorm': construct_map(Resnorm, mask),
                   'iterations': construct_map(Iterations, mask),
                   'elapsed_time_s': elapsed_s}

    if options.get('uncertainty_quant', False):
        output_dict.update(compute_FIMmaps(TE_s, Pme, C_rho, C_phi, C_w, C_R, mask))

    return output_dict


def construct_param_maps(Pme, constraints_matrices, mask):
    return [construct_map(vals, mask)
            for vals in extract_variables(Pme, constraints_matrices)]


def extract_variables(Pm, constraints_matrices):
    pmn = Pm.copy()
    Cs = constraints_matrices

    vals = []
    for iptype in range(pmn.shape[-1]):
        C = Cs[iptype]
        for ivar in range(C.shape[-1]):
            isNonzero = list(C[:, ivar] != 0)
            pmn[..., isNonzero, iptype] /= C[..., isNonzero, ivar]

        indVars = np.where(np.diag(C.T.dot(C)))[0]
        for ind in indVars:
            vals.append(np.squeeze(pmn[..., ind, iptype]))

    return vals


def construct_map(vals1D, mask):
    map_ = np.zeros(mask.shape, dtype=vals1D.dtype).ravel()
    map_[mask.ravel()] = vals1D
    return map_.reshape(mask.shape)


def compute_FIMmaps(TE_s, Pm, C_rho, C_phi, C_w, C_R, mask):
    CRLBs, NSAs, Invariants, FIMs, FIMinvs = \
        sim.compute_FIMparams(TE_s, Pm, C_rho, C_phi, C_w, C_R)
    n = CRLBs.shape[-1]
    return {'CRLBs': np.array([construct_map(CRLBs[:, i], mask)
                                for i in range(n)]),
            'NSAs': np.array([construct_map(NSAs[:, i], mask)
                               for i in range(n)]),
            'trFIM': construct_map(Invariants[:, 0], mask),
            'trInvFIM': construct_map(Invariants[:, 1], mask),
            'detFIM': construct_map(Invariants[:, 2], mask)}


@njit(parallel=True)
def map_varpro(TE_s, Sig, Pm, C_rho, C_phi, C_w, C_R, tol, itermax, pos_constraint=None):
    """map varpro solver for all voxels list of MR signals

    :param TE_s: 1d np.array, with echo times in seconds
    :param Sig: 2d np.ndarray,  measured complex MR signal for all voxels
                shape # voxel x # TE's
    :param Pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :param C_rho: 2d np.ndarray, constraints matrix for magnitudes
    :param C_phi: 2d np.ndarray, constraints matrix for phases
    :param C_w: 2d np.ndarray, constraints matrix for resonance frequencies
    :param C_R: 2d np.ndarray, constraints matrix for relaxation rates
    :param tol: float, desired precision on the residual
    :param itermax: int, maximal number of iterations
    :returns: parameter matrix array, residual, # of iterations
    :rtype: tuple of three 1d numpy.arrays with dtypes: float, float, int

    """
    nTE = len(TE_s)
    nVoxel = Sig.shape[0]
    assert Sig.shape[-1] == nTE
    assert Pm.shape[0] == nVoxel

    # initialize output
    Iterations = np.zeros(nVoxel)
    Resnorm = np.zeros(nVoxel)
    for i in prange(nVoxel):
        # voxel input
        sig = Sig[i, ...].reshape(Sig.shape[1:])
        pm = Pm[i, ...].reshape(Pm.shape[1:])

        # apply func
        pm, resnorm, iterations = varpro(TE_s, sig, pm, C_rho, C_phi, C_w, C_R,
                                         tol, itermax, pos_constraint)

        # assign voxel output to array result
        Pm[i, ...] = pm
        Iterations[i] = iterations
        Resnorm[i] = resnorm

    return Pm, Resnorm, Iterations


@njit
def varpro(TE_s, sig, pm, C_rho, C_phi, C_w, C_R, tol, itermax, pos_constraint=None):
    """solve chemical species separation by VARPRO,
    the variable projection method

    :param TE_s: 1d np.array, with echo times in seconds
    :param sig: 1d np.array, measured complex MR signal
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :param C_rho: 2d np.ndarray, constraints matrix for magnitudes
    :param C_phi: 2d np.ndarray, constraints matrix for phases
    :param C_w: 2d np.ndarray, constraints matrix for resonance frequencies
    :param C_R: 2d np.ndarray, constraints matrix for relaxation rates
    :param tol: float, desired precision on the residual
    :param itermax: int, maximal number of iterations
    :returns: parameter matrix, residual, # of iterations
    :rtype: tuple (float, float, int)

    """
    C_rho = C_rho.astype(np.float64)
    C_rho_c = C_rho.astype(np.complex128)
    C_phi = C_phi.astype(np.float64)
    C_phi_c = np.zeros_like(C_rho_c)
    C_phi_c[:, :C_phi.shape[-1]] = C_phi.astype(np.complex128)
    C_w = C_w.astype(np.float64)
    C_R = C_R.astype(np.float64)
    M, Ntot, Nm, Np, Nf, Nr = get_paramsCount(C_rho, C_phi, C_w, C_R)

    eps = 2 * tol
    i = 0
    cond_thres = 1e20

    while eps > tol and i < itermax:
        # update linear parameters
        A = get_Amatrix(TE_s, pm, pos_constraint)
        if np.isnan(A).any() or np.isinf(A).any():
            break
        if np.linalg.cond(A) > cond_thres:
            break
        # print(np.dot(A, C_rho_c).dtypem sig.dtype)
        rho = np.linalg.lstsq(np.dot(A, C_rho_c), sig, rcond=-1)[0]

        Cm_rho = np.dot(C_rho_c, rho)
        Cm_phi = np.dot(C_phi_c, rho)
        pm[:, 0] = np.abs(Cm_rho)
        pm[:, 1] = np.angle(Cm_phi)

        # update nonlinear parameters
        r = sig - build_signal(TE_s, pm, pos_constraint)
        J = get_Jacobian(TE_s, pm, C_rho, C_phi, C_w, C_R, pos_constraint)
        rr = np.concatenate((r.real, r.imag), axis=0)
        JJ = np.concatenate((J.real, J.imag), axis=0)
        if np.isnan(JJ).any() or np.isinf(JJ).any():
            break
        if np.linalg.cond(JJ) > cond_thres:
            break
        updates = np.linalg.lstsq(JJ, rr, rcond=-1)[0]
        eps = np.linalg.norm(updates)  # for convergence criterium

        pm_update = np.zeros_like(pm)
        pm_update[:, 1] = np.dot(C_phi, updates[Nm:Nm+Np])
        pm_update[:, 2] = np.dot(C_w, updates[Nm+Np:Nm+Np+Nf])
        pm_update[:, 3] = np.dot(C_R, updates[Nm+Np+Nf:Nm+Np+Nf+Nr])
        pm += pm_update

        i += 1

    resnorm = np.linalg.norm(sig - build_signal(TE_s, pm, pos_constraint)) # residual norm

    return pm, resnorm, i


# def varpro_iter(TE_s, sig, pm, C_rho, C_phi, C_w, C_R, tol, itermax):
#     """solve chemical species separation (IDEAL or VARPRO) by tracking

#     :param TE_s: 1d np.array, with echo times in seconds
#     :param sig: 1d np.array, measured complex MR signal
#     :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
#     :param C_rho: 2d np.ndarray, constraints matrix for magnitudes
#     :param C_phi: 2d np.ndarray, constraints matrix for phases
#     :param C_w: 2d np.ndarray, constraints matrix for resonance frequencies
#     :param C_R: 2d np.ndarray, constraints matrix for relaxation rates
#     :param tol: float, desired precision on the residual
#     :param itermax: int, maximal number of iterations
#     :param varpro: bool, wether VARPRO or IDEAL is used
#     :returns: parameter matrix, residual, # of iterations
#     :rtype: tuple (float, float, int)

#     """
#     C_rho = C_rho.astype(np.float64)
#     C_rho_c = C_rho.astype(np.complex128)
#     C_phi = C_phi.astype(np.float64)
#     C_w = C_w.astype(np.float64)
#     C_R = C_R.astype(np.float64)
#     M, Ntot, Nm, Np, Nf, Nr = get_paramsCount(C_rho, C_phi, C_w, C_R)

#     resnorm_iter = np.zeros(itermax)
#     eps_iter = np.zeros(itermax)
#     pm_iter = np.zeros((itermax+1, C_rho.shape[0], 4))
#     pm_iter[0, :, :] = pm

#     eps = 2 * tol
#     i = 1
#     cond_thres = 1e6

#     while eps > tol and i < itermax:
#         # update linear parameters

#         A = get_Amatrix(TE_s, pm)
#         if np.isnan(A).any() or np.isinf(A).any():
#             break
#         if np.linalg.cond(A) > cond_thres:
#             break
#         rho = np.linalg.lstsq(np.dot(A, C_rho_c), sig, rcond=-1)[0]
#         Cm_rho = np.dot(C_rho_c, rho)
#         pm[:, 0] = np.abs(Cm_rho)
#         pm[:, 1] = np.angle(Cm_rho)

#         # update nonlinear parameters
#         r = sig - build_signal(TE_s, pm)
#         J = get_Jacobian(TE_s, pm, C_rho, C_phi, C_w, C_R)
#         rr = np.concatenate((r.real, r.imag), axis=0)
#         JJ = np.concatenate((J.real, J.imag), axis=0)
#         if np.isnan(JJ).any() or np.isinf(JJ).any():
#             break
#         if np.linalg.cond(JJ) > cond_thres:
#             break
#         updates = np.linalg.lstsq(JJ, rr, rcond=-1)[0]
#         eps = np.linalg.norm(updates)  # for convergence criterium

#         pm_update = np.zeros_like(pm)
#         pm_update[:, 1] = np.dot(C_phi, updates[Nm:Nm+Np])
#         pm_update[:, 2] = np.dot(C_w, updates[Nm+Np:Nm+Np+Nf])
#         pm_update[:, 3] = np.dot(C_R, updates[Nm+Np+Nf:Nm+Np+Nf+Nr])
#         pm += pm_update

#         resnorm_iter[i] = np.linalg.norm(sig - build_signal(TE_s, pm))
#         eps = np.linalg.norm(updates)
#         eps_iter[i] = eps
#         pm_iter[i, :, :] = pm

#         i += 1

#     return pm_iter[:i], resnorm_iter[:i], eps_iter[:i]


@njit
def get_Amatrix(TE_s, pm, pos_constraint=None):
    """set up A matrix from voxel signal equation
    of dimension #echoes x #species

    :param TE_s: 1d np.array, with echo times in seconds
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :returns: A matrix
    :rtype: 2d np.ndarray

    """
    A = np.zeros((len(TE_s), pm.shape[0]), dtype=np.complex128)
    if pos_constraint:
        for i in range(A.shape[1]):
            A[:, i] = np.exp((2j * np.pi * pm[i, 2] - np.exp(pm[i, 3])) * TE_s)
    else:
        for i in range(A.shape[1]):
            A[:, i] = np.exp((2j * np.pi * pm[i, 2] - pm[i, 3]) * TE_s)

    return A


@njit
def build_signal(TE_s, pm, pos_constraint=None):
    """forward simulate MR signal at echo times TE_s

    :param TE_s: 1d np.array, with echo times in seconds
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :returns:
    :rtype: complex 1d np.ndarray

    """
    rho = pm[:, 0] * np.exp(1.j * pm[:, 1])
    A = get_Amatrix(TE_s, pm, pos_constraint)
    return np.dot(A, rho.astype(A.dtype))


@njit
def get_Jacobian(TE_s, pm, C_rho, C_phi, C_w, C_R, pos_constraint=None):
    """compute Jacobian

    :param TE_s: 1d np.array, with echo times in seconds
    :param pm: 2d np.ndarray, parameter matrix of dimension #species x 4
    :param C_rho: 2d np.ndarray, constraints matrix for magnitudes
    :param C_phi: 2d np.ndarray, constraints matrix for phases
    :param C_w: 2d np.ndarray, constraints matrix for resonance frequencies
    :param C_R: 2d np.ndarray, constraints matrix for relaxation rates
    :returns: Jacobian
    :rtype: complex 2d np.array

    """
    A = get_Amatrix(TE_s, pm, pos_constraint)
    P = np.diag(np.exp(1.j * pm[:, 1]))
    D = np.diag(pm[:, 0] * np.exp(1.j * pm[:, 1]))
    AD = np.dot(A, D.astype(A.dtype))
    T = np.diag(TE_s).astype(A.dtype)
    Jm = np.dot(np.dot(A, P), C_rho.astype(A.dtype))
    Jp = 1.j * np.dot(AD, C_phi.astype(A.dtype))
    Jf = 2.j * np.pi * np.dot(T, np.dot(AD, C_w.astype(AD.dtype)))
    Jr = -np.dot(T, np.dot(AD, C_R.astype(AD.dtype)))
    # return np.c_[Jm, Jp, Jf, Jr]  # not possible for njit
    Jm = np.atleast_2d(Jm.T).T  # hack to get same number of dimensions
    Jp = np.atleast_2d(Jp.T).T  # needed for hstack
    Jf = np.atleast_2d(Jf.T).T
    Jr = np.atleast_2d(Jr.T).T
    J = np.hstack((Jm, Jp, Jf, Jr))
    # J = J[np.all(~(J==0), axis=0), :]
    # J = J[:, (~(J==0), axis=1)]
    return J


@njit
def add_noise(signal, SNR):
    """add noise with SNR to signal

    :param signal: complex 1d np.array, sampled MR signal at echo times TE_s
    :param SNR: float
    :returns: noisy signal
    :rtype: complex 1d np.array, sampled MR signal at echo times TE_s w/ noise

    """
    sz = signal.shape
    noise_r = np.random.normal(0, 1/SNR, sz)
    noise_i = np.random.normal(0, 1/SNR, sz)
    return (signal.real + noise_r) + 1j * (signal.imag + noise_i)


@njit
def get_paramsCount(C_rho, C_phi, C_w, C_R):
    """count how many free variables are ought to be solved

    :param C_rho: 2d np.ndarray, constraints matrix for magnitudes
    :param C_phi: 2d np.ndarray, constraints matrix for phases
    :param C_w: 2d np.ndarray, constraints matrix for resonance frequencies
    :param C_R: 2d np.ndarray, constraints matrix for relaxation rates
    :returns: # all variable, # magnitudes, # phases, # freqs, # relaxations
    :rtype: tuple of ints

    """
    assert C_rho.shape[0] == C_phi.shape[0] == C_w.shape[0] == C_R.shape[0]
    Nm = C_rho.shape[-1]
    Np = C_phi.shape[-1]
    Nf = C_w.shape[-1]
    Nr = C_R.shape[1]
    M = C_rho.shape[0]
    Ntot = Nm + Np + Nf + Nr
    return M, Ntot, Nm, Np, Nf, Nr


def get_params(pm, C_rho, C_phi, C_w, C_R):
    """FIXME extract variable parameters from paramter matrix based on the
    constraint matrices

    :param pm:
    :param C_rho:
    :param C_phi:
    :param C_w:
    :param C_R:
    :returns:
    :rtype: 1d np.array

    """
    _, Ntot, Nm, Np, Nf, Nr = get_paramsCount(C_rho, C_phi, C_w, C_R)
    return np.r_[pm[0:Nm, 0],
                 pm[Nm:Nm+Np, 1],
                 pm[Nm+Np:Nm+Np+Nf, 2],
                 pm[Nm+Np+Nf:Nm+Np+Nf+Nr, 3]]


def set_params(beta, pm, C_rho, C_phi, C_w, C_R):
    """FIXME according to constraint matrices C[mpfr]
    set variables beta in the parameter matrix pm

    :param beta:
    :param pm:
    :param C_rho:
    :param C_phi:
    :param C_w:
    :param C_R:
    :returns:
    :rtype:

    """
    deshielding_Hz = np.concatenate(([0], pm[1:, 2] - pm[0, 2]))
    _, Ntot, Nm, Np, Nf, Nr = get_paramsCount(C_rho, C_phi, C_w, C_R)
    pm[:, 0] = np.dot(C_rho, beta[0:Nm])
    pm[:, 1] = np.dot(C_phi, beta[Nm:Nm+Np])
    pm[:, 2] = np.dot(C_w, *beta[Nm+Np:Nm+Np+Nf]) + deshielding_Hz
    pm[:, 3] = np.dot(C_R, *beta[Nm+Np+Nf:Nm+Np+Nf+Nr])
    return pm
