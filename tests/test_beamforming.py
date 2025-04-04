import numpy as np
import numpy.testing as npt
import pytest

import spharpy


def test_dolph_cheby_mainlobe():
    N = 7
    theta0 = np.pi/6
    d_nm = spharpy.beamforming.dolph_chebyshev_weights(
        N, theta0, design_criterion='mainlobe')

    truth = np.loadtxt('tests/data/dolph_cheby_mainlobe.csv', delimiter=',')
    npt.assert_allclose(d_nm, truth)

def test_dolph_cheby_sidelobe():
    N = 7
    R_dB = 50
    R = 10**(R_dB/20)
    d_nm = spharpy.beamforming.dolph_chebyshev_weights(
        N, R, design_criterion='sidelobe')

    truth = np.loadtxt('tests/data/dolph_cheby_sidelobe.csv', delimiter=',')
    npt.assert_allclose(d_nm, truth)

def test_mvdr_weights():
    nSH = 4
    sig = np.random.randn(1000, nSH)
    sphCov = np.cov(sig, rowvar=False)
    az = np.array([45, 60, 10, 20])/180*np.pi
    el = np.array([30, 20, 10, 0])/180*np.pi
    
    coord = spharpy.samplings.Coordinates.from_spherical(np.array([1,1,1,1]) ,az, el)

    d_mvdr = spharpy.beamforming.mvdr_weights(
        sphCov, coord, diag_loading=0)
    print(d_mvdr)

def test_re_max():
    N = 7
    g_nm = spharpy.beamforming.rE_max_weights(N, normalize=False)

    truth = np.loadtxt('tests/data/re_max_weights.csv', delimiter=',')
    npt.assert_allclose(g_nm, truth)

    g_nm_norm = spharpy.beamforming.rE_max_weights(N, normalize=True)

    Y = spharpy.spherical.spherical_harmonic_basis_real(
        N, spharpy.samplings.Coordinates(1, 0, 0))

    npt.assert_allclose(Y @ np.diag(g_nm_norm) @ Y.T, 1)


def test_max_front_back():
    N = 7
    f_nm_norm = spharpy.beamforming.maximum_front_back_ratio_weights(
        N, normalize=True)

    Y = spharpy.spherical.spherical_harmonic_basis_real(
        N, spharpy.samplings.Coordinates(1, 0, 0))

    npt.assert_allclose(Y @ np.diag(f_nm_norm) @ Y.T, 1)

    with pytest.raises(RuntimeError, match='did not converge'):
        spharpy.beamforming.maximum_front_back_ratio_weights(30)


def test_normalize_weights():
    n_max = 7
    normalized_weights = spharpy.beamforming.normalize_beamforming_weights(
        np.ones(n_max+1), n_max)

    # pwd weights are const 4*pi/(n_max+1)**2
    expected = np.ones(n_max+1)*4*np.pi/(n_max+1)**2

    npt.assert_allclose(expected, normalized_weights)
