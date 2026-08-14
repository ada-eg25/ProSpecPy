import warnings

import numpy as np
import pytest

from prospecpy.baseline import (
    arpls_baseline,
    arpls_baseline_second_deriv_weights,
)


def generate_synthetic_spectrum():
    """
    Generate a synthetic FTIR-like spectrum.

    Returns
    -------
    x : ndarray
        Wavenumber axis.

    spectrum : ndarray
        Synthetic spectrum containing baseline,
        Gaussian peak and noise.

    baseline : ndarray
        Ground-truth baseline.
    """

    np.random.seed(42)

    x = np.linspace(
        1800,
        2200,
        400,
    )

    # Smooth ground-truth baseline
    baseline = (
        0.2
        + 0.0002 * (x - 1800)
        + 0.0000002 * (x - 2000) ** 2
    )

    # Gaussian absorption peak
    peak = np.exp(
        -((x - 2000) ** 2)
        /
        (2 * 8 ** 2)
    )

    # Random noise
    noise = np.random.normal(
        0,
        0.005,
        len(x),
    )

    spectrum = baseline + peak + noise

    return x, spectrum, baseline


def calculate_rmse(
    y_true,
    y_pred,
):
    """
    Calculate root mean square error.
    """

    return np.sqrt(
        np.mean(
            (y_true - y_pred) ** 2
        )
    )


def test_arpls_returns_valid_baseline():
    """
    Test standard arPLS baseline correction.

    Checks:
    - output length
    - numerical validity
    """

    _, spectrum, _ = generate_synthetic_spectrum()

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        baseline = arpls_baseline(
            spectrum,
            lam=1e5,
            ratio=1e-6,
            max_iter=50,
        )

    assert len(baseline) == len(spectrum)

    assert np.isfinite(baseline).all()


def test_arpls_recovers_synthetic_baseline():
    """
    Test whether arPLS can recover a known
    synthetic baseline.

    This validates numerical behaviour.
    """

    _, spectrum, true_baseline = (
        generate_synthetic_spectrum()
    )

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        estimated_baseline = arpls_baseline(
            spectrum,
            lam=1e5,
            ratio=1e-6,
            max_iter=50,
        )

    rmse = calculate_rmse(
        true_baseline,
        estimated_baseline,
    )

    assert rmse < 0.05


def test_peak_guided_arpls_returns_valid_baseline():
    """
    Test peak-guided arPLS baseline correction.

    Checks:
    - peak-guided method runs successfully
    - output validity
    """

    x, spectrum, _ = generate_synthetic_spectrum()

    peak_wavenumbers = [
        2000,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        baseline = (
            arpls_baseline_second_deriv_weights(
                spectrum,
                x,
                peak_wavenumbers,
                lam=1e6,
                ratio=1e-6,
                max_iter=50,
                peak_window=25,
                peak_weight=0.5,
                alpha=0.8,
            )
        )

    assert len(baseline) == len(spectrum)

    assert np.isfinite(baseline).all()


def test_peak_guided_arpls_modifies_peak_region():
    """
    Test that peak-guided weighting changes
    baseline estimation around supplied peaks.

    This verifies that the additional peak prior
    information is incorporated into arPLS.
    """

    x, spectrum, _ = generate_synthetic_spectrum()

    peak_wavenumbers = [
        2000,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        standard_baseline = arpls_baseline(
            spectrum,
            lam=1e5,
            ratio=1e-6,
            max_iter=50,
        )

        peak_guided_baseline = (
            arpls_baseline_second_deriv_weights(
                spectrum,
                x,
                peak_wavenumbers,
                lam=1e5,
                ratio=1e-6,
                max_iter=50,
                peak_window=25,
                peak_weight=0.3,
                alpha=0.8,
            )
        )

    peak_region = (
        np.abs(x - 2000) < 15
    )

    difference = np.mean(
        np.abs(
            peak_guided_baseline[peak_region]
            -
            standard_baseline[peak_region]
        )
    )

    # Peak-guided prior should influence
    # the baseline estimation near peaks.
    assert difference > 1e-4


def test_peak_guided_arpls_accepts_multiple_peak_positions():
    """
    Test that multiple peak priors are accepted.
    """

    x, spectrum, _ = generate_synthetic_spectrum()

    peak_wavenumbers = [
        1950,
        2000,
        2050,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        baseline = (
            arpls_baseline_second_deriv_weights(
                spectrum,
                x,
                peak_wavenumbers,
                lam=1e6,
                ratio=1e-6,
                max_iter=50,
                peak_window=25,
                peak_weight=0.5,
                alpha=0.8,
            )
        )

    assert len(baseline) == len(spectrum)

    assert np.isfinite(baseline).all()


def test_arpls_rejects_too_short_input():
    """
    Test input validation for insufficient
    number of data points.
    """

    short_spectrum = np.array(
        [
            1.0,
            2.0,
        ]
    )

    with pytest.raises(
        ValueError,
        match="at least 3 data points",
    ):
        arpls_baseline(
            short_spectrum
        )