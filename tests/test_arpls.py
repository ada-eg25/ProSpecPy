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

    Contains:
    - smooth curved baseline
    - Gaussian peak
    - small random noise
    """

    np.random.seed(42)

    x = np.linspace(1800, 2200, 400)

    # Smooth baseline
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

    # Noise
    noise = np.random.normal(
        0,
        0.005,
        len(x),
    )

    spectrum = baseline + peak + noise

    return x, spectrum


def test_arpls_returns_valid_baseline():
    """
    Test standard arPLS baseline correction.

    Checks:
    - baseline length
    - numerical validity
    """

    _, spectrum = generate_synthetic_spectrum()

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


def test_peak_guided_arpls_returns_valid_baseline():
    """
    Test peak-guided arPLS baseline correction.

    Checks:
    - peak-guided method runs successfully
    - output is valid
    """

    x, spectrum = generate_synthetic_spectrum()

    peak_wavenumbers = [
        2000,
    ]

    with warnings.catch_warnings():
        warnings.simplefilter(
            "ignore",
            FutureWarning,
        )

        baseline = arpls_baseline_second_deriv_weights(
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

    assert len(baseline) == len(spectrum)

    assert np.isfinite(baseline).all()


def test_peak_guided_arpls_accepts_multiple_peak_positions():
    """
    Test that peak-guided arPLS accepts multiple peak positions.

    This verifies that multiple detected peaks can be provided
    as prior information during baseline estimation.
    """

    x, spectrum = generate_synthetic_spectrum()

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

        baseline = arpls_baseline_second_deriv_weights(
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

    assert len(baseline) == len(spectrum)

    assert np.isfinite(baseline).all()


def test_arpls_rejects_too_short_input():
    """
    Test input validation for insufficient data points.
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