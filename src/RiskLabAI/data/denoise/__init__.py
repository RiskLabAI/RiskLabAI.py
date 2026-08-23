"""
RiskLabAI Data Denoising Module

Implements noise reduction techniques for covariance and correlation
matrices, based on Random Matrix Theory (RMT) and the
Marcenko-Pastur distribution.

Reference:
    De Prado, M. (2018) Advances in financial machine learning.
    John Wiley & Sons, Chapter 2.
"""

from .denoising import (
    corr_to_cov,
    cov_to_corr,
    denoise_cov,
    denoised_corr,
    find_max_eval,
    fit_kde,
    marcenko_pastur_pdf,
    optimal_portfolio,
    pca,
)
from .nercome import nercome_denoised_covariance

__all__ = [
    "marcenko_pastur_pdf",
    "fit_kde",
    "find_max_eval",
    "pca",
    "denoised_corr",
    "denoised_corr2",
    "cov_to_corr",
    "corr_to_cov",
    "denoise_cov",
    "optimal_portfolio",
    # NERCOME (Appraisal 24 admit)
    "nercome_denoised_covariance",
]
