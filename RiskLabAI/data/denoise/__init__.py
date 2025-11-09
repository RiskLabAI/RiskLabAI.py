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
    marcenko_pastur_pdf,
    fit_kde,
    find_max_eval,
    denoised_corr,
    cov_to_corr,
    corr_to_cov,
    denoise_cov,
    optimal_portfolio,
    optimal_portfolio_denoised,
)

__all__ = [
    "marcenko_pastur_pdf",
    "fit_kde",
    "find_max_eval",
    "denoised_corr",
    "cov_to_corr",
    "corr_to_cov",
    "denoise_cov",
    "optimal_portfolio",
    "optimal_portfolio_denoised",
]