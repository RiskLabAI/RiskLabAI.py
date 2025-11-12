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
    pca,
    denoised_corr,
    denoised_corr2,
    cov_to_corr,
    corr_to_cov,
    denoise_cov,
)

__all__ = [
    "marcenko_pastur_pdf",
    "fit_kde",
    "find_max_eval",
    "pca",
    "denoised_corr",      # Constant Residual Eigenvalue method
    "denoised_corr2",     # Targeted Shrinkage method
    "cov_to_corr",
    "corr_to_cov",
    "denoise_cov",
]