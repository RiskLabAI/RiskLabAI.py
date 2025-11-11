"""
Tests for modules in features/entropy_features/
"""

import pytest
import numpy as np
from RiskLabAI.features.entropy_features.shannon import shannon_entropy
from RiskLabAI.features.entropy_features.lempel_ziv import lempel_ziv_entropy
from RiskLabAI.features.entropy_features.pmf import probability_mass_function
from RiskLabAI.features.entropy_features.plug_in import plug_in_entropy_estimator
from RiskLabAI.features.entropy_features.kontoyiannis import kontoyiannis_entropy

# --- Test Data ---
MSG_LOW = "AAAAAAAAAA"
MSG_MED = "ABABABABAB"
MSG_HIGH = "ABCDEFGHIJ"

# --- Shannon Tests ---
def test_shannon_entropy():
    assert np.isclose(shannon_entropy(MSG_LOW), 0.0)
    assert np.isclose(shannon_entropy(MSG_MED), 1.0)
    assert np.isclose(shannon_entropy(MSG_HIGH), np.log2(10))
    assert np.isclose(shannon_entropy(""), 0.0)

# --- Lempel-Ziv Tests ---
def test_lempel_ziv_entropy():
    # lib = {"A", "AA", "AAA", "AAAA"} -> len=4
    assert np.isclose(lempel_ziv_entropy(MSG_LOW), 4 / 10.0)
    # lib = {"A", "B", "AB", "ABA", "BAB"} -> len=5
    assert np.isclose(lempel_ziv_entropy(MSG_MED), 5 / 10.0)
    # lib = {"A", "B", "C", ..., "J"} -> len=10
    assert np.isclose(lempel_ziv_entropy(MSG_HIGH), 10 / 10.0)

# --- PMF Tests ---
def test_probability_mass_function():
    pmf = probability_mass_function(MSG_LOW, approximate_word_length=1)
    assert pmf == {"A": 1.0}

    pmf_2 = probability_mass_function(MSG_MED, approximate_word_length=2)
    # "AB", "BA", "AB", "BA", "AB", "BA", "AB", "BA", "AB" (9 windows)
    # 5 "AB", 4 "BA"
    assert np.isclose(pmf_2["AB"], 5 / 9.0)
    assert np.isclose(pmf_2["BA"], 4 / 9.0)

# --- Plug-in Tests ---
def test_plug_in_estimator():
    # word_len=1 -> same as shannon
    assert np.isclose(plug_in_entropy_estimator(MSG_LOW, 1), 0.0)
    assert np.isclose(plug_in_entropy_estimator(MSG_MED, 1), 1.0)
    
    # word_len=2
    # H = -( (5/9)*log2(5/9) + (4/9)*log2(4/9) ) = 0.991
    # H_norm = H / 2
    h = -( (5/9)*np.log2(5/9) + (4/9)*np.log2(4/9) )
    assert np.isclose(plug_in_entropy_estimator(MSG_MED, 2), h / 2.0)

# --- Kontoyiannis Tests ---
def test_kontoyiannis_entropy():
    # Expanding window
    # L_i for "AAAAA" is [1, 1, 2, 3, 4]
    # points = range(2, 5) -> [2, 3, 4]
    # i=2: n=2, L_i(msg, 2, 2) -> "A" in "AA" -> L=2. sum += log2(2)/2 = 0.5
    # i=3: n=3, L_i(msg, 3, 3) -> "AA" in "AAA" -> L=3. sum += log2(3)/3 = 0.528
    # i=4: n=4, L_i(msg, 4, 4) -> "A" in "AAAA" -> L=2. sum += log2(4)/2 = 1.0
    # h = (0.5 + 0.528 + 1.0) / 3 = 2.028 / 3 = 0.676
    assert np.isclose(kontoyiannis_entropy("AAAAA"), 0.62055, atol=1e-3)


    # Rolling window
    # window=3. points=range(3, 5) -> [3, 4]
    # i=3: n=3. L_i(message, 3, 3) -> "AA" in "AAA" -> L=3. sum += log2(3)/3 = 0.528
    # i=4: n=3. L_i(message, 4, 3) -> "A" in "AAA" -> L=2. sum += log2(3)/2 = 0.792
    # h = (0.528 + 0.792) / 2 = 0.66
    assert np.isclose(kontoyiannis_entropy("AAAAA", window=3), 0.660, atol=1e-3)