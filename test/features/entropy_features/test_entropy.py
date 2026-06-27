"""
Tests for modules in features/entropy_features/
"""

import numpy as np
from scipy.special import psi

from RiskLabAI.features.entropy_features.bias_corrected import (
    _grassberger_g,
    _mean_entropy_given_beta,
    _nsb_beta_grid,
    grassberger_entropy,
    miller_madow_entropy,
    nsb_entropy,
)
from RiskLabAI.features.entropy_features.kontoyiannis import kontoyiannis_entropy
from RiskLabAI.features.entropy_features.lempel_ziv import lempel_ziv_entropy
from RiskLabAI.features.entropy_features.plug_in import plug_in_entropy_estimator
from RiskLabAI.features.entropy_features.pmf import (
    ngram_counts,
    probability_mass_function,
)
from RiskLabAI.features.entropy_features.shannon import shannon_entropy

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
    h = -((5 / 9) * np.log2(5 / 9) + (4 / 9) * np.log2(4 / 9))
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


# --- Bias-corrected entropy (Miller-Madow, Grassberger, NSB) ---

_SYMBOLS = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _uniform_message(alphabet_size, n, rng):
    """A message of n draws from a uniform alphabet of the given size (true H = log2 K)."""
    return "".join(chr(33 + x) for x in rng.integers(0, alphabet_size, size=n))


def test_ngram_counts_shared_path():
    """ngram_counts gives integer counts; the PMF is those counts normalized (plug-in unchanged)."""
    counts = ngram_counts(MSG_MED, 2)
    assert counts == {"AB": 5, "BA": 4}
    pmf = probability_mass_function(MSG_MED, 2)
    assert np.isclose(pmf["AB"], 5 / 9.0) and np.isclose(pmf["BA"], 4 / 9.0)


def test_plug_in_behaviour_unchanged():
    """The pmf refactor leaves the plug-in / shannon outputs bit-for-bit identical."""
    assert plug_in_entropy_estimator(MSG_LOW, 1) == 0.0
    assert plug_in_entropy_estimator(MSG_MED, 1) == 1.0
    h = -((5 / 9) * np.log2(5 / 9) + (4 / 9) * np.log2(4 / 9))
    assert plug_in_entropy_estimator(MSG_MED, 2) == h / 2.0
    assert shannon_entropy(MSG_HIGH) == plug_in_entropy_estimator(MSG_HIGH, 1)


def test_grassberger_published_values():
    """G_n matches the published Grassberger (2008) values G_1 and G_2."""
    gamma = 0.5772156649015329
    g1 = _grassberger_g(np.array([1.0]))[0]
    g2 = _grassberger_g(np.array([2.0]))[0]
    assert np.isclose(g1, -gamma - np.log(2.0))
    assert np.isclose(g2, 2.0 - gamma - np.log(2.0))


def test_nsb_beta_zero_limit():
    """As beta -> 0 the NSB posterior mean reduces to the psi (Bayesian a=0) estimator (nats)."""
    counts = np.array([5.0, 3.0, 2.0, 1.0, 1.0])
    n = counts.sum()
    expected = psi(n + 1.0) - (1.0 / n) * np.sum(counts * psi(counts + 1.0))
    got = _mean_entropy_given_beta(
        np.array([1e-9]), k=5, n=n, counts=counts, k_obs=counts.size
    )[0]
    assert np.isclose(got, expected, atol=1e-6)


def test_corrections_converge_when_well_sampled():
    """N >> K: all four agree with the plug-in within a small tolerance (no over-correction)."""
    rng = np.random.default_rng(0)
    msg = _uniform_message(8, 8000, rng)
    plug = plug_in_entropy_estimator(msg, 1)
    mm = miller_madow_entropy(msg, 1)
    gr = grassberger_entropy(msg, 1)
    ns = nsb_entropy(msg, 1, alphabet_size=8)
    for value in (mm, gr, ns):
        assert abs(value - plug) < 0.02
    # No over-correction: a correction must not exceed the true entropy by more than noise.
    true_h = np.log2(8)
    for value in (mm, gr, ns):
        assert value <= true_h + 0.05


def test_bias_ordering_when_undersampled():
    """
    Deep undersampling: |bias(plug)| >= |bias(MM)| >= |bias(Grassberger)| >= |bias(NSB)| on average.

    The pre-registered monotone ordering holds in genuine undersampling; near critical sampling
    (K_eff/N ~ 1) Miller-Madow can over-correct on a uniform source, so the fixture is clearly deep
    (base alphabet 16, word length 2 -> K_eff = 256, N = 40, ratio ~ 6.6).
    """
    rng = np.random.default_rng(1)
    base, word, n = 16, 2, 40
    true_h = np.log2(base)  # per-symbol entropy of the uniform source
    plug, mm, gr, ns = [], [], [], []
    for _ in range(150):
        msg = _uniform_message(base, n, rng)
        plug.append(plug_in_entropy_estimator(msg, word))
        mm.append(miller_madow_entropy(msg, word))
        gr.append(grassberger_entropy(msg, word))
        ns.append(nsb_entropy(msg, word, alphabet_size=base))
    bias = [abs(np.mean(x) - true_h) for x in (plug, mm, gr, ns)]
    assert bias[0] >= bias[1] >= bias[2] >= bias[3]


def test_edge_cases_empty_and_single_symbol():
    """Empty, too-short, and single-symbol-type messages are handled without error."""
    for estimator in (miller_madow_entropy, grassberger_entropy, nsb_entropy):
        assert estimator("", 1) == 0.0
        assert estimator("AB", 3) == 0.0  # shorter than the word length
    # A single symbol type: plug-in is exactly 0; the corrections are finite and small.
    assert plug_in_entropy_estimator(MSG_LOW, 1) == 0.0
    assert miller_madow_entropy(MSG_LOW, 1) == 0.0  # K_hat - 1 = 0
    assert np.isfinite(grassberger_entropy(MSG_LOW, 1))
    assert np.isfinite(nsb_entropy(MSG_LOW, 1, alphabet_size=2))


def test_nsb_large_alphabet_stability_and_caching():
    """NSB is finite and uses a cached beta grid for a very large effective alphabet."""
    rng = np.random.default_rng(2)
    msg = _uniform_message(60, 80, rng)
    _nsb_beta_grid.cache_clear()
    v1 = nsb_entropy(msg, 2, alphabet_size=60)  # K_eff = 3600
    v2 = nsb_entropy(msg, 2, alphabet_size=60)
    assert np.isfinite(v1)
    assert v1 == v2  # deterministic
    assert _nsb_beta_grid.cache_info().hits >= 1  # second call hit the cache


def test_nsb_default_alphabet_from_message():
    """Without alphabet_size, NSB uses the observed distinct symbols and still returns a value."""
    rng = np.random.default_rng(3)
    msg = _uniform_message(16, 40, rng)
    assert np.isfinite(nsb_entropy(msg, 1))
