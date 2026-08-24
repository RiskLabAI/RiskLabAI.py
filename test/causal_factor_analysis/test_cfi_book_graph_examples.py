"""Hand-verifiable graph tests for the 2023 book examples."""

from __future__ import annotations

from RiskLabAI.causal_factor_analysis import (
    CausalDAG,
    causal_role_evidence,
    check_backdoor_adjustment_set,
    check_instrument,
    d_separation,
    minimal_backdoor_adjustment_sets,
    minimal_frontdoor_adjustment_sets,
)


def _dag(nodes, edges, observed=None):
    node_tuple = tuple(nodes)
    return CausalDAG(
        node_tuple,
        tuple(edges),
        node_tuple if observed is None else tuple(observed),
    )


def test_figures_1_2_and_13_require_the_common_cause_for_backdoor_adjustment():
    graph = _dag(("X", "Y", "Z"), (("Z", "X"), ("Z", "Y")))
    assert d_separation(graph, ("X",), ("Y",)).separated is False
    assert d_separation(graph, ("X",), ("Y",), ("Z",)).separated is True
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == (("Z",),)
    role = causal_role_evidence(graph, "X", "Y", "Z")
    assert role.common_cause_paths == (("X", "Z", "Y"),)


def test_figure_3_frontdoor_mediator_identifies_when_confounder_is_latent():
    graph = _dag(
        ("M", "U", "X", "Y"),
        (("U", "X"), ("U", "Y"), ("X", "M"), ("M", "Y")),
        observed=("M", "X", "Y"),
    )
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == ()
    assert minimal_frontdoor_adjustment_sets(graph, "X", "Y") == (("M",),)


def test_figure_4_instrument_satisfies_both_named_graph_profiles():
    graph = _dag(
        ("U", "W", "X", "Y"),
        (("W", "X"), ("X", "Y"), ("U", "X"), ("U", "Y")),
    )
    result = check_instrument(graph, "W", "X", "Y")
    assert result.pearl_graphical is True
    assert result.cfi_simple is True


def test_figure_8_hypothetical_factor_mechanism_has_backdoor_and_frontdoor_sets():
    edges = (
        ("MOM", "HML"),
        ("MOM", "PC"),
        ("HML", "OI"),
        ("OI", "PC"),
    )
    observed_graph = _dag(("HML", "MOM", "OI", "PC"), edges)
    assert minimal_backdoor_adjustment_sets(observed_graph, "HML", "PC") == (("MOM",),)

    latent_graph = _dag(
        ("HML", "MOM", "OI", "PC"),
        edges,
        observed=("HML", "OI", "PC"),
    )
    assert minimal_backdoor_adjustment_sets(latent_graph, "HML", "PC") == ()
    assert minimal_frontdoor_adjustment_sets(latent_graph, "HML", "PC") == (("OI",),)


def test_figures_16_to_18_conditioning_on_collider_opens_the_path():
    graph = _dag(("X", "Y", "Z"), (("X", "Z"), ("Y", "Z")))
    assert d_separation(graph, ("X",), ("Y",)).separated is True
    conditioned = d_separation(graph, ("X",), ("Y",), ("Z",))
    assert conditioned.separated is False
    assert tuple(path.nodes for path in conditioned.paths if path.is_open) == (
        ("X", "Z", "Y"),
    )


def test_figure_19_conditioning_on_chain_mediator_blocks_total_effect_path():
    graph = _dag(("X", "Y", "Z"), (("X", "Z"), ("Z", "Y")))
    assert d_separation(graph, ("X",), ("Y",)).separated is False
    assert d_separation(graph, ("X",), ("Y",), ("Z",)).separated is True
    role = causal_role_evidence(graph, "X", "Y", "Z")
    assert role.mediator_on_directed_paths == (("X", "Z", "Y"),)


def test_figure_20_empty_backdoor_set_is_valid_but_conditioning_mediator_is_not():
    graph = _dag(
        ("W", "X", "Y", "Z"),
        (("X", "Z"), ("W", "Z"), ("Z", "Y"), ("W", "Y")),
    )
    assert minimal_backdoor_adjustment_sets(graph, "X", "Y") == ((),)
    assert check_backdoor_adjustment_set(graph, "X", "Y", ()).admissible is True
    conditioned = check_backdoor_adjustment_set(graph, "X", "Y", ("Z",))
    assert conditioned.admissible is False
    assert conditioned.descendants_in_set == ("Z",)
