"""The selector layer for both reasoning chains.

What runs here and what does not.

`cot_selector.py` runs. It rebuilds the quantity graph and the reranker features from files in this
release, and it is the one to read if you want to see a selector actually execute.

The three `pot_*` modules do not run here. They are the program-of-thought selector as it was
fitted, and they import five modules from the upstream retrieval package: `core`, `graph_features`,
`graph_training`, `label_graph_training` and `retrieval`. That package carries a Contriever
checkpoint and the embedding stack built around it, which is a GPU dependency an order of magnitude
larger than this release and not ours to redistribute. Importing one of them raises a message that
says so rather than a bare ImportError.

What ships instead is the fitted result, which is what a reader actually needs to check the paper:
`data/selector/pot_selector_frozen.json` carries the coefficients, the feature order and the
training-source declaration, and `data/selector/pot_function_graph.json` carries the graph they
were fitted against. `tests/test_bundle.py` checks that the graph's sha256 matches the hash the
frozen selector was minted with, so the pair cannot drift apart unnoticed.
"""
from __future__ import annotations

UPSTREAM_MODULES = ("core", "graph_features", "graph_training",
                    "label_graph_training", "retrieval")

DELIBERATE_STOP = "[dependency-not-released]"

UPSTREAM_NOTE = (
    DELIBERATE_STOP + " this module is the program-of-thought selector as fitted, and it\n"
    "imports {missing} from the upstream retrieval package. That package is not part of this\n"
    "release: it carries a Contriever "
    "checkpoint and the embedding stack around it, which is a GPU dependency far larger than this "
    "bundle and not ours to redistribute.\n"
    "The fitted result ships instead, and is what the paper's claims rest on:\n"
    "    data/selector/pot_selector_frozen.json   coefficients, feature order, training source\n"
    "    data/selector/pot_function_graph.json    the graph they were fitted against\n"
    "`python tests/test_bundle.py` checks that the graph hash matches the one the selector was "
    "frozen with. For a selector that runs end to end here, read code/selector/cot_selector.py."
)


def upstream_required(missing: str) -> SystemExit:
    """The error a pot_* module raises when its upstream dependency is absent, which is always."""
    return SystemExit(UPSTREAM_NOTE.format(missing=missing))
