# Selector layer: how retrieval candidates are built and chosen

The two reasoning chains use separate selectors. They are not the same construction, and the paper
describes them separately for that reason.

What runs and what does not. `cot_selector.py` executes here, rebuilding the quantity graph and
the reranker features from released files. The three `pot_*` modules do not: they import five
modules from the upstream retrieval package, which carries a Contriever checkpoint and is not
ours to redistribute. Each stops with a message naming the missing modules. What ships instead is
the fitted result, in `data/selector/pot_selector_frozen.json` and `pot_function_graph.json`, and
the self-test checks that the graph's sha256 is the one the selector was frozen against.

## Program-of-Thought chain, GPT-4o

| | |
|---|---|
| Candidates | frozen Contriever top-30 |
| Graph | static, built from the function corpus alone. Edges are the union of two relations: shared `article_title`, and Contriever $k$-nearest neighbours with $k=4$ |
| Expansion | exactly one hop, uncapped |
| Selector | linear, 4 features: reciprocal retrieval rank, reciprocal graph depth, edge weight, degree scale |
| Output | at most 3 functions |
| Training | 511 direct `function_id` relevance labels, FinanceReasoning Easy and Medium, 403 / 66 / 42 split, pairwise ranking |

```
pot_build_graph.py           builds the static graph, reads the corpus only
pot_candidate_protocol.py    one-hop expansion over that graph
pot_frozen_selector.py       inference-time scoring plus a 28-field provenance check
pot_train_selector.py        fits the selector from the relevance labels
```

Frozen artifact: `../../data/selector/pot_selector_frozen.json`.
Graph artifact: `../../data/selector/pot_function_graph.json`, 3,133 nodes, 15,889 directed edges,
degree min 4 median 5 max 11, zero isolated nodes.

## Chain-of-Thought chain, DeepSeek-R1

| | |
|---|---|
| Candidates | BM25 top-30 |
| Graph | edges connect functions sharing a normalised input or output quantity name |
| Expansion | nominally two hops, capped at 80 candidates. The cap is reached during the first hop on every query we inspected, so it is single-hop in practice |
| Selector | linear reranker, 56 features |
| Output | top 10 |
| Training | 890 direct `function_id` relevance labels, FinanceReasoning Easy, Medium and Hard, 678 / 106 / 106 split |

```
cot_selector.py         quantity graph, expansion, feature extraction, reranking
cot_train_selector.py   fits the reranker from the relevance labels
```

Frozen artifact: `../../data/selector/cot_selector_frozen.json`.

**A note on which artifact produced the reported numbers.** The upstream bundle also contains a
selector fitted on Easy and Medium only, 736 accepted labels, 560 / 88 / 88. That one did not
produce any number reported in this paper. The artifact shipped here, 890 labels including the
Hard split, is the one the published Chain-of-Thought results were run with. We state this because
the two are easy to confuse and only one of them is load bearing.

## What is not reproducible from this directory

Selector *training* cannot be rerun here, because it needs the FinanceReasoning relevance labels,
which belong to that dataset rather than to us and are cited in the paper. What can be inspected is
the full training code, the frozen coefficients, the training split counts, and the candidate
protocol fingerprint that pins training-time and inference-time candidate construction to the same
definition. Selector *inference* is reproducible: the graph, the coefficients, and the expansion
code are all here.

Neither selector reads a FinExam-10K item, answer, or correctness signal at any point. The
provenance check in `pot_frozen_selector.py` enforces this by asserting a zero-count identifier
intersection against the benchmark before the selector will load.
