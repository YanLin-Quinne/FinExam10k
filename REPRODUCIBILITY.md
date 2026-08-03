# Reproducibility statement

## Environment

```
python 3.11
numpy, scikit-learn, matplotlib
```

No GPU is required for anything in this bundle. The bundle carries derived model predictions, not
model weights, so no inference is rerun.

## What reproduces exactly

Every statistic the paper reports over the released public partition. Deterministic given the
files here: bootstrap and cross-validation both use seed 202607, fixed in the scripts.

```
cd code/analysis
python why_hard372.py            # error concentration, exact null by dynamic programming
python where_graph_wins.py       # slice sweep, 220 tests, Benjamini-Hochberg corrected
python rq2_error_analysis.py     # rescue and harm decomposition by intervention condition
cd ../figures
python make_heatmap_subject.py   # subject heatmap
python make_taxonomy_figure.py   # curriculum taxonomy
```

## What does not reproduce, and why

**Every held-out number, which includes the entire router evaluation.** The router is fitted on the
public partition and scored once on the 5,088 held-out items, and the held-out partition is not
released. This is the property that makes the leaderboard protocol meaningful. The router code
ships in full so its protocol can be inspected: `code/analysis/router_gate.py` is the gate reported
in the paper, and `code/analysis/router.py` and `code/rag_variants/router_v2.py` are the two
variants evaluated alongside it.

**Statistics quoted over all 10,198 items**, since those cover both partitions. Scripts report the
item count they actually ran on, so a smaller count is visible rather than silent.

## Model selection discipline

Four router variants were fitted. Selection among them used out-of-fold accuracy inside the public
partition, never held-out accuracy, because choosing by held-out score is itself fitting to the
test set. The reported variant has the highest public out-of-fold score and is also the only one
whose compute cost is comparable to the no-intervention baseline, so both criteria agree. All four
ship: `code/analysis/router_gate.py` is the reported one, `code/analysis/router.py` is the pairwise
variant, and `code/rag_variants/` holds the multiclass and per-condition variants together with a table
comparing all four. One of the four is worse than never intervening and is kept for that reason.

## Selector layer

`code/selector/` holds the graph construction, the candidate protocol, the inference-time selector
and the training code for both chains, together with the frozen coefficients and the function
graph. `data/selector/per_item_sidecar_public_5110.json` gives, for every released item, the
candidate pool size, the number of functions actually injected, and the parser and executor verdict
for both retrieval conditions. That file is what makes the stratified analysis in RQ2 checkable
without rerunning anything.

Selector training itself cannot be rerun from this bundle: it consumes FinanceReasoning relevance
labels, which belong to that dataset. The code, the split counts, the coefficients and the
candidate-protocol fingerprint are all here, so the procedure is inspectable even where it is not
re-executable. Held-out per-item sidecars are withheld for the same reason the held-out items are.

## Compute

Model inference for the seventeen leaderboard systems and the nine intervention conditions was run once
and frozen before analysis. Ten systems were API served. Seven were evaluated locally on NVIDIA
H100 accelerators, six of them on a single card and one on two. Decoding used temperature 0,
top-p 1, seed 202607, a 32,768 token context and an 8,192 token completion limit. The appendix
gives the byte-exact prompt.

## Known deviations

The bundle ships derived predictions rather than raw generations. Raw generations are large and
their file names encode provider account structure, so they are withheld for anonymity. The
predictions are sufficient to recompute every reported accuracy, band, concentration statistic and
rescue-harm decomposition on the public partition.
