# Retrieval and routing variants that were tried

The paper reports one routing configuration. Four were fitted. This directory holds the three that
were not reported, so the selection can be checked rather than taken on trust.

Selection used out-of-fold accuracy inside the public partition. It never used held-out accuracy,
because choosing among variants by their held-out score is itself fitting to the test set, and it
would invalidate every variant at once rather than just the chosen one.

| Variant | File | Features | Cost | Public OOF | Held-out | vs always direct |
|---|---|---|---|---:|---:|---|
| Pairwise, direct against the graph condition | `../analysis/router.py` | 23, cross-condition | 3x | 68.45 | 71.36 | $+0.53$, $p=0.107$ |
| Multiclass over three conditions | `router_multiclass.py` | 27 | 3x | 66.79 | 70.46 | $-0.37$, $p=0.475$ |
| One binary model per condition | `router_per_condition.py` | 27 | 3x | 68.83 | 71.21 | $+0.37$, $p=0.252$ |
| **Gate on the direct call alone** | `../analysis/router_gate.py` | 27, single-condition | **1.08x** | **68.85** | 71.23 | $+0.39$, $p=0.045$ |

Two things the table is meant to make visible.

**The multiclass variant is worse than not intervening.** Labelling each item with its best condition is
the obvious formulation and it loses to the policy of always taking the direct branch. The reason
is that on 82 percent of items every condition agrees, so the argmax label is mostly noise with respect
to the decision that matters. Replacing it with the incremental question, would switching help,
recovers the loss immediately.

**The reported variant is the cheapest, not just the best scoring.** The three 3x variants read the
predictions of every condition before choosing, so they have already paid for every condition and belong in the
same budget class as majority voting, which needs no training at all. The gate sees only what the
first call produced and pays for a second call only where it fires. Public out-of-fold score and
compute cost point at the same variant, which is why it is the one reported.

Held-out numbers cannot be recomputed from this bundle. The held-out partition is not released.
The scripts run and their protocol is inspectable, but they will report on the public partition and
say so.
