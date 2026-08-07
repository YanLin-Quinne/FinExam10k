# Requirement scenarios

## A. Public project presentation

The README uses the paper title without reviewer packaging language. It remains identity-free,
places the existing Figure 1 exactly once immediately before Citation, retains the leaderboard
preview, links to the offline leaderboard, and provides valid relative links to the public exports
and diagnostic files.

## B. Leaderboard behavior

The site loads package owned JSON, exposes all 17 systems in each benchmark view, preserves the
three paper categories, supports view, category, and model filters plus sortable columns, and has
clear empty and load-error states.

## C. Accessible responsive layout

The page uses semantic tables and controls, retains visible keyboard focus, provides a skip link and
live result status, and keeps the table usable on narrow screens.

## D. Public export equivalence

Regeneration produces 5,110 records with the required columns. JSON, JSONL, CSV, and XLSX contain
the same IDs and values, and the export summary contains counts and file sizes only.

## E. Scientific reproduction

The public reproduction entry point and aggregate arithmetic reproduce the published public Gate,
selector, RQ1, and RQ2 results using frozen inputs without file-summary machinery.

## F. Release metadata scope

Tracked release metadata contains no file-content summaries or source version tokens. Dataset
structure, split boundaries, matrix alignment, feature dimensions, and
published result assertions remain covered.

## G. GitHub Pages deployment

The assembled site serves the same leaderboard at the repository Pages root and at
`leaderboard.html`. Stylesheets, scripts, aggregate JSON, visual assets, and all four public data
downloads resolve within the deployed project subpath. Only the 5,110-record public exports and
aggregate leaderboard data are staged; no held-out item files are published.

## H. README scientific content

The README records the full, public, and held-out counts, the frozen difficulty bands, the
Context-Complete Reasoning Track, and the two nested diagnostics. It distinguishes structural
record completeness from detached parent evidence, lists all 17 leaderboard models, and describes
the matched Function-RAG, FunctionGraph-RAG, verification, and frozen public gate conditions.
