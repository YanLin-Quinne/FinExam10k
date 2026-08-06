# Requirement scenarios

## A. Anonymous package presentation

The README references an existing Figure 1 asset and leaderboard preview, links to the expected
offline leaderboard, and provides relative links to JSON, JSONL, CSV, and XLSX public exports.

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

## F. Anonymous package scope

Tracked release metadata contains no file-content summaries or source version tokens. Dataset
structure, split boundaries, matrix alignment, feature dimensions, and
published result assertions remain covered.
