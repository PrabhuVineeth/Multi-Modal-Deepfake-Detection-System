# Mentor Checklist for 6-Page IEEE Submission

## Must Do Before Submission
- Replace placeholder authors and affiliations.
- Use the generated `architecture_diagram.png` as Fig. 1, or replace it with a cleaner conference-style redraw.
- Decide whether to include the deployment screenshot or omit it to preserve the 6-page limit.
- Verify final metrics from the latest run; update Tables II-IV if newer values exist.
- Compile `MDDS_IEEE_6page_draft.tex` in Overleaf using IEEEtran.
- Keep the paper to 6 pages including references unless the conference permits extra reference pages.

## Page Budget
- Title/Abstract/Index Terms: 0.5 page
- Introduction: 0.75 page
- Literature Survey + Research Gap: 0.75 page
- Methodology: 1.25 pages
- Experimental Setup: 0.5 page
- Results and Discussion: 1.25 pages
- Deployment + Limitations + Conclusion: 0.6 page
- References: 0.4 page

## Recommended Figures/Tables
- Fig. 1: MDDS architecture, one-column or full-width if allowed.
- Table I: Dataset summary.
- Table II: LAV-DF threshold comparison.
- Table III: Dataset-aware calibration summary.
- Table IV: FakeAVCeleb ML baseline.

## Mentor Notes
- Do not overclaim pixel-level localization; call heatmaps frame-level anomaly overlays.
- State clearly that thresholds are validation-selected.
- State clearly that FaceForensics++ is visual-only in this implementation.
- Mention extraction failures and post-extraction imbalance for the classical ML baseline.
