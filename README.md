# Biochar emerging-contaminant adsorption: final analysis code

Code and computational evidence for **Estimating removal advantages among characterized biochars: A 24-method benchmark for emerging contaminant adsorption**.

This repository packages the final analysis from the 2026-09-24 local research archive. The 28 Python source files retain the archived analysis; three scripts use the renamed `dataset/` input directory. It includes the 24-method benchmark, inner-only TOPSIS selection, grouped validation, paired LightGBM SHAP, conditional operational ALE, plotting scripts, fixed splits, and saved numerical outputs. Manuscript drafts, reviewer correspondence, and third-party article PDFs are excluded.

## Install

Use Python 3.12 (original environment: 3.12.3). From the repository root:

```bash
python -m venv .venv
# Windows PowerShell: .venv/Scripts/Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
```

The numerical dependencies are pinned to the archived environment. Native LightGBM `pred_contrib=True` computes TreeSHAP; the separate `shap` package is not needed. Package installation on a fresh machine was not tested during this upload.

## Repository contents

- `analysis/scripts/`: final numerical and plotting code, including the bundled plotting helpers in `vendor/`.
- `analysis/data/`: 1,348 derived conditions before analysis exclusions, and the original-row mapping.
- `analysis/evidence/`: saved predictions, fit caches, split definitions, protocols, diagnostic checks, and final summaries. Earlier caches remain when used by final scripts or for provenance.
- `dataset/Raw_data.xlsx` and `Raw_data.csv`: unchanged copies of the supplied experimental data. The input paths in the analysis scripts resolve this directory directly.
- `docs/software_versions.json`: original environment record.
- `SHA256SUMS.json`: SHA-256 checksums of all committed release files except the checksum file itself.

The analysis uses 3,757 original records, 1,279 eligible conditions after exclusions, and 559 matched material queries. Repeated fits are not additional experimental observations.

## Verify the included results

```bash
python verify_release.py
python analysis/scripts/verify_full24_data_lineage.py
python analysis/scripts/verify_static_diagnostics_full24.py
python analysis/scripts/verify_topsis_surfaces.py
```

The scientific checks verify raw-to-condition lineage, diagnostic tables and matched queries, and independent reconstruction of TOPSIS selections and held-out metrics. They rewrite their corresponding verification reports. The release checksum check should therefore be run first.

## Run analyses

The repository includes the computational caches needed by the final scripts. Commands below may rewrite derived outputs; use a fresh clone for experiments.

Reassemble benchmark summaries from the saved predictions:

```bash
python analysis/scripts/summarize_model24_topsis.py
python analysis/scripts/full24_analysis.py
python analysis/scripts/summarize_pareto_24.py
```

The original three-seed benchmark is resumable and reuses existing checkpoint files:

```bash
python analysis/scripts/run_model24_topsis.py --seed 11 --jobs 6
python analysis/scripts/run_model24_topsis.py --seed 23 --jobs 6
python analysis/scripts/run_model24_topsis.py --seed 47 --jobs 6
```

These commands on the supplied repository reuse cached fits; they are not a clean retraining test. The earliest raw-data aggregation code is not present in the supplied final archive; its saved derived inputs and independent lineage verification are included.

Recompute the latest explanatory analyses using the existing selections:

```bash
python analysis/scripts/paired_shap.py
python analysis/scripts/conditional_operational_ale.py
```

These perform 30 LightGBM refits and 15 selected-model refits respectively, and assert agreement with frozen predictions. Current SHAP outputs are in `paired_shap_v1`; current pH/temperature ALE outputs are in `adsorption_effects_v2`. The older pooled ALE is retained only for provenance and other inputs.

Additional validation entry points are `reviewer_repeated_validation.py`, `reviewer_input_sensitivity.py`, `reviewer_summaries.py`, and `original_capacity_validation.py`. Full retraining can be computationally expensive.

## Rebuild final figures from included evidence

```bash
python analysis/scripts/plot_panel_integration.py
python analysis/scripts/plot_adsorption_effects.py
python analysis/scripts/plot_reviewer_controls.py
python analysis/scripts/plot_paired_shap.py
python analysis/scripts/plot_capacity_validation.py
```

Outputs are written under `analysis/figures/` (ignored by Git). These are the final figure entry points; `plot_full24_revision.py` also contains earlier builders used by them. Exact typography depends on installed fonts.

## Data provenance and interpretation

The source dataset accompanies Jaffari et al., *Chemical Engineering Journal* 466 (2023), 143073, DOI: [10.1016/j.cej.2023.143073](https://doi.org/10.1016/j.cej.2023.143073). The copies supplied with this project are retained unchanged; their original terms and attribution continue to apply. No additional reuse license is assigned by this upload.

The source labels are candidate-source mappings. Grouped validation and source holdout have different scopes. SHAP describes fitted-model contributions, not isolated causal mechanisms. Conditional ALE restricts pH responses to 25 degrees C and temperature responses to pH 7; repeated-seed ranges are sensitivity summaries, not confidence intervals. The capacity analysis reconstructs a reported protocol rather than reproducing unknown original fitted objects.

## Release verification

See `docs/upload_validation.json` for checks actually run for this release. Upload validation does not imply that every model was retrained or every figure regenerated.
