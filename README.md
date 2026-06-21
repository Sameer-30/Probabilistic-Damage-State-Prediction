# Probabilistic Damage-State Prediction — 2015 Gorkha Earthquake

Ordinal prediction of building damage grade with calibrated uncertainty. Three probabilistic models — **NGBoost**, an **MC-Dropout** neural network, and a **deep ensemble** — each return a full predictive distribution over damage states rather than a single label, and report where they are uncertain.

## Motivation

The same structural attributes can map to different damage outcomes, so a useful model should report a probability distribution over damage states, not one deterministic label. For risk assessment, knowing *where the model is uncertain* matters as much as the prediction itself: a confident "moderate" and a borderline "moderate" should be treated differently in a loss estimate.

## Data

~260,000 surveyed buildings from the 2015 Gorkha earthquake ([DrivenData "Richter's Predictor"](https://www.drivendata.org/competitions/57/nepal-earthquake/)). Features include building age, footprint/height percentages, floor and family counts, geographic region, foundation/roof/floor types, and binary superstructure-material flags (mud-mortar stone, adobe, cement-mortar brick, timber, RC, etc.).

Target: `damage_grade` ∈ {1, 2, 3} (low / medium / high), treated as **ordinal** (1 < 2 < 3).

## Approach

**Ordinal formulation (CORN).** Instead of a plain 3-way softmax, the neural models predict K−1 conditional probabilities of exceeding each successive damage threshold, recovering class probabilities via a monotone cumulative product. This guarantees a rank-consistent distribution — the model cannot assign mass to grade 3 and grade 1 while skipping grade 2.

```
P(y > t) = ∏_{k ≤ t} σ(f_k(x))   →   P(y = t) from successive differences
```

**Three uncertainty sources, compared:**

| Model | Accuracy | Ordinal MAE | Uncertainty from |
|---|---|---|---|
| NGBoost | ≈ 0.66 | ≈ 0.34 | Predictive entropy |
| PNN — MC-Dropout | ≈ 0.66 | ≈ 0.34 | Stochastic forward passes |
| PNN — Deep Ensemble | ≈ 0.65 | ≈ 0.36 | Member disagreement |

## Key result

Raw accuracy is comparable across models, as expected for tabular data. The result that matters is **uncertainty quality**: ranking test buildings by the model's own confidence and binning them, accuracy rises monotonically with confidence (~57% on the least-confident bin vs ~72% on the most-confident). Predicted severe-damage probabilities are also well calibrated against observed frequencies.

The model is right more often exactly when it reports being sure — the property that makes an uncertainty estimate trustworthy rather than decorative.

## Usage

Designed to run top-to-bottom in Google Colab.

```bash
pip install ngboost torch scikit-learn pandas numpy matplotlib
```

1. Download `train_values.csv` and `train_labels.csv` from DrivenData.
2. Upload both to the Colab session.
3. Run `gorkha_probabilistic_damage.py`.

Outputs: model comparison table, calibration curve, confidence-vs-accuracy plot, and example predictive distributions for the most- and least-certain buildings.

## Limitations

- Only three coarse damage grades; finer resolution would sharpen prediction and calibration.
- No per-building ground-motion intensity — models learn from structural and geographic attributes, not a physical intensity measure.
- NGBoost training is slow at this sample size; subsampling keeps it practical.
- Calibration is evaluated on a held-out split from the same event; transfer to other earthquakes needs separate validation.

## Context

Part of a broader effort to bring rigorous uncertainty quantification into seismic risk estimation and performance-based earthquake engineering.
