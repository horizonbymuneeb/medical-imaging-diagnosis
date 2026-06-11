# medical-imaging-diagnosis

A CPU-only CT pipeline for lesion classification: it generates synthetic head
phantoms in Hounsfield units, applies clinical windowing and normalisation,
samples balanced patches, trains a small CNN with early stopping, and reports
AUC, sensitivity at a fixed specificity, confusion matrices and calibration
error. Everything runs on a laptop CPU in a few seconds, with no dataset
download and no GPU.

The point of the project is the pipeline, not a headline accuracy number: the
synthetic phantom is a stand-in for a real DICOM series, so each stage can be
tested for the property that actually matters (does the window clip the tissue,
does the sampler avoid drawing all-negative batches, does temperature scaling
change probabilities without changing the ranking).

## What is in the box

| Module | Job |
| --- | --- |
| `src/medimg/phantom.py` | Synthetic CT head phantom: skull ring, grey/white matter, ventricles, a hyperdense lesion, Gaussian or Rician noise. Deterministic per seed. |
| `src/medimg/volume.py` | `Volume` container with spacing/origin/modality, `.npz` save/load, resampling, crop/pad, slicing, intensity statistics, bounding boxes. |
| `src/medimg/windowing.py` | Brain/lung/abdomen/bone window presets, HU-to-`[0,1]` windowing, min-max/z-score/percentile normalisation, clipped-fraction diagnostics. |
| `src/medimg/patching.py` | Patch extraction with border padding, lesion connected components, positive sampling from lesion voxels, distance-constrained negative sampling, class balancing, train/val split. |
| `src/medimg/augment.py` | Label-safe augmentation: flips, 90-degree rotations, gamma, intensity scaling, Gaussian noise, and paired translation for image+mask. |
| `src/medimg/dataset.py` | On-demand `torch` datasets built from phantom seeds, batching, train/validation loaders, class-balance statistics. |
| `src/medimg/model.py` | `SmallCNN` (three conv blocks, global average pooling, one logit), parameter counting, checkpoints, probability inference. |
| `src/medimg/train.py` | Training loop, `EarlyStopping` that restores the best weights, `TrainConfig`, config-driven CLI. |
| `src/medimg/metrics.py` | Hand-written ROC points, AUC, sensitivity at specificity, confusion matrix, per-class precision/recall/F1, Youden and F1 threshold selection. |
| `src/medimg/calibration.py` | Reliability curve, ECE, maximum calibration error, Brier score, temperature scaling fitted by NLL. |
| `src/medimg/config.py` | YAML loading with a single-pass validator, deep merge, dotted lookup, config fingerprint. |
| `src/medimg/demo.py` | End-to-end demo of every stage, printed as JSON. |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch
pip install -r requirements.txt

make test                          # pytest -q
python -m src.medimg.demo --config config.yaml
python -m src.medimg.train --config config.yaml --out artifacts/run.json
```

`config.yaml` holds the real tunables: phantom geometry and noise, window preset
and normalisation mode, patch size and positive fraction, augmentation
probabilities, model width, and optimiser/early-stopping settings.

## Design notes

**Windowing before normalisation, not instead of it.** A brain window maps
0-80 HU to the full range, so CSF and grey matter stay distinguishable after
min-max scaling. Normalising raw HU first would compress everything into the
bottom 2 % of the range because air at -1000 HU dominates the histogram. The
tests pin the window edges (`level ± width/2` maps to 0 and 1) because an
off-by-half-width error is invisible in a plot and expensive in training.

**Negative patches are sampled by distance, not by random draw.** Lesions cover
well under 1 % of a slice, so uniform sampling yields batches with no positives
at all. `negative_centers` rejects candidates within one patch size of any
lesion voxel and raises after a bounded number of attempts rather than looping
forever when the lesion covers the slice.

**Augmentation is split into label-safe and paired operations.** Flips,
rotations by 90 degrees and intensity changes cannot move the lesion out of the
patch, so they are applied to the image alone. Translation is the one geometric
operation that can destroy a label, so it is only available as
`translate_pair(image, mask)` which shifts both arrays with the same offset.

**Early stopping restores the best weights.** Stopping is the easy half; the
loop keeps a copy of the state dict at the best validation loss and reloads it
before evaluation, so the reported probabilities come from the kept epoch, not
the last one.

**Calibration is measured on the same probabilities that are reported.** The
balanced patch sampler changes the effective prevalence, so confidence and
accuracy diverge; temperature scaling is fitted on validation logits and the
test asserts that AUC is unchanged (temperature scaling is monotone) while NLL
does not get worse.

## Tests

`python -m pytest -q` runs **144 tests in about 6.5 s** on CPU. The suite is
organised one file per module:

- `tests/test_phantom.py` - ellipsoid volume vs the analytic formula, skull and
  lesion voxels present, lesion mask equals painted voxels, determinism per
  seed, Gaussian noise standard deviation, Rician noise non-negativity.
- `tests/test_volume.py` - metadata validation, `.npz` round-trip, resampling
  shape and spacing, crop/pad offsets, slice bounds, stats and tight bounding
  boxes.
- `tests/test_windowing.py` - window edges, clipping behaviour, exact
  normalisation values, batch shape checks, unknown-preset errors.
- `tests/test_patching.py` - patch values and border padding, lesion centroids,
  positive centres landing on lesions, negative-centre distance constraint, exact
  50/50 balancing, deterministic disjoint split.
- `tests/test_augment.py` - probability-1 and probability-0 behaviour, gamma on
  out-of-range values, noise standard deviation, paired translation keeps the
  mask aligned with the image.
- `tests/test_dataset.py` - item shapes, seed determinism, loader split sizes,
  class-balance statistics.
- `tests/test_model.py` - logit shape, patch-size independence, parameter counts,
  checkpoint round-trip reproduces predictions, threshold behaviour.
- `tests/test_train.py` - early-stopping trigger and weight restoration, loss
  decreases on separable data, `fit` history keys and best-weight evaluation.
- `tests/test_metrics.py` - AUC for perfect/reversed/tied rankings, hand-computed
  0.75 case, ROC monotonicity, confusion-matrix layout, per-class metrics,
  threshold selection.
- `tests/test_calibration.py` - empty-bin handling, ECE of calibrated and
  overconfident predictions, Brier score, temperature fit improves NLL without
  changing AUC.
- `tests/test_config.py` - the repository `config.yaml` validates, all problems
  reported at once, deep merge, dotted lookup, fingerprint stability.
- `tests/test_demo.py` - every demo section runs and the CLI writes JSON.

## Limitations

- The phantom is geometric, not anatomical: it has no sulci, no bone marrow, no
  scanner-specific noise correlation. A model trained on it will not transfer to
  real CT.
- Classification is patch-level (lesion present / absent). There is no lesion
  localisation, segmentation or detection head.
- Only axial slices are used; coronal and sagittal sampling exists in
  `take_slice`/`slice_for_spec` but is not exercised by the training path.
- The CNN is deliberately tiny; it is not a baseline for clinical accuracy.
- `requirements.txt` pins minimum versions rather than exact versions, so a
  future NumPy/PyTorch release may need small adjustments.
