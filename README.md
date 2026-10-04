# road-masquerade-floor

Payload-level trivial-baseline floor for the ROAD CAN masquerade captures
(ICCE 2027, SS-02 — Project 1 of `ICCE2027_SS02_Project_Blueprints.docx`).

Environment: `conda activate roadmasq` (python 3.11, pins in `requirements.txt`). CPU only.

## Layout

| Path | Contents |
|---|---|
| `configs/captures.yaml` | Real capture filenames, families, injection intervals (Step C2) |
| `configs/experiment.yaml` | Windows, thresholds, seeds |
| `data/raw/` | `road.zip` + extraction. Never edited. Not committed. |
| `data/interim/` | Parquet caches (regenerable) |
| `src/detectors.py` | T1 value-set, T2 range, T3 signal envelope, T4 payload novelty |
| `src/evaluate.py` | Windowing under interval labels, AUC/PR-AUC, det. rate @1% FPR, bootstrap CIs |
| `tests/` | `pytest -q tests` |

## Invariant

Detectors are fitted on ambient captures only. `fit()` raises
`AttackDataInFitError` if any frame with `family != "ambient"` reaches it.

## Day-1 log

- **Comparator repo (Step D1).** `pmoriano/benchmarking-unsupervised-online-IDS-masquerade-attacks`
  imports `signal_based_preprocess_functions` from a private path
  (`/home/cloud/Projects/CAN/detect/`) that is not in the repository, and hard-codes
  `/home/cloud/...` data paths. It cannot run as published. Per the blueprint's
  fallback, comparator numbers are **quoted** from the papers unless that module is
  obtained from the authors. What *is* reusable is their protocol: signals resampled
  at 100 Hz, sliding windows, a window is positive iff it overlaps the injection
  interval, AUC-ROC over windows. `src/evaluate.py` follows this overlap rule.
- **Blueprint gaps.** §7.3 (software environment) and §8.4 (pseudocode) are empty in
  the .docx; the implementation follows the §8.2 definitions and §9.2 metrics.
- **Download.** Zenodo single-stream ≈ 85 KB/s; `data/raw/pdl.sh` does a 16-way
  ranged download. Official checksum: `md5:cab184cfc2fe12c0834bc46188c0f330`.

## Novelty / related-work log (4 Oct 2026)

- **Guerra et al., arXiv:2408.17235 (2024)** — NOT in blueprint Table I; add it. Supervised
  per-frame ML/DL on ROAD (LCCDE, RF, LSTM, DCNN, TAN). Restates that masquerade is
  produced in post-processing, but builds no trivial/payload-rule floor. Their binary
  LSTM "perfectly classif[ies] almost every attack type" (masquerade included) —
  consistent with H1 and useful as a quoted comparator. Does not pre-empt.
- Project 2: "UAV Intrusion Detection Based on iTransformer for MAVLink Message ID
  Sequence" (ACM, doi 10.1145/3789456.3789458, 2026) — single-corpus 4-class flood
  classification; no cross-campaign transfer. Add to Project 2 Table I. Does not pre-empt.

## Data facts established in Step C2/C3 (correct the blueprint)

- **13 masquerade captures, 5 attack types** (not "five captures"): correlated_signal ×3,
  max_speedometer ×3, reverse_light_off ×3, reverse_light_on ×3, max_engine_coolant_temp ×1.
  Each is paired with its fabrication original. Total 33 attacks = 13 fab + 13 masq + 3 fuzzing + 4 accelerator.
- **Signal CSVs** exist for the 13 masquerade + 4 accelerator captures + all 12 ambient, not for
  fabrication. `signal_extractions/ambient/` ALSO contains copies of the attack CSVs — never fit on that folder.
- **Labels are per message, not only per interval**: signal CSVs carry `Label`. Our reconstructed
  per-frame labels on raw logs match ROAD's exactly on all 13 masquerade captures (tests/test_io.py).
- **Construction verified in data**: non-target traffic is byte-identical between each masquerade
  capture and its fabrication twin; injected frame counts match (±2 at interval edges).
- max_engine_coolant has only **43** injected frames — wide CIs expected.

## First result (4 Oct, `results/floor.csv`)

Fit on 10 ambient captures; FPR calibrated on 2 held-out (basic_short, extended_short); 1 s windows, 0.25 s stride.

- **Solved by trivial rules** (T2 byte range, window AUC 1.00, ~97–99% detection @1% FPR):
  correlated_signal ×3, max_speedometer ×3. Coolant: AUC 1.00 but 0% @1% FPR because its
  excursion (b5 = 255 vs ambient max 162) is smaller than benign held-out excursions (up to 128 on 0x207/0x6fc).
- **Resist all trivial rules** (AUC ≈ 0.5): reverse_light_off ×3, reverse_light_on ×3 — the
  injected byte (b2 of 0xd0 = 0x04/0x0C) is one of only two values ever seen in ambient; the attack is a
  *context* anomaly (right value, wrong driving state).
- **Blueprint H2 is false as stated.** Correlated-signal does NOT resist: 3 of its 4 injected
  wheel-speed high bytes (0x59, 0x45, 0xFF) exceed the ambient maximum (53), highway driving included.
- Fabrication twins give identical numbers → the effect is inherited from the fabrication payloads.

## Final results (4 Oct 2026) — see paper/main.pdf

- vs published (Moriano et al., JISA 2026, quoted): delta-over-trivial −0.26 / 0.00 / −0.12 on speedometer /
  coolant / correlated; +0.28 / +0.25 on reverse light off / on.
- IsolationForest control: worse than T2 on extreme-value types; only payload model that separates reverse-light-on (0.86).
- Ablation: excluding the target AID → 0.24–0.71; only the bytes written by the attack carry the signal.
- Sensitivity: T2 = 1.00 on extreme types at every window length and with 10% of ambient data.
- CrySyS (experiment e, 26 recordings, 312 masquerade logs, leave-one-recording-out): T2 AUC 0.49; only 1.8% of
  modified frames leave the ambient range → the shortcut is specific to ROAD's construction.

## Reproduce

```bash
conda activate roadmasq; export PYTHONNOUSERSITE=1
bash data/raw/pdl.sh && python -m src.cli inventory && python -m src.cli cache
pytest -q tests
python experiments/run_floor.py; python experiments/run_controls.py
python experiments/run_ablation.py; python experiments/run_sensitivity.py
bash data/raw/crysys/pdl.sh && python experiments/run_crysys.py
python experiments/make_tables.py && python -m src.figures
paper/build.sh                     # needs conda env texenv (tectonic)
```
