# mavlink-cross-campaign

Cross-campaign generalisation of MAVLink message-ID intrusion detection
(ICCE 2027, SS-02 — Project 2 of `ICCE2027_SS02_Project_Blueprints.docx`).

Environment: `conda activate mavx` and **`export PYTHONNOUSERSITE=1`** (the user site
`~/.local` has packages that otherwise leak in). torch 2.4.1; the GPU driver is currently
not reachable (`nvidia-smi` fails), so everything runs on CPU.

## Data (data/raw/, not committed)

| Campaign | Source | Files | Stack | Attacks |
|---|---|---|---|---|
| **OLD** (MUVIDS, 2021) | HCRL Msg-ID Google Drive pcap, `hitl_100000_*` | normal ×2, heartbeat ×2, ping ×2, request ×2 (100k frames each) | HITL, PX4-style, 192.168.20.x | heartbeat / ping / request flooding |
| **NEW** (GUIDE, 2022) | same archive, `*(filtered).pcapng` | normal, heartbeat, ping, request, suspension, corruption (1 file each) | HITL, 163.152.x | same 3 floods + suspension + corruption |
| **SHADOW** (GALAXY, 2025) | HCRL Dropbox | 36 pcapng: {Direct,Relay}×{Armed,Disarmed}×{Benign,Attack} | ArduPilot SITL + Mission Planner, 192.168.0.x | passive relay MITM (Shadow-GCS) |

All captures are MAVLink 2.0 UAV→GCS frames (sysid 1, compid 1); the floods are sent *to*
the UAV and are visible only through the UAV's response stream.

## Step C2 go/no-go — decision: **GO** (4 Oct 2026)

1. *Message IDs recoverable from both/all corpora?* **Yes** — parsed directly from pcap
   (`src/io_mavlink.py`), no derived features needed.
2. *ID ranges consistent with the MAVLink standard space?* **Yes.** Vocabularies: OLD 24, NEW 29,
   SHADOW 34. **OLD ⊂ NEW** (all 24 OLD IDs occur in NEW; NEW-only: 46, 76, 77, 83, 253).
   SHADOW shares only 15 IDs with OLD and 18 with NEW and has 16 SHADOW-only IDs (ArduPilot
   dialect, e.g. 11030) → HCRL↔SHADOW transfer will be OOV-dominated, as the blueprint warned.
3. *Attack periods localisable?* **No — labels are per file** in all three corpora (each attack
   file is captured entirely under attack; per-decile rates confirm the attack is uniform
   through each file). This is the labelling GUIDE, MUVIDS and GALAXY all use, and a
   temporal-block split works, so it is a usable split → GO, with the caveat stated in the paper.

### Design change vs the blueprint

The Msg-ID archive contains **two campaigns** (OLD 2021, NEW 2022) with the **same three
attack types**. Primary experiment = OLD ↔ NEW transfer with the attack type held fixed
(removes the cross-attack confound that a Msg-ID → SHADOW design has: flooding vs relay).
SHADOW becomes a third, far (cross-stack) campaign: benign-only false-alarm transfer + OOV analysis.

### Observations that matter for the paper

- Ping/request floods: UAV output jumps from ~365 to ~2,100 frames/s and HEARTBEAT share drops
  to 0 in BOTH campaigns → a rate rule should transfer.
- Heartbeat flood: OLD rate drops to 150–350 frames/s; NEW rate is ~364 (≈ normal). **Same attack
  name, different footprint across campaigns** → expected transfer failure; good error-analysis story.
- SHADOW attack files contain a second UDP flow (~130 frames from another source port, MAVProxy's
  own messages) that never appears in benign files → a candidate dataset shortcut (count of flows /
  source ports), worth reporting.

## GALAXY paper check (novelty, blueprint II §5B.3)

Lee & Kim, PRDC 2025, doi 10.1109/PRDC67299.2025.00014. Evaluates on SHADOW-GCS with an 8:2
**file-based split within the dataset**, and on the MUVIDS flooding dataset **separately**
(Table VIII). **No train-on-one / test-on-other experiment** → cross-campaign transfer is not
pre-empted. Notes: labels are by source MAC (benign 08:00:27:74:10:db, attack 08:00:27:38:27:83);
GALAXY window 30, sequence length 128; reports MUVIDS recall 0.878 on SHADOW-GCS (Table III).

## Results (4 Oct 2026) — see paper/main.pdf

- Within-campaign supervised AUC 1.00 (HCRL) / 0.93–0.94 (SHADOW); OLD↔NEW transfer 0.74–0.95;
  any transfer involving SHADOW 0.15–0.52 (mostly below chance). Trivial rate/histogram: 0.73–0.85 across stacks.
- NEW heartbeat flood: rate 362 vs 366 fps, JSD 0.007 to normal → no footprint; supervised 1.00 within, LSTM/CNN 0.53/0.58 transferred.
- Recording-identity control: two BENIGN OLD recordings separated with AUC 0.61–0.88 under a within-file split.
- Timing features: within-SHADOW 0.94 → 1.00, cross-campaign unchanged; per-campaign z-scoring does not help (H3 not supported).

## Reproduce

```bash
conda activate mavx; export PYTHONNOUSERSITE=1
python -m src.inventory                     # data.yaml + parquet cache
pytest -q tests
python experiments/run_matrix.py --seeds 5  # experiment (b), ~8 min on 24 cores
python experiments/run_recording_control.py
python experiments/run_errors.py
bash experiments/run_ablation_sensitivity.sh
python experiments/make_tables.py && python -m src.figures
paper/build.sh                              # needs conda env texenv (tectonic)
```
