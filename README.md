# ICCE 2027 — two IDS evaluation papers (private working repo)

| Folder | Paper |
|---|---|
| `road-masquerade-floor/` | *How Much of the ROAD Masquerade Benchmark Is a Payload Shortcut?* (CAN bus, cars) |
| `mavlink-cross-campaign/` | *Does MAVLink Intrusion Detection Survive a Change of Campaign?* (drones) |

Top level: the project blueprint (`ICCE2027_SS02_Project_Blueprints.docx`), the two draft PDFs, and
`outreach_drafts.md` (unsent emails to the session chair and the dataset owners).

Each project folder has its own README with results, environment and reproduce steps, and keeps its
full commit history (merged here with `git subtree`).

## What is NOT in this repo

- **Datasets** (about 20 GB; files exceed GitHub's 100 MB limit, and the HCRL MAVLink datasets state no
  licence). Download scripts: `road-masquerade-floor/data/raw/pdl.sh` (ROAD, Zenodo, CC-BY-4.0) and
  `road-masquerade-floor/data/raw/crysys/pdl.sh` (CrySyS, figshare, CC0). MAVLink sources are listed in
  `mavlink-cross-campaign/README.md`.
- **Per-byte NLL dumps** (`road-masquerade-floor/results/nll/`, 742 MB): regenerate with
  `experiments/dump_ctx_nll.py` from the included model checkpoints.
- **The GALAXY paper PDF**: an IEEE Xplore download under an institutional licence, not ours to redistribute.

## Updating this repo from the working folders

`./sync_from_working_folders.sh` pulls new commits from `/home/ailab/VM/Lav/{road-masquerade-floor,mavlink-cross-campaign}`,
refreshes the result files and PDFs, commits, and pushes.
