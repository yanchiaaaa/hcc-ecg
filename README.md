# HCC-ECG

This repository contains the code required to train and evaluate HCC-ECG, a conditional ECG generation model using ICD-graph, clinical text, and tabular conditions.

## Repository Scope

This release includes the model, training, generation, preprocessing, evaluation, and qualitative visualization code needed for paper reproduction.

## Main Components

- `HCCECG.py`: main training entry point for the full HCC-ECG model and selected ablations.
- `module/`: DiT-based ECG generator architectures.
- `utils/`: training loops and EMA utilities.
- `VAE/`: ECG VAE model and training code.
- `dataset/`: dataset and preprocessing utilities.
- `icd_graph/`: ICD-10-CM graph construction and hyperbolic ICD embedding training.
- `test_scripts/generate_hcc_ecg.py`: unified generation script.
- `test_scripts/evaluate_hcc_ecg.py`: generated ECG evaluation.
- `test_scripts/evaluate_interlead_correlation.py`: inter-lead correlation evaluation.
- `test_scripts/visualize_12lead_ecg.py`: 12-lead ECG visualization.
- `scripts/`: qualitative LBBB/RBBB generation scripts used for paper visualization.
- `config/`: example training configuration files.

## Included Ablations

Only the ablation settings used for the released paper package are included:

- Text + tabular conditioning
- ICD + tabular conditioning

## Data

We evaluate HCC-ECG on MIMIC-IV-ECG and PTB-XL.

**MIMIC-IV-ECG.** We use the MIMIC-IV-ECG matched subset v1.0 together with MIMIC-IV-ECG-Ext-ICD v1.0.1. ECG waveforms are loaded from the WFDB paths in `record_list.csv`, and diagnostic reports are obtained by concatenating the non-empty `report_0`-`report_17` fields in `machine_measurements.csv`. Each original ECG is a 12-lead, 500 Hz, 10-second recording. We clean each lead with `neurokit2.ecg_clean`, resample the signal to 1024 time points, and filter records with abnormal amplitudes, failed heart-rate estimation, or missing age. We further filter ICD-10 diagnoses by ECG-morphology relevance, retaining codes likely to affect ECG patterns and excluding background chronic or vascular diagnoses with weak direct ECG specificity. Reports that are explicitly normal and contain no abnormal keywords are assigned a normal ECG label when no retained ICD code is available. We use the provided patient-level 20-fold assignment, with folds 0-17, 18, and 19 for training, validation, and testing, resulting in 308,404, 17,033, and 17,107 valid samples, respectively.

Download links:

- MIMIC-IV-ECG matched subset v1.0: https://physionet.org/content/mimic-iv-ecg/1.0/
- MIMIC-IV-ECG-Ext-ICD v1.0.1: https://physionet.org/content/mimic-iv-ecg-ext-icd/1.0.1/

**PTB-XL.** For PTB-XL v1.0.3, we use the high-resolution 500 Hz ECG records specified by `filename_hr`. Signals are cleaned lead-wise and resampled to the same 1024-point format. Reports are taken from the `report` field in `ptbxl_database.csv`. Since PTB-XL does not provide EHR-level ICD diagnoses, we map `scp_codes` to ICD-like codes using a curated SCP-to-ICD mapping and keep ECG-relevant labels such as normal rhythm, arrhythmia, ischemia or infarction, conduction disorders, abnormal ECG findings, and selected electrolyte or pacemaker-related conditions. We follow the official patient-level `strat_fold` split, using folds 1-8, 9, and 10 for training, validation, and testing, resulting in 17,274, 2,162, and 2,175 valid samples, respectively.

Download link:

- PTB-XL v1.0.3: https://physionet.org/content/ptb-xl/1.0.3/

## Provided Checkpoints

The following pretrained artifacts can be placed under `checkpoints/`:

- `checkpoints/hcc_ecg_full_ema.pth`: EMA checkpoint of the full HCC-ECG generator.
- `checkpoints/ecg_vae_ema.pth`: ECG VAE checkpoint used to decode generated latents into 12-lead ECG waveforms.
- `checkpoints/icd_graph_data.pt`: ICD-10-CM graph structure with `code_to_id`, `id_to_code`, and `edge_index`.
- `checkpoints/icd_hyperbolic_best.pth`: pretrained hyperbolic ICD graph embeddings.

When these files are available, the default config paths in `config/HCCECG.json` are already aligned with the checkpoint names above.

## Before Training

Before launching HCC-ECG training, prepare the following artifacts and update the corresponding paths in `config/HCCECG.json`.

1. Install the Python dependencies listed in `requirements.txt`. The ICD graph embedding trainer also requires `geoopt`.
2. Download the source ECG datasets: MIMIC-IV-ECG matched subset v1.0, MIMIC-IV-ECG-Ext-ICD v1.0.1, and/or PTB-XL v1.0.3.
3. Build the cleaned diagnosis table with `dataset/preprocess_records_with_diag.py`. This step filters ECG-morphology-relevant ICD-10 labels and produces the diagnosis table consumed by `dataset/mimic_iv_ecg_dataset.py`.
4. Prepare the clinical text encoder used for report embeddings, for example a local Bio_ClinicalBERT checkpoint, and set the model path in the preprocessing script.
5. Train or provide the ECG VAE checkpoint, then preprocess ECG records into VAE-latent training and validation files. The DiT training configs expect `dataset_path`, `val_dataset_path`, and `vae_path` to point to these artifacts.
6. Use the provided ICD graph artifacts in `checkpoints/`, or rebuild them with the commands below. The DiT config fields `icd_graph_path` and `icd_embeddings_path` should point to `checkpoints/icd_graph_data.pt` and `checkpoints/icd_hyperbolic_best.pth`, respectively.
7. Set `checkpoints_dir`, GPU device, batch size, and other training hyperparameters in the config file.

## Typical Usage

Train full model:

```bash
python HCCECG.py config/HCCECG.json
```

Build and train ICD graph embeddings:

```bash
python icd_graph/build_icd_graph.py \
  --xml_file icd_graph/icd10cm/icd10cm_tabular_2024.xml \
  --output_path icd_graph/icd_graph_data.pt

python icd_graph/train_icd_hyperbolic.py \
  --data_path icd_graph/icd_graph_data.pt \
  --save_dir icd_graph/checkpoints \
  --embed_dim 768 \
  --num_neg 20
```

Generate samples:

```bash
python -m test_scripts.generate_hcc_ecg \
  --test_data_path /path/to/test_latents.pt \
  --model_type joint \
  --checkpoint_path checkpoints/hcc_ecg_full_ema.pth \
  --output_dir /path/to/output \
  --batch_size 64 \
  --scale 1.5 \
  --num_sampling_steps 35
```

## Path Configuration

Update dataset paths in `config/` and command-line arguments according to your local data and output directories. The provided checkpoint paths assume the pretrained artifacts are stored under `checkpoints/`.
