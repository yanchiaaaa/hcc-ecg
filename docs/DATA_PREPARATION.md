# Data Preparation

HCC-ECG training expects preprocessed ECG latent tensors paired with condition dictionaries. Each sample should contain at least:

- `data`: VAE latent tensor, or ECG waveform when training/evaluating components that operate on raw waveforms.
- `label.text`: machine/report text.
- `label.text_embed`: precomputed clinical text embedding.
- `label.icd`: ICD-10 code list.
- `label.icd_text`: ICD concept text list.
- `label.age`
- `label.gender`
- `label.hr`

Recommended preparation order:

1. Download the source datasets listed in the main README.
2. Run `dataset/preprocess_records_with_diag.py` to create the cleaned ICD diagnosis table.
3. Use `dataset/mimic_iv_ecg_dataset.py` with a local clinical text encoder to read ECG waveforms, clean/resample them, estimate heart rate, and attach text/ICD/tabular conditions.
4. Train or provide the VAE checkpoint.
5. Run the preprocessing pipeline that encodes ECG waveforms into VAE latents and saves train/validation/test `.pt` files.
6. Build `icd_graph/icd_graph_data.pt` and train `icd_hyperbolic_best.pth` as described in `icd_graph/README.md`.

Update the paths in `config/HCCECG.json` after these files are prepared.
