# Model Training

Before training, make sure the following config fields in `config/HCCECG.json` point to valid files or directories:

- `dependencies.dataset_path`: training VAE-latent dataset.
- `dependencies.val_dataset_path`: validation VAE-latent dataset.
- `dependencies.vae_path`: trained ECG VAE checkpoint.
- `dependencies.icd_graph_path`: ICD graph file built by `icd_graph/build_icd_graph.py`.
- `dependencies.icd_embeddings_path`: hyperbolic ICD embedding checkpoint trained by `icd_graph/train_icd_hyperbolic.py`.
- `dependencies.checkpoints_dir`: output directory for DiT checkpoints.

Single-GPU training:

```bash
python HCCECG.py config/HCCECG.json
```

Multi-GPU training:

```bash
torchrun --nproc_per_node=NUM_GPUS HCCECG.py config/HCCECG.json
```

Ablation configs are available at `config/ablation_text_tabular.json` and `config/ablation_icd_tabular.json`.
