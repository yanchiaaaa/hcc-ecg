# ICD Graph Embeddings

This directory contains the code used to build the ICD-10-CM hierarchy graph and train Poincare-ball ICD graph embeddings for HCC-ECG.

## Build the ICD Graph

```bash
python icd_graph/build_icd_graph.py \
  --xml_file icd_graph/icd10cm/icd10cm_tabular_2024.xml \
  --output_path icd_graph/icd_graph_data.pt
```

The output file contains:

- `code_to_id`: ICD code to graph node index.
- `id_to_code`: graph node index to ICD code.
- `edge_index`: directed parent-child edges with shape `(2, E)`.

## Train Hyperbolic Embeddings

```bash
python icd_graph/train_icd_hyperbolic.py \
  --data_path icd_graph/icd_graph_data.pt \
  --save_dir icd_graph/checkpoints \
  --embed_dim 768 \
  --num_neg 20 \
  --epochs 500 \
  --batch_size 2048
```

The best checkpoint is saved as `icd_hyperbolic_best.pth`. HCC-ECG loads this checkpoint together with `icd_graph_data.pt` through `icd_graph_loader.py`.

## Evaluate Embeddings

```bash
python icd_graph/evaluate_icd_hyperbolic.py \
  --checkpoint_path icd_graph/checkpoints/icd_hyperbolic_best.pth \
  --data_path icd_graph/icd_graph_data.pt
```

Add `--run_tsne --output_dir icd_graph/checkpoints` to save a TSNE visualization.
