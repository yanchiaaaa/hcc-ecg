# Generation

Use the unified generation script:

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

Supported released model types:

- `joint`
- `text_tabular`
- `icd_tabular`
- `uncond`
