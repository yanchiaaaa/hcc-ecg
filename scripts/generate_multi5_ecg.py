"""
Shared HCC-ECG generation helpers.

This module provides the full-model factory, checkpoint loading, CFG forward
wrapper, VAE decoder loading, latent decoding, and condition normalization
utilities used by the qualitative visualization scripts.
"""

import sys
import os
import argparse
import json
import time

import torch
import torch.utils.data
from tqdm import tqdm

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from diffusers import DPMSolverMultistepScheduler
from icd_graph_loader import ICDGraphEmbeddingLoader
from VAE.vae_model import VAE_Decoder

def create_model(model_kwargs):
    from module.dit_tri_stream_noproj_newcfg import DiT_TripleStream_ECG
    return DiT_TripleStream_ECG(
        in_channels=model_kwargs['in_channels'],
        seq_length=model_kwargs['seq_length'],
        hidden_size=model_kwargs['hidden_size'],
        depth=model_kwargs['depth'],
        num_heads=model_kwargs['num_heads'],
        icd_embed_dim=model_kwargs['icd_embed_dim'],
        text_embed_dim=model_kwargs['text_embed_dim'],
        mlp_ratio=model_kwargs.get('mlp_ratio', 4.0),
        dropout=0.0,
        use_rope=model_kwargs.get('use_rope', False),
    )


def load_checkpoint(model, checkpoint_path, use_ema=True):
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if use_ema and 'ema_state_dict' in checkpoint and checkpoint['ema_state_dict'] is not None:
        print("  Using EMA weights")
        state_dict = checkpoint['ema_state_dict']
        if isinstance(state_dict, dict) and 'shadow' in state_dict:
            state_dict = state_dict['shadow']
        model.load_state_dict(state_dict)
    elif 'model_state_dict' in checkpoint:
        print("  Using standard model weights")
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("  Using raw state_dict")
        model.load_state_dict(checkpoint)

    epoch_info = checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'
    print(f"  Checkpoint epoch: {epoch_info}")
    return model


def forward_full_model(model, x, t, batch_data, device):
    icd_embeds = batch_data['icd_embeds']
    text_embeds = batch_data['text_embeds']
    icd_mask = batch_data['icd_mask']
    age = batch_data['age']
    gender = batch_data['gender']
    hr = batch_data['hr']

    null_icd = torch.zeros_like(icd_embeds)
    null_text = torch.zeros_like(text_embeds)
    null_mask = torch.zeros_like(icd_mask)

    icd_in = torch.cat([null_icd, icd_embeds], dim=0)
    text_in = torch.cat([null_text, text_embeds], dim=0)
    mask_in = torch.cat([null_mask, icd_mask], dim=0)
    age_in = torch.cat([torch.zeros_like(age), age], dim=0)
    gender_in = torch.cat([torch.zeros_like(gender), gender], dim=0)
    hr_in = torch.cat([torch.zeros_like(hr), hr], dim=0)

    return model(x, t, icd_in, text_in, age_in, gender_in, hr_in, icd_mask=mask_in)


def prepare_batch_data(y, device, age_mean=64.1502, age_std=17.6533, hr_mean=81.3950, hr_std=21.4822):
    data = {}
    if 'icdgraph_embed' not in y or 'icdgraph_mask' not in y:
        raise KeyError("collate_fn must generate 'icdgraph_embed' and 'icdgraph_mask' from ICD codes")

    data['icd_embeds'] = y['icdgraph_embed'].to(device)
    data['icd_mask'] = y['icdgraph_mask'].to(device)

    if 'text_embed' in y:
        text_embed = y['text_embed'].to(device)
    else:
        batch_size = data['icd_embeds'].shape[0]
        text_embed = torch.zeros(batch_size, 1, 768, device=device)

    if text_embed.dim() == 2:
        text_embed = text_embed.unsqueeze(1)
    data['text_embeds'] = text_embed

    age_tensor = y['age'].to(device).float()
    if age_tensor.dim() == 3:
        age_tensor = age_tensor.squeeze(-1)
    elif age_tensor.dim() == 1:
        age_tensor = age_tensor.view(-1, 1)
    data['age'] = (age_tensor - age_mean) / age_std

    gender_tensor = y['gender'].to(device).float()
    if gender_tensor.dim() == 3:
        gender_tensor = gender_tensor.squeeze(-1)
    elif gender_tensor.dim() == 1:
        gender_tensor = gender_tensor.view(-1, 1)
    data['gender'] = gender_tensor

    hr_tensor = y['heart rate'].to(device).float()
    if hr_tensor.dim() == 3:
        hr_tensor = hr_tensor.squeeze(-1)
    elif hr_tensor.dim() == 1:
        hr_tensor = hr_tensor.view(-1, 1)
    data['hr'] = (hr_tensor - hr_mean) / hr_std

    data['icd_codes'] = y.get('icd_codes', None)
    return data


def load_vae_decoder(vae_path: str, device: torch.device) -> VAE_Decoder:
    print(f"\n🧬 Loading VAE decoder from: {vae_path}")
    decoder = VAE_Decoder()
    state = torch.load(vae_path, map_location=device, weights_only=False)
    decoder.load_state_dict(state['decoder'])
    decoder.eval().to(device)
    print("  VAE decoder loaded")
    return decoder


def decode_latents(latents: torch.Tensor, decoder: VAE_Decoder, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    if latents.dim() == 2:
        latents = latents.unsqueeze(0)

    chunks = []
    with torch.no_grad():
        n = latents.shape[0]
        for i in range(0, n, batch_size):
            batch = latents[i:i + batch_size].to(device).float()
            chunks.append(decoder(batch).cpu())
    return torch.cat(chunks, dim=0)


def _json_safe(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def plot_ecg_12lead(waveform, save_path: str, sample_rate: float = 102.4):
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(save_path) or '.', exist_ok=True)
    arr = waveform.detach().cpu().float().numpy() if torch.is_tensor(waveform) else waveform

    try:
        import ecg_plot

        try:
            ecg_plot.plot(arr, sample_rate=sample_rate)
        except Exception:
            ecg_plot.plot(arr.transpose(1, 0), sample_rate=sample_rate)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        return
    except Exception:
        plt.close('all')

    if arr.ndim != 2:
        raise ValueError(f"ECG waveform must be 2D, got shape={arr.shape}")
    if arr.shape[0] == 12 and arr.shape[1] != 12:
        arr = arr.transpose(1, 0)
    if arr.shape[1] < 12:
        raise ValueError(f"Expected 12-lead ECG waveform, got shape={arr.shape}")

    lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    t = [i / sample_rate for i in range(arr.shape[0])]
    fig, axes = plt.subplots(6, 2, figsize=(12, 10), sharex=True)
    axes = axes.reshape(-1)

    for lead_idx in range(12):
        ax = axes[lead_idx]
        ax.plot(t, arr[:, lead_idx], linewidth=0.8)
        ax.set_title(lead_names[lead_idx], fontsize=9)
        ax.grid(True, linewidth=0.3, alpha=0.5)

    axes[-2].set_xlabel('Time (s)')
    axes[-1].set_xlabel('Time (s)')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def tensor_from_any(value, dtype=torch.float32):
    if torch.is_tensor(value):
        return value.detach().cpu().to(dtype=dtype)
    return torch.as_tensor(value, dtype=dtype)


def normalize_embedding(value, name: str) -> torch.Tensor:
    emb = tensor_from_any(value, dtype=torch.float32)
    if emb.dim() == 1:
        emb = emb.unsqueeze(0)
    if emb.dim() != 2 or emb.shape[-1] != 768:
        raise ValueError(f"{name} must have shape (768,) or (L, 768), got {tuple(emb.shape)}")
    return emb.contiguous()


def normalize_scalar(value, default=0.0) -> float:
    if value is None:
        return float(default)
    if torch.is_tensor(value):
        value = value.detach().cpu().view(-1)[0].item()
    elif isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    return float(value)


def normalize_gender_value(value) -> str:
    if torch.is_tensor(value):
        return 'M' if float(value.detach().cpu().view(-1)[0].item()) >= 0.5 else 'F'
    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if isinstance(value, str):
        return value.strip().upper()[:1] if value.strip() else 'F'
    return 'M' if float(value) >= 0.5 else 'F'


def parse_icd_codes(value):
    if value is None:
        return []
    if isinstance(value, str):
        try:
            import ast
            parsed = ast.literal_eval(value)
            value = parsed
        except Exception:
            value = [value]
    elif not isinstance(value, (list, tuple)):
        value = [value]
    if len(value) == 1 and isinstance(value[0], (list, tuple)):
        value = value[0]
    return [str(v).replace('.', '').strip().upper() for v in value if str(v).strip()]


def normalize_condition_label(label_item: dict) -> dict:
    """Normalize label fields for HCC-ECG conditioning.

    OptimizedCollateFn reads ICD graph conditions from label['icd']. When a
    dataset keeps diagnosis text in label['icd'] and true ICD-10 codes in
    label['icd_codes'], use icd_codes for generation while preserving the
    original diagnosis text under diag_text_list.
    """
    label_item = label_item.copy()

    if 'icd_codes' in label_item and label_item['icd_codes'] is not None:
        original_icd = label_item.get('icd', None)
        if original_icd is not None and 'diag_text_list' not in label_item:
            label_item['diag_text_list'] = original_icd
        label_item['icd'] = parse_icd_codes(label_item['icd_codes'])
    else:
        label_item['icd'] = parse_icd_codes(label_item.get('icd', []))

    if not label_item['icd']:
        raise ValueError('HCC-ECG generation requires non-empty ICD codes in label["icd_codes"] or label["icd"]')

    if 'text_embed' in label_item and label_item['text_embed'] is not None:
        label_item['text_embed'] = normalize_embedding(label_item['text_embed'], 'text_embed')
    if 'age' in label_item:
        label_item['age'] = normalize_scalar(label_item['age'])
    if 'hr' in label_item:
        label_item['hr'] = normalize_scalar(label_item['hr'])
    elif 'heart rate' in label_item:
        label_item['hr'] = normalize_scalar(label_item['heart rate'])
    if 'gender' in label_item:
        label_item['gender'] = normalize_gender_value(label_item['gender'])
    return label_item

def generate_ecg_dataset(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else 'cpu')
    target_classes = set(args.target_classes)
    num_repeats = args.num_repeats

    print("=" * 70)
    print("  ECG Directed Dataset Generation Pipeline (VAE-Decoded Waveform)")
    print(f"  Target Classes    : {sorted(target_classes)}")
    print(f"  Repeats per Sample: {num_repeats}x")
    print("  Model Type        : full joint (ICD + Text + Tabular)")
    print(f"  Input Dataset     : {args.input_path}")
    print(f"  Output Dataset    : {args.output_path}")
    print(f"  VAE Decoder       : {args.vae_path}")
    print(f"  Batch Size        : {args.batch_size}")
    print("=" * 70)

    print(f"\nLoading input dataset...")
    input_data = torch.load(args.input_path, map_location='cpu', weights_only=False)
    print(f"  Loaded {len(input_data)} total samples from dataset")

    data_list = []
    idx_list = []
    class_stats = {c: 0 for c in target_classes}

    print(f"\n🔍 Filtering and duplicating target classes...")
    for idx, sample_data in input_data.items():
        label_item = sample_data['label'].copy()
        class_id = label_item.get('label', -1)

        if class_id in target_classes:
            data_item = sample_data['data']
            if not torch.is_tensor(data_item):
                data_item = torch.as_tensor(data_item)
            if data_item.dim() == 2:
                data_item = data_item.unsqueeze(0)

            label_item = normalize_condition_label(label_item)

            for r in range(num_repeats):
                data_list.append((data_item, label_item))
                new_idx = f"{idx}_gen_{r}"
                idx_list.append(new_idx)
                class_stats[class_id] += 1

    total_to_generate = len(data_list)
    print(f"  Valid conditions found. Will generate {total_to_generate} samples in total.")
    for cid, count in class_stats.items():
        print(f"    ↳ Class {cid}: {count} samples planned")

    if total_to_generate == 0:
        print("No samples matched the target classes. Exiting.")
        return {}

    print(f"\nLoading ICD graph embeddings...")
    icd_loader = ICDGraphEmbeddingLoader(
        graph_data_path=args.icd_graph_path,
        embeddings_path=args.icd_embeddings_path,
        special_tokens=['NORM'],
        logger=None,
    )
    icd_embeddings, code_to_id = icd_loader.load()
    icd_embeddings = icd_embeddings.to(device)

    try:
        from optimized_collate_fn import OptimizedCollateFn
    except ImportError:
        sys.path.insert(0, _PROJECT_DIR)
        from optimized_collate_fn import OptimizedCollateFn

    collate_fn_handler = OptimizedCollateFn(
        icd_graph_embeddings=icd_embeddings,
        code_to_id=code_to_id,
        use_precomputed_text=True,
        enable_icd_cache=True,
        verbose=False,
    )

    print(f"\n🤖 Loading generation model...")
    model_kwargs = {
        'in_channels': 4,
        'seq_length': 128,
        'hidden_size': args.hidden_size,
        'depth': args.depth,
        'num_heads': args.num_heads,
        'icd_embed_dim': args.icd_embed_dim,
        'text_embed_dim': args.text_embed_dim,
        'mlp_ratio': 4.0,
        'use_rope': args.use_rope,
    }
    model = create_model(model_kwargs)
    model = load_checkpoint(model, args.checkpoint_path, use_ema=args.use_ema)
    model = model.to(device).eval()

    decoder = load_vae_decoder(args.vae_path, device)

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type='dpmsolver++',
        solver_order=2,
    )

    print(f"\n🎲 Generating ECG waveforms...")
    output_data = {}
    visualization_manifest = []
    visualized_count = 0
    vis_dir = args.vis_dir or f"{os.path.splitext(args.output_path)[0]}_vis"
    if args.num_visualize > 0:
        os.makedirs(vis_dir, exist_ok=True)
        print(f"  First {args.num_visualize} decoded samples will be plotted to: {vis_dir}")

    total_generated = 0
    t_start = time.time()

    with torch.no_grad():
        batch_loader = torch.utils.data.DataLoader(
            data_list,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=collate_fn_handler,
        )

        global_idx = 0
        for batch_idx, (batch_data_raw, batch_labels) in enumerate(tqdm(batch_loader, desc='Generating')):
            curr_bs = batch_data_raw.shape[0] if batch_data_raw is not None else len(batch_labels['icd'])
            prepared_batch = prepare_batch_data(batch_labels, device)

            xi = torch.randn(curr_bs, 4, 128, device=device)
            scheduler.set_timesteps(args.num_sampling_steps)

            for t_step in scheduler.timesteps:
                xi_in = torch.cat([xi, xi], dim=0)
                t_in = t_step.expand(curr_bs * 2).to(device)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    noise_pred = forward_full_model(model, xi_in, t_in, prepared_batch, device)

                eps_uncond, eps_cond = noise_pred.chunk(2)
                noise_guided = eps_uncond + args.scale * (eps_cond - eps_uncond)

                if args.rescale_phi > 0.0:
                    std_dims = [1, 2, 3] if eps_cond.dim() == 4 else [1, 2]
                    std_cond = eps_cond.std(dim=std_dims, keepdim=True)
                    std_guided = noise_guided.std(dim=std_dims, keepdim=True)
                    noise_rescaled = noise_guided * (std_cond / (std_guided + 1e-8))
                    noise_guided = args.rescale_phi * noise_rescaled + (1 - args.rescale_phi) * noise_guided

                xi = scheduler.step(noise_guided, t_step, xi)['prev_sample']

            gen_ecg = decode_latents(xi, decoder, device, args.decode_batch_size)

            for i in range(curr_bs):
                if global_idx + i < len(idx_list):
                    orig_gen_idx = idx_list[global_idx + i]

                    base_key = orig_gen_idx.split('_gen_')[0]
                    lookup_key = int(base_key) if base_key.isdigit() else base_key

                    original_sample = input_data[lookup_key]
                    out_label = original_sample['label'].copy()
                    out_label.pop('icd_embed', None)

                    output_data[orig_gen_idx] = {
                        'data': gen_ecg[i].float(),
                        'label': out_label,
                        **{k: v for k, v in original_sample.items() if k not in ['data', 'label']}
                    }

                    if visualized_count < args.num_visualize:
                        class_id = out_label.get('label', 'NA')
                        safe_key = str(orig_gen_idx).replace(os.sep, '_')
                        image_path = os.path.join(vis_dir, f"{visualized_count:03d}_{safe_key}_class{class_id}.png")
                        plot_ecg_12lead(gen_ecg[i], image_path, sample_rate=args.sample_rate)
                        visualization_manifest.append({
                            'generated_key': str(orig_gen_idx),
                            'source_key': str(lookup_key),
                            'class_id': _json_safe(class_id),
                            'image_path': image_path,
                            'label': _json_safe(out_label),
                        })
                        visualized_count += 1

            global_idx += curr_bs
            total_generated += curr_bs

    elapsed = time.time() - t_start

    print(f"\nSaving VAE-decoded ECG output dataset to: {args.output_path}")
    os.makedirs(os.path.dirname(args.output_path) or '.', exist_ok=True)
    torch.save(output_data, args.output_path)

    if visualization_manifest:
        manifest_path = os.path.join(vis_dir, 'visualization_manifest.json')
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(visualization_manifest, f, indent=2, ensure_ascii=False)
        print(f"  Visualization manifest saved to: {manifest_path}")

    print(f"\n" + "=" * 70)
    print(f"  Generation Summary (VAE-Decoded Waveform)")
    print("=" * 70)
    print(f"  Target Classes    : {sorted(target_classes)}")
    print(f"  Repeats per Sample: {num_repeats}x")
    print("-" * 70)
    for cid, count in class_stats.items():
        print(f"  Class {cid} generated: {count} samples")
    print("-" * 70)
    print(f"  Total generated   : {total_generated} samples")
    print(f"  Output data shape : ECG waveform, typically (1024, 12)")
    if visualization_manifest:
        print(f"  Visualizations    : {len(visualization_manifest)} images in {vis_dir}")
    print(f"  Generation time   : {elapsed:.1f}s ({total_generated / elapsed:.1f} samples/s)")
    print("=" * 70)

    return output_data


# CLI
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ECG Waveform Dataset Directed Generation")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)

    parser.add_argument("--target_classes", type=int, nargs='+', required=True,
                        help="Target class IDs to generate, e.g. --target_classes 1 2")
    parser.add_argument("--num_repeats", type=int, default=1,
                        help="Number of stochastic generations per matched sample")

    parser.add_argument("--checkpoint_path", type=str, required=True)

    parser.add_argument("--icd_graph_path", type=str, default="checkpoints/icd_graph_data.pt")
    parser.add_argument("--icd_embeddings_path", type=str, default="checkpoints/icd_hyperbolic_best.pth")
    parser.add_argument("--vae_path", type=str,
                        default="checkpoints/ecg_vae_ema.pth")
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--vis_dir", type=str, default=None,
                        help="Directory for generated ECG visualizations; defaults to output_path with _vis suffix")
    parser.add_argument("--num_visualize", type=int, default=4,
                        help="Number of generated ECGs to visualize; set 0 to disable")
    parser.add_argument("--sample_rate", type=float, default=102.4)

    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--rescale_phi", type=float, default=0.7)
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)

    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--icd_embed_dim", type=int, default=768)
    parser.add_argument("--text_embed_dim", type=int, default=768)
    parser.add_argument("--use_rope", action="store_true", default=False)

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_ema", action="store_true", default=True)
    parser.add_argument("--use_amp", action="store_true", default=False)

    args = parser.parse_args()
    output_dataset = generate_ecg_dataset(args)
