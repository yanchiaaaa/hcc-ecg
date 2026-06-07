"""
HCC-ECG unified generation pipeline with ICD and text fields.

Supported released modes:
  1. joint        - full model: ICD + Text + Tabular
  2. text_tabular - ablation: Text + Tabular
  3. icd_tabular  - ablation: ICD + Tabular
  4. uncond       - unconditional generation baseline
"""

import os
import sys
import time
import argparse
import torch
import torch.nn.functional as F
from tqdm import tqdm
from torch.utils.data import DataLoader

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _PROJECT_ROOT)

from dataset.mimic_iv_ecg_dataset import DictDataset
from diffusers import DPMSolverMultistepScheduler
from icd_graph_loader import ICDGraphEmbeddingLoader

from optimized_collate_fn import OptimizedCollateFn



def create_model(model_type, model_kwargs):
    
    if model_type in ('joint', 'uncond'):
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
    elif model_type in ('icd_only', 'text_only', 'tabular_only'):
        raise ValueError("This open-source package includes only joint, uncond, text_tabular, and icd_tabular model types.")
    elif model_type == 'text_tabular':
        from module.dit_ablation_text_tabular import DiT_TextTabular_ECG
        return DiT_TextTabular_ECG(
            in_channels=model_kwargs['in_channels'],
            seq_length=model_kwargs['seq_length'],
            hidden_size=model_kwargs['hidden_size'],
            depth=model_kwargs['depth'],
            num_heads=model_kwargs['num_heads'],
            text_embed_dim=model_kwargs['text_embed_dim'],
            mlp_ratio=model_kwargs.get('mlp_ratio', 4.0),
            dropout=0.0,
            use_rope=model_kwargs.get('use_rope', False),
        )
    elif model_type == 'icd_tabular':
        from module.dit_ablation_icd_tabular import DiT_ICDTabular_ECG
        return DiT_ICDTabular_ECG(
            in_channels=model_kwargs['in_channels'],
            seq_length=model_kwargs['seq_length'],
            hidden_size=model_kwargs['hidden_size'],
            depth=model_kwargs['depth'],
            num_heads=model_kwargs['num_heads'],
            icd_embed_dim=model_kwargs['icd_embed_dim'],
            mlp_ratio=model_kwargs.get('mlp_ratio', 4.0),
            dropout=0.0,
            use_rope=model_kwargs.get('use_rope', False),
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


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
    
    epoch_info = checkpoint.get('epoch', 'unknown')
    print(f"  Checkpoint epoch: {epoch_info}")
    return model



def forward_joint(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
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
    
    age_in = torch.cat([torch.full_like(age, 99999.0), age], dim=0)
    gender_in = torch.cat([torch.zeros_like(gender), gender], dim=0)
    hr_in = torch.cat([torch.full_like(hr, 99999.0), hr], dim=0)

    noise_pred = model(x, t, icd_in, text_in, age_in, gender_in, hr_in, icd_mask=mask_in)
    return noise_pred


def forward_uncond(model, x, t, batch_data, device):
    
    B = x.shape[0]
    
    null_icd = torch.zeros(B, batch_data['icd_embeds'].shape[1],
                           batch_data['icd_embeds'].shape[2], device=device)
    null_text = torch.zeros(B, batch_data['text_embeds'].shape[1],
                            batch_data['text_embeds'].shape[2], device=device)
    null_mask = torch.zeros(B, batch_data['icd_mask'].shape[1], device=device)
    zero_age = torch.full((B, 1), 99999.0, device=device)
    zero_gender = torch.zeros(B, 1, device=device)
    zero_hr = torch.full((B, 1), 99999.0, device=device)

    noise_pred = model(x, t, null_icd, null_text, zero_age, zero_gender, zero_hr, icd_mask=null_mask)
    return noise_pred


def forward_icd_only(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
    icd_embeds = batch_data['icd_embeds']
    icd_mask = batch_data['icd_mask']

    null_icd = torch.zeros_like(icd_embeds)
    null_mask = torch.zeros_like(icd_mask)

    icd_in = torch.cat([null_icd, icd_embeds], dim=0)
    mask_in = torch.cat([null_mask, icd_mask], dim=0)

    noise_pred = model(x, t, icd_in, icd_mask=mask_in)
    return noise_pred


def forward_text_only(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
    text_embeds = batch_data['text_embeds']

    null_text = torch.zeros_like(text_embeds)
    text_in = torch.cat([null_text, text_embeds], dim=0)

    noise_pred = model(x, t, text_in)
    return noise_pred


def forward_tabular_only(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
    age = batch_data['age']
    gender = batch_data['gender']
    hr = batch_data['hr']

    zero_age = torch.full_like(age, 99999.0)
    zero_gender = torch.zeros_like(gender)
    zero_hr = torch.full_like(hr, 99999.0)

    age_in = torch.cat([zero_age, age], dim=0)
    gender_in = torch.cat([zero_gender, gender], dim=0)
    hr_in = torch.cat([zero_hr, hr], dim=0)

    noise_pred = model(x, t, age_in, gender_in, hr_in)
    return noise_pred


def forward_text_tabular(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
    text_embeds = batch_data['text_embeds']
    age = batch_data['age']
    gender = batch_data['gender']
    hr = batch_data['hr']

    null_text = torch.zeros_like(text_embeds)
    zero_age = torch.full_like(age, 99999.0)
    zero_gender = torch.zeros_like(gender)
    zero_hr = torch.full_like(hr, 99999.0)

    text_in = torch.cat([null_text, text_embeds], dim=0)
    age_in = torch.cat([zero_age, age], dim=0)
    gender_in = torch.cat([zero_gender, gender], dim=0)
    hr_in = torch.cat([zero_hr, hr], dim=0)

    noise_pred = model(x, t, text_in, age_in, gender_in, hr_in)
    return noise_pred


def forward_icd_tabular(model, x, t, batch_data, device):
    
    B = x.shape[0] // 2
    icd_embeds = batch_data['icd_embeds']
    icd_mask = batch_data['icd_mask']
    age = batch_data['age']
    gender = batch_data['gender']
    hr = batch_data['hr']

    null_icd = torch.zeros_like(icd_embeds)
    null_mask = torch.zeros_like(icd_mask)
    zero_age = torch.full_like(age, 99999.0)
    zero_gender = torch.zeros_like(gender)
    zero_hr = torch.full_like(hr, 99999.0)

    icd_in = torch.cat([null_icd, icd_embeds], dim=0)
    mask_in = torch.cat([null_mask, icd_mask], dim=0)
    age_in = torch.cat([zero_age, age], dim=0)
    gender_in = torch.cat([zero_gender, gender], dim=0)
    hr_in = torch.cat([zero_hr, hr], dim=0)

    noise_pred = model(x, t, icd_in, age_in, gender_in, hr_in, icd_mask=mask_in)
    return noise_pred


FORWARD_FN = {
    'joint': forward_joint,
    'uncond': forward_uncond,
    'text_tabular': forward_text_tabular,
    'icd_tabular': forward_icd_tabular,
}


# def prepare_batch_data(y, device, age_mean=64.1502, age_std=17.6533,
#                        hr_mean=81.3950, hr_std=21.4822):

def prepare_batch_data(y, device, age_mean=62.6021, age_std=31.8827,
                       hr_mean=73.9253, hr_std=17.0864):
    
    data = {}

    # ICD
    data['icd_embeds'] = y['icdgraph_embed'].to(device)
    data['icd_mask'] = y['icdgraph_mask'].to(device)

    # Text
    text_embed = y['text_embed'].to(device)
    if text_embed.dim() == 2:
        text_embed = text_embed.unsqueeze(1)
    data['text_embeds'] = text_embed

    data['age'] = ((y['age'].to(device).float().view(-1, 1) - age_mean) / age_std)
    data['gender'] = y['gender'].to(device).float().view(-1, 1)
    data['hr'] = ((y['heart rate'].to(device).float().view(-1, 1) - hr_mean) / hr_std)

    data['icd_codes'] = y.get('icd', y.get('icd_codes', None))
    data['text_raw'] = y.get('text', None)
    data['age_raw'] = y['age'].cpu().float()
    data['gender_raw'] = y['gender'].cpu().float()
    data['hr_raw'] = y['heart rate'].cpu().float()

    return data



def generate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    model_type = args.model_type
    use_cfg = (model_type != 'uncond')

    print("=" * 70)
    print(f"  HCC-ECG Unified Generator v2 (with ICD & Text Fields)")
    print(f"  Model Type : {model_type}")
    print(f"  Checkpoint : {args.checkpoint_path}")
    print(f"  CFG Scale  : {args.scale}" if use_cfg else "  CFG Scale  : disabled (uncond)")
    print(f"  Rescale φ  : {args.rescale_phi}")
    print(f"  Steps      : {args.num_sampling_steps}")
    print(f"  Batch Size : {args.batch_size}")
    print("=" * 70)

    icd_loader = ICDGraphEmbeddingLoader(
        graph_data_path=args.icd_graph_path,
        embeddings_path=args.icd_embeddings_path,
        special_tokens=['NORM'],
        logger=None,
    )
    icd_embeddings, code_to_id = icd_loader.load()

    test_dataset = DictDataset(path=args.test_data_path)
    collate_fn_handler = OptimizedCollateFn(
        icd_graph_embeddings=icd_embeddings,
        code_to_id=code_to_id,
        use_precomputed_text=True,
        enable_icd_cache=True,
        verbose=False,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size,
        shuffle=False, collate_fn=collate_fn_handler,
    )

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
    model = create_model(model_type, model_kwargs)
    model = load_checkpoint(model, args.checkpoint_path, use_ema=args.use_ema)
    model = model.to(device).eval()

    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Model params: {param_count:.2f}M")

    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085, beta_end=0.012,
        algorithm_type="dpmsolver++", solver_order=2,
    )

    forward_fn = FORWARD_FN[model_type]

    target_nums = args.nums if args.nums else len(test_dataset)
    total_generated = 0
    
    results = {
        'gen_latents': [],
        'real_latents': [],
        'text_embeds': [],
        'icd_embeds': [],
        'icd_mask': [],         # ICD mask
        'cond_hr': [],
        'cond_age': [],
        'cond_gender': [],
        'icd_codes_list': [],
        'texts_raw': [],
        'texts': [],
        'hr_raw': [],
        'age_raw': [],
        'gender_raw': [],
    }

    t_start = time.time()

    with torch.no_grad():
        for x_real, y in tqdm(test_loader, desc=f"Generating [{model_type}]"):
            if total_generated >= target_nums:
                break

            curr_bs = x_real.shape[0]
            if total_generated + curr_bs > target_nums:
                curr_bs = target_nums - total_generated
                x_real = x_real[:curr_bs]
                for k in y:
                    if isinstance(y[k], torch.Tensor):
                        y[k] = y[k][:curr_bs]
                    elif isinstance(y[k], list):
                        y[k] = y[k][:curr_bs]

            batch_data = prepare_batch_data(y, device)

            xi = torch.randn(curr_bs, 4, 128, device=device)
            scheduler.set_timesteps(args.num_sampling_steps)

            for t_step in scheduler.timesteps:
                if use_cfg:
                    xi_in = torch.cat([xi, xi], dim=0)
                    t_in = t_step.expand(curr_bs * 2).to(device)
                else:
                    xi_in = xi
                    t_in = t_step.expand(curr_bs).to(device)

                with torch.cuda.amp.autocast(enabled=args.use_amp):
                    noise_pred = forward_fn(model, xi_in, t_in, batch_data, device)

                if use_cfg:
                    eps_uncond, eps_cond = noise_pred.chunk(2)
                    noise_guided = eps_uncond + args.scale * (eps_cond - eps_uncond)

                    if args.rescale_phi > 0.0:
                        std_cond = eps_cond.std(dim=list(range(1, eps_cond.ndim)), keepdim=True)
                        std_guided = noise_guided.std(dim=list(range(1, noise_guided.ndim)), keepdim=True)
                        noise_rescaled = noise_guided * (std_cond / (std_guided + 1e-8))
                        noise_guided = (args.rescale_phi * noise_rescaled + (1 - args.rescale_phi) * noise_guided)

                    noise_final = noise_guided
                else:
                    noise_final = noise_pred

                xi = scheduler.step(noise_final, t_step, xi)['prev_sample']

            results['gen_latents'].append(xi.cpu())
            results['real_latents'].append(x_real.cpu())
            
            results['text_embeds'].append(batch_data['text_embeds'].cpu())
            results['icd_embeds'].append(batch_data['icd_embeds'].cpu())
            results['icd_mask'].append(batch_data['icd_mask'].cpu())
            results['cond_hr'].append(batch_data['hr_raw'].view(-1).cpu())
            results['cond_age'].append(batch_data['age_raw'].view(-1).cpu())
            results['cond_gender'].append(batch_data['gender_raw'].view(-1).cpu())
            
            if batch_data.get('icd_codes') is not None:
                results['icd_codes_list'].extend(batch_data['icd_codes'])
            if batch_data.get('text_raw') is not None:
                text_batch = batch_data['text_raw']
                if isinstance(text_batch, list):
                    clean_raw = []
                    clean_norm = []
                    for t in text_batch:
                        if isinstance(t, (list, tuple)):
                            t = t[0] if len(t) > 0 else ''
                        t = str(t)
                        clean_raw.append(t)
                        clean_norm.append(t.strip().lower())
                    results['texts_raw'].extend(clean_raw)
                    results['texts'].extend(clean_norm)
                else:
                    t = str(text_batch)
                    results['texts_raw'].append(t)
                    results['texts'].append(t.strip().lower())
            results['hr_raw'].append(batch_data['hr_raw'].cpu())
            results['age_raw'].append(batch_data['age_raw'].cpu())
            results['gender_raw'].append(batch_data['gender_raw'].cpu())
            
            total_generated += curr_bs

    elapsed = time.time() - t_start
    print(f"\nGenerated {total_generated} samples in {elapsed:.1f}s "
          f"({total_generated / elapsed:.1f} samples/s)")

    print("\n📦 Preparing data for save...")
    
    max_icd_len = max(t.size(1) for t in results['icd_embeds'])
    embed_dim = results['icd_embeds'][0].size(2)

    padded_icd_embeds = []
    padded_icd_masks = []

    for embed, mask in zip(results['icd_embeds'], results['icd_mask']):
        curr_len = embed.size(1)
        if curr_len < max_icd_len:
            pad_size = (0, 0, 0, max_icd_len - curr_len)
            embed = F.pad(embed, pad_size, "constant", 0)
            mask_pad_size = (0, max_icd_len - curr_len)
            mask = F.pad(mask, mask_pad_size, "constant", 0)
        padded_icd_embeds.append(embed)
        padded_icd_masks.append(mask)

    final_data = {
        'gen_latents': torch.cat(results['gen_latents']),           # (N, 4, 128)
        'real_latents': torch.cat(results['real_latents']),         # (N, 4, 128)
        
        'text_embeds': torch.cat(results['text_embeds']),           # (N, 1, 768)
        'icd_embeds': torch.cat(padded_icd_embeds),                 # (N, max_len, 768)
        'icd_mask': torch.cat(padded_icd_masks),                    # (N, max_len)
        
        'cond_hr': torch.cat(results['cond_hr']),                   # (N,)
        'cond_age': torch.cat(results['cond_age']),                 # (N,)
        'cond_gender': torch.cat(results['cond_gender']),           # (N,)
        'hr_raw': torch.cat(results['hr_raw']),                     # (N,)
        'age_raw': torch.cat(results['age_raw']),                   # (N,)
        'gender_raw': torch.cat(results['gender_raw']),             # (N,)
        
        'model_type': model_type,
        'cfg_scale': args.scale if use_cfg else 0.0,
        'rescale_phi': args.rescale_phi,
        'num_sampling_steps': args.num_sampling_steps,
        'checkpoint_path': args.checkpoint_path,
        'use_ema': args.use_ema,
        'total_samples': total_generated,
        
        'icd_codes_list': results['icd_codes_list'],
        'texts_raw': results['texts_raw'],
        'texts': results['texts'],
    }

    if args.save_path is None:
        save_dir = args.output_dir
        os.makedirs(save_dir, exist_ok=True)
        ema_tag = "_ema" if args.use_ema else ""
        args.save_path = os.path.join(
            save_dir,
            f"gen_{model_type}{ema_tag}_s{args.scale}_phi{args.rescale_phi}_steps{args.num_sampling_steps}.pt"
        )

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(final_data, args.save_path)
    print(f"\nSaved to: {args.save_path}")
    print(f"  Total keys saved: {len(final_data)}")
    print(f"  gen_latents shape: {final_data['gen_latents'].shape}")
    print(f"  icd_codes_list length: {len(final_data['icd_codes_list'])}")
    print(f"  texts_raw length: {len(final_data['texts_raw'])}")
    
    print(f"\nSaved fields summary:")
    print(f"  {'Field':<20} {'Type':<15} {'Shape/Info':<45}")
    print(f"  {'-'*20} {'-'*15} {'-'*45}")
    for k, v in final_data.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k:<20} {'Tensor':<15} {str(v.shape):<45}")
        elif isinstance(v, list):
            print(f"  {k:<20} {'List':<15} {'len=' + str(len(v)):<45}")
        else:
            print(f"  {k:<20} {type(v).__name__:<15} {str(v)[:40]:<45}")


# CLI

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="HCC-ECG Unified Generation v2 (with ICD & Text Fields)"
    )

    parser.add_argument("--model_type", type=str, required=True,
                        choices=["joint", "text_tabular", "icd_tabular", "uncond"],
                        help="Model type or ablation mode")
    parser.add_argument("--checkpoint_path", type=str,
                        default="checkpoints/hcc_ecg_full_ema.pth",
                        help="Model checkpoint path")

    parser.add_argument("--test_data_path", type=str,
                        default="data/example_test_latents.pt")
    parser.add_argument("--icd_graph_path", type=str,
                        default="checkpoints/icd_graph_data.pt")
    parser.add_argument("--icd_embeddings_path", type=str,
                        default="checkpoints/icd_hyperbolic_best.pth")
    parser.add_argument("--output_dir", type=str,
                        default="outputs/generation_results_v2")

    parser.add_argument("--scale", type=float, default=1.5,
                        help="CFG guidance scale")
    parser.add_argument("--rescale_phi", type=float, default=0.7,
                        help="Guidance rescale factor")
    parser.add_argument("--num_sampling_steps", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--nums", type=int, default=None,
                        help="Number of samples to generate; None uses the full test set")
    parser.add_argument("--save_path", type=str, default=None)

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
    generate(args)
