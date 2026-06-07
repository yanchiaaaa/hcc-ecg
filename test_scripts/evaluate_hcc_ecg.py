

import sys
import os
import argparse
import json
import time
import warnings

import numpy as np
import torch
from scipy.linalg import sqrtm
from scipy.stats import pearsonr
from tqdm import tqdm

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)  
sys.path.insert(0, _PROJECT_ROOT)

_FOUNDER_DIR = 'external/ecgfounder'
if _FOUNDER_DIR not in sys.path:
    sys.path.insert(0, _FOUNDER_DIR)

from VAE.vae_model import VAE_Decoder
from finetune_model import ft_12lead_ECGFounder
from clip.clip_model import CLIP

warnings.filterwarnings('ignore', category=RuntimeWarning)



def z_score_normalize(signal: torch.Tensor) -> torch.Tensor:
    if signal.dim() == 3:
        B = signal.shape[0]
        flat = signal.contiguous().view(B, -1)
        mean = flat.mean(dim=1, keepdim=True)
        std = flat.std(dim=1, keepdim=True)
        return ((flat - mean) / (std + 1e-8)).view(signal.shape)
    else:
        return (signal - signal.mean()) / (signal.std() + 1e-8)


def extract_features_batch(ecg_batch: torch.Tensor, model: torch.nn.Module, device: torch.device, batch_size: int = 32) -> torch.Tensor:
    if ecg_batch.shape[1] > ecg_batch.shape[2]:
        ecg_batch = ecg_batch.transpose(1, 2)
    all_features = []
    with torch.no_grad():
        for i in range(0, ecg_batch.shape[0], batch_size):
            batch = ecg_batch[i:i + batch_size].to(device)
            batch = z_score_normalize(batch)
            _, features = model(batch)
            all_features.append(features.cpu())
    return torch.cat(all_features, dim=0)


def compute_fid(M1: np.ndarray, M2: np.ndarray, eps: float = 1e-6) -> float:
    if np.isnan(M1).any() or np.isnan(M2).any(): return float('nan')
    mu1, mu2 = M1.mean(axis=0), M2.mean(axis=0)
    sigma1 = np.cov(M1, rowvar=False) + np.eye(M1.shape[1]) * eps
    sigma2 = np.cov(M2, rowvar=False) + np.eye(M2.shape[1]) * eps
    diff_sq = np.sum((mu1 - mu2) ** 2)
    covmean, _ = sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean): covmean = covmean.real
    return float(diff_sq + np.trace(sigma1 + sigma2 - 2.0 * covmean))


def estimate_heart_rate_batch(ecg_batch: torch.Tensor, fs: float = 102.4, lead_idx: int = 1) -> np.ndarray:
    N, L, C = ecg_batch.shape
    hr_list = np.full(N, np.nan)
    min_lag = int(fs * 0.3)
    max_lag = min(int(fs * 2.0), L - 1)
    for i in range(N):
        sig = ecg_batch[i, :, lead_idx].numpy().astype(np.float64)
        sig = sig - sig.mean()
        full_corr = np.correlate(sig, sig, mode='full')
        autocorr = full_corr[L - 1:]
        if autocorr[0] <= 0: continue
        autocorr = autocorr / autocorr[0]
        search_region = autocorr[min_lag:max_lag + 1]
        if len(search_region) == 0: continue
        peak_idx = np.argmax(search_region) + min_lag
        if autocorr[peak_idx] < 0.1: continue
        rr_sec = peak_idx / fs
        hr_list[i] = 60.0 / rr_sec
    return hr_list


def compute_hr_metrics(gen_hr: np.ndarray, cond_hr: np.ndarray, real_hr: np.ndarray, N: int) -> dict:
    results = {}
    valid_cond = (cond_hr > 20) & (cond_hr < 250)
    valid_gen, valid_real = ~np.isnan(gen_hr), ~np.isnan(real_hr)
    
    mask_gen = valid_cond & valid_gen
    n_valid = mask_gen.sum()
    results['HR_valid_ratio'] = float(n_valid / N)
    
    if n_valid >= 2:
        errors = gen_hr[mask_gen] - cond_hr[mask_gen]
        results['HR_MAE'] = float(np.abs(errors).mean())
        results['HR_RMSE'] = float(np.sqrt((errors ** 2).mean()))
        r, p_val = pearsonr(cond_hr[mask_gen], gen_hr[mask_gen])
        results['HR_Corr'] = float(r)
    else:
        results['HR_MAE'] = results['HR_RMSE'] = results['HR_Corr'] = float('nan')

    mask_real = valid_cond & valid_real
    if mask_real.sum() >= 2:
        real_errors = real_hr[mask_real] - cond_hr[mask_real]
        results['HR_MAE_real_ref'] = float(np.abs(real_errors).mean())
    else:
        results['HR_MAE_real_ref'] = float('nan')

    if not np.isnan(results.get('HR_MAE', np.nan)) and not np.isnan(results.get('HR_MAE_real_ref', np.nan)):
        results['HR_MAE_delta'] = results['HR_MAE'] - results['HR_MAE_real_ref']
    else:
        results['HR_MAE_delta'] = float('nan')
    return results, n_valid


def compute_mfd(features: np.ndarray, device: torch.device = None, batch_size: int = 1000) -> float:
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    feats = torch.from_numpy(features).float().to(device)
    N = feats.shape[0]
    norms_sq = (feats * feats).sum(dim=1)
    total_dist, total_pairs = 0.0, 0
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        ni = end_i - i
        norms_i = norms_sq[i:end_i]
        for j in range(i, N, batch_size):
            end_j = min(j + batch_size, N)
            nj = end_j - j
            norms_j = norms_sq[j:end_j]
            dot_product = feats[i:end_i] @ feats[j:end_j].T
            dist_sq = torch.clamp(norms_i.unsqueeze(1) + norms_j.unsqueeze(0) - 2 * dot_product, min=0.0)
            dists = torch.sqrt(dist_sq)
            if i == j:
                mask = torch.triu(torch.ones(ni, nj, device=device, dtype=torch.bool), diagonal=1)
                total_dist += dists[mask].sum().item()
                total_pairs += mask.sum().item()
            else:
                total_dist += dists.sum().item()
                total_pairs += ni * nj
    return float(total_dist / total_pairs) if total_pairs > 0 else 0.0


def compute_vendi_score(features: np.ndarray, max_samples: int = 5000, device: torch.device = None) -> float:
    if device is None: device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    N = features.shape[0]
    if N > max_samples:
        indices = np.random.choice(N, max_samples, replace=False)
        features = features[indices]
    feats = torch.from_numpy(features).float().to(device)
    norms = feats.norm(dim=1, keepdim=True).clamp(min=1e-8)
    feats_norm = feats / norms
    K = feats_norm @ feats_norm.T
    eigenvalues = torch.clamp(torch.linalg.eigvalsh(K), min=0.0)
    total = eigenvalues.sum()
    if total <= 0: return 0.0
    eigenvalues = eigenvalues / total
    nonzero = eigenvalues[eigenvalues > 1e-12]
    return float(torch.exp(-(nonzero * torch.log(nonzero)).sum()).item())


def decode_latents(latents: torch.Tensor, decoder: torch.nn.Module, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    ecgs = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch_size):
            batch = latents[i:i + batch_size].to(device)
            decoded = decoder(batch)
            ecgs.append(decoded.cpu())
    return torch.cat(ecgs, dim=0)


def compute_clip_score_batch(ecgs: torch.Tensor, text_embeddings: torch.Tensor, clip_model: torch.nn.Module, device: torch.device, batch_size: int = 64) -> float:
    
    if text_embeddings.dim() == 3:
        text_embeddings = text_embeddings.squeeze(1)
        
    N = ecgs.shape[0]
    total_clip_score = 0.0
    
    clip_model.eval()
    with torch.no_grad():
        for i in range(0, N, batch_size):
            end_idx = min(i + batch_size, N)
            batch_ecgs = ecgs[i:end_idx].to(device)
            batch_text = text_embeddings[i:end_idx].to(device, dtype=torch.float)
            
            signal_embedding = clip_model.encode_signal(batch_ecgs)
            signal_features = clip_model.ecg_projector(signal_embedding)
            text_features = clip_model.text_projector(batch_text)
            
            signal_features = signal_features / signal_features.norm(dim=-1, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            sample_clip_scores = (signal_features * text_features).sum(dim=-1)
            total_clip_score += sample_clip_scores.sum().item()
            
    return float(total_clip_score / N)



def fast_evaluate(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("=" * 70)
    print("  HCC-ECG Comprehensive Evaluation Pipeline")
    print("=" * 70)

    print(f"\nLoading: {args.data_path}")
    data = torch.load(args.data_path, map_location='cpu', weights_only=False)

    gen_latents = data['gen_latents']
    real_latents = data['real_latents']
    text_embeds = data.get('text_embeds', None)
    
    if text_embeds is not None and text_embeds.shape[-1] == 1536:
        print("  Detected 1536-dim text_embeds, slicing to 768 dims for CLIP.")
        text_embeds = text_embeds[..., :768]

    model_type = data.get('model_type', 'unknown')
    cfg_scale = data.get('cfg_scale', 'N/A')
    rescale_phi = data.get('rescale_phi', 'N/A')
    num_steps = data.get('num_sampling_steps', 'N/A')
    N = gen_latents.shape[0]

    cond_hr = data.get('cond_hr', None)

    print("\n📦 Loading models...")
    decoder = VAE_Decoder()
    decoder.load_state_dict(torch.load(args.vae_path, map_location=device, weights_only=False)['decoder'])
    decoder.eval().to(device)
    print("  VAE decoder")

    ecgfounder = ft_12lead_ECGFounder(device=device, pth=args.ecgfounder_path, n_classes=1, linear_prob=False)
    ecgfounder.return_features = True
    ecgfounder.eval()
    print("  ECGFounder")

    if text_embeds is not None:
        clip_model = CLIP(embed_dim=64)
        clip_model.load_state_dict(torch.load(args.clip_path, map_location=device))
        clip_model.eval().to(device)
        print("  CLIP model")

    is_vae_latent = (gen_latents.shape[1] == 4)
    if is_vae_latent:
        print("\n[VAE mode] Decoding latents to waveforms...")
        gen_ecgs = decode_latents(gen_latents, decoder, device, args.decode_batch_size)
        real_ecgs = decode_latents(real_latents, decoder, device, args.decode_batch_size)
    else:
        print("\n[No-VAE mode] Formatting waveforms...")
        gen_ecgs = gen_latents.transpose(1, 2).cpu()
        real_ecgs = real_latents.transpose(1, 2).cpu()

    print("\n🧠 Extracting ECGFounder features...")
    M_gen = extract_features_batch(gen_ecgs, ecgfounder, device, args.feature_batch_size)
    M_real = extract_features_batch(real_ecgs, ecgfounder, device, args.feature_batch_size)

    results = {
        'model_type': model_type, 'cfg_scale': cfg_scale, 'rescale_phi': rescale_phi,
        'num_sampling_steps': num_steps, 'num_samples': N
    }
    M_gen_np, M_real_np = M_gen.numpy(), M_real.numpy()

    print("\n" + "=" * 70)
    print("Computing evaluation metrics")
    print("=" * 70)

    print("\n[1/5] FID/rFID...")
    results['FID'] = compute_fid(M_real_np, M_gen_np)
    print(f"  FID = {results['FID']:.4f}")
    
    half = N // 2
    if half >= 50:
        perm_gen, perm_real = np.random.permutation(N)[:half], np.random.permutation(N)
        fid_num = compute_fid(M_gen_np[perm_gen], M_real_np[perm_real[:half]])
        fid_base = compute_fid(M_real_np[perm_real[:half]], M_real_np[perm_real[half:2 * half]])
        if fid_base > 0 and not np.isnan(fid_base) and not np.isnan(fid_num):
            results['rFID'] = fid_num / fid_base
            print(f"  rFID = {results['rFID']:.4f}")

    print("\n[2/5] HR Conditional Consistency...")
    if cond_hr is not None:
        gen_hr = estimate_heart_rate_batch(gen_ecgs, fs=102.4)
        real_hr = estimate_heart_rate_batch(real_ecgs, fs=102.4)
        hr_results, _ = compute_hr_metrics(gen_hr, cond_hr.numpy(), real_hr, N)
        results.update(hr_results)
        print(f"  HR-MAE  = {results.get('HR_MAE', float('nan')):.2f} bpm")
        print(f"  HR-Corr = {results.get('HR_Corr', float('nan')):.4f}")

    print("\n[3/5] Diversity - MFD...")
    mfd_gen = compute_mfd(M_gen_np, device=device)
    mfd_real = compute_mfd(M_real_np, device=device)
    results['MFD_ratio'] = mfd_gen / mfd_real if mfd_real > 0 else float('nan')
    print(f"  MFD_ratio = {results['MFD_ratio']:.4f}")

    print("\n[4/5] Diversity - Vendi Score...")
    vs_gen = compute_vendi_score(M_gen_np, max_samples=args.vendi_max_samples, device=device)
    vs_real = compute_vendi_score(M_real_np, max_samples=args.vendi_max_samples, device=device)
    results['Vendi_ratio'] = vs_gen / vs_real if vs_real > 0 else float('nan')
    print(f"  Vendi_ratio = {results['Vendi_ratio']:.4f}")

    print("\n[5/5] Text Alignment (CLIP)...")
    if text_embeds is not None:
        clip_score_gen = compute_clip_score_batch(gen_ecgs, text_embeds, clip_model, device, args.decode_batch_size)
        clip_score_real = compute_clip_score_batch(real_ecgs, text_embeds, clip_model, device, args.decode_batch_size)
        results['CLIP_Score'] = clip_score_gen
        results['CLIP_Score_real'] = clip_score_real
        results['rCLIP'] = clip_score_gen / clip_score_real if clip_score_real > 0 else float('nan')
        print(f"  CLIP (Gen)  = {results['CLIP_Score']:.4f}")
        print(f"  CLIP (Real) = {results['CLIP_Score_real']:.4f}")
        print(f"  rCLIP       = {results['rCLIP']:.4f}")
    else:
        print("  text_embeds missing; skipped CLIP evaluation.")

    elapsed = time.time() - t_start

    # Summary
    print("\n" + "=" * 70)
    print("  📋 Final Results Summary")
    print("=" * 70)
    print(f"  FID        : {results.get('FID', float('nan')):.4f}")
    print(f"  rFID       : {results.get('rFID', float('nan')):.4f}")
    print(f"  HR-MAE     : {results.get('HR_MAE', float('nan')):.2f} bpm")
    print(f"  HR-Corr    : {results.get('HR_Corr', float('nan')):.4f}")
    print(f"  MFD ratio  : {results.get('MFD_ratio', float('nan')):.4f}")
    print(f"  Vendi ratio: {results.get('Vendi_ratio', float('nan')):.4f}")
    if text_embeds is not None:
        print(f"  CLIP Score : {results.get('CLIP_Score', float('nan')):.4f}")
        print(f"  rCLIP      : {results.get('rCLIP', float('nan')):.4f}")
    print(f"  ─────────────────────────────────────────────────")
    print(f"  Time       : {elapsed:.1f}s")
    
    if args.save_results:
        result_path = args.data_path.replace('.pt', '_eval_results.json')
        json_results = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else (None if isinstance(v, float) and np.isnan(v) else v)) for k, v in results.items()}
        json_results['data_path'] = args.data_path
        json_results['eval_time_sec'] = round(elapsed, 1)

        with open(result_path, 'w') as f:
            json.dump(json_results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {result_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HCC-ECG Full Evaluation")
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--vae_path", type=str, default="checkpoints/ecg_vae_ema.pth")
    parser.add_argument("--ecgfounder_path", type=str, default="checkpoints/12_lead_ECGFounder.pth")
    parser.add_argument("--clip_path", type=str, default="checkpoints/clip_best.pth")
    
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--feature_batch_size", type=int, default=32)
    parser.add_argument("--vendi_max_samples", type=int, default=5000)

    parser.add_argument("--save_results", action="store_true", default=True)
    parser.add_argument("--no_save", dest="save_results", action="store_false")

    args = parser.parse_args()
    results = fast_evaluate(args)
