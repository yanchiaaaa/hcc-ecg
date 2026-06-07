

import sys
import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from torch.utils.data import DataLoader

sys.path.insert(0,'VAE')
from vae_model import VAE_Decoder

sys.path.insert(0,'dataset')
try:
    from mimic_processed_dataset import MIMIC_IV_ECG_Processed_Dataset
except ImportError:
    from dataset.mimic_processed_dataset import MIMIC_IV_ECG_Processed_Dataset

LEAD_NAMES = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

PDF_NAME_REAL = "Twelve_Lead_Correlation_real"
PDF_NAME_SYNTHETIC = "Twelve_Lead_Correlation_HCC-ECG"
PDF_NAME_DIFF = "Correlation_Difference_Real-HCC-ECG"


def decode_latents(latents: torch.Tensor, decoder: torch.nn.Module, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    
    ecgs = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch_size):
            batch = latents[i:i + batch_size].to(device)
            decoded = decoder(batch)
            ecgs.append(decoded.cpu())
    return torch.cat(ecgs, dim=0)


def compute_persample_correlation(ecg_tensor: torch.Tensor):
    
    if ecg_tensor.shape[2] == 12:
        ecg_tensor = ecg_tensor.transpose(1, 2)
        
    N, C, L = ecg_tensor.shape
    correlations = []
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in tqdm(range(N), leave=False, desc="Computing Corrs"):
            sig = ecg_tensor[i].numpy() # (12, 1024)
            c = np.corrcoef(sig)
            
            if np.isnan(c).any():
                np.nan_to_num(c, copy=False, nan=0.0)
                np.fill_diagonal(c, 1.0)
                
            correlations.append(c)
            
    correlations = np.stack(correlations, axis=0) # (N, 12, 12)
    
    mean_corr = np.mean(correlations, axis=0)
    std_corr = np.std(correlations, axis=0)
    
    return mean_corr, std_corr


def plot_and_save_heatmaps(
    corr_real,
    corr_gen,
    diff_matrix,
    save_path,
    model_name="Synthetic",
):
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    save_dir = os.path.dirname(save_path)
    real_pdf = os.path.join(save_dir, f"{PDF_NAME_REAL}.pdf")
    gen_pdf = os.path.join(save_dir, f"{PDF_NAME_SYNTHETIC}.pdf")
    diff_pdf = os.path.join(save_dir, f"{PDF_NAME_DIFF}.pdf")

    def _save_single_heatmap(matrix, out_path, vmin, vmax, center):
        fig, ax = plt.subplots(figsize=(6.4, 5.8), facecolor='white')
        sns.heatmap(
            matrix,
            ax=ax,
            cmap='coolwarm',
            vmin=vmin,
            vmax=vmax,
            center=center,
            xticklabels=LEAD_NAMES,
            yticklabels=LEAD_NAMES,
            square=True,
            cbar_kws={"shrink": 0.9},
            annot=False,
        )
        ax.tick_params(axis='x', rotation=45)
        fig.savefig(out_path, bbox_inches='tight', pad_inches=0.08, facecolor='white')
        plt.close(fig)

    _save_single_heatmap(
        corr_real,
        real_pdf,
        vmin=-1,
        vmax=1,
        center=0,
    )
    _save_single_heatmap(
        corr_gen,
        gen_pdf,
        vmin=-1,
        vmax=1,
        center=0,
    )
    _save_single_heatmap(
        diff_matrix,
        diff_pdf,
        vmin=-0.1,
        vmax=0.1,
        center=0,
    )

    print(f"\nHeatmap visualization saved to (PDF): {real_pdf}")
    print(f"Heatmap visualization saved to (PDF): {gen_pdf}")
    print(f"Heatmap visualization saved to (PDF): {diff_pdf}")


def evaluate_correlation(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    t_start = time.time()

    print("=" * 70)
    print(" 📈 Inter-lead Correlation (Raw ECG Baseline & Per-Sample Averaging)")
    print(f"Use VAE Decoding: {args.use_vae}")
    print("=" * 70)

    print(f"\nLoading generated data: {args.data_path}")
    data = torch.load(args.data_path, map_location='cpu')
    
    gen_data_raw = data.get('gen_latents', data.get('gen_data', data.get('gen_ecgs', None)))
    
    if gen_data_raw is None:
        print("Error: could not find 'gen_latents' or 'gen_data' in the .pt file")
        return
        
    model_type = data.get('model_type', 'unknown')
    N_target = gen_data_raw.shape[0]
    print(f"  Found {N_target} generated samples.")

    if args.use_vae:
        print("Loading VAE decoder...")
        decoder = VAE_Decoder()
        vae_ckpt = torch.load(args.vae_path, map_location=device, weights_only=False)
        decoder.load_state_dict(vae_ckpt['decoder'] if 'decoder' in vae_ckpt else vae_ckpt)
        decoder.eval().to(device)

        print("Decoding latents to 12-lead ECG waveforms...")
        gen_ecgs = decode_latents(gen_data_raw, decoder, device, args.batch_size)
        model_type += " (Decoded)"
    else:
        print("Skipping VAE decoder. Data is used directly as ECG waveforms...")
        if torch.is_tensor(gen_data_raw):
            gen_ecgs = gen_data_raw.cpu()
        else:
            gen_ecgs = torch.tensor(gen_data_raw).cpu()
        model_type += " (Raw)"

    print(f"  Generated ECGs shape : {list(gen_ecgs.shape)}")

    print(f"\n📥 Loading Raw Real ECG Data from {args.raw_data_path}...")
    raw_dataset = MIMIC_IV_ECG_Processed_Dataset(data_path=args.raw_data_path, usage='test')
    
    raw_loader = DataLoader(raw_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    
    real_ecgs = []
    loaded_samples = 0
    
    for batch_data, _ in raw_loader:
        if batch_data.dim() == 4 and batch_data.shape[1] == 1:
            batch_ecg = batch_data.squeeze(1)
        else:
            batch_ecg = batch_data
            
        real_ecgs.append(batch_ecg)
        loaded_samples += batch_ecg.shape[0]
        
        if loaded_samples >= N_target:
            break
            
    real_ecgs = torch.cat(real_ecgs, dim=0)[:N_target]
    print(f"  Raw real ECGs shape: {list(real_ecgs.shape)}")

    print("\n🧮 Computing Per-Sample Pearson Correlation Matrices...")
    print("  [Real Data (Raw)]")
    mean_corr_real, std_corr_real = compute_persample_correlation(real_ecgs)
    print("  [Synthetic Data]")
    mean_corr_gen, std_corr_gen = compute_persample_correlation(gen_ecgs)

    diff_matrix = mean_corr_real - mean_corr_gen
    abs_diff_matrix = np.abs(diff_matrix)
    np.fill_diagonal(abs_diff_matrix, 0)
    
    avg_corr_err = np.sum(abs_diff_matrix) / (12 * 11)
    max_corr_err = np.max(abs_diff_matrix)
    
    std_real_scalar = np.sum(std_corr_real * (1 - np.eye(12))) / (12 * 11)
    std_gen_scalar = np.sum(std_corr_gen * (1 - np.eye(12))) / (12 * 11)

    print("\n🎨 Plotting Heatmaps...")
    save_filename = os.path.basename(args.data_path).replace('.pt', '_raw_correlation.pdf')
    save_path = os.path.join(os.path.dirname(args.data_path), save_filename)
    plot_and_save_heatmaps(
        mean_corr_real,
        mean_corr_gen,
        diff_matrix,
        save_path,
        model_name=model_type,
    )

    elapsed = time.time() - t_start
    print("\n" + "=" * 70)
    print(" 📋 Inter-lead Correlation Results (Raw vs Gen)")
    print("=" * 70)
    print(f"  Model Type            : {model_type}")
    print(f"  Avg. Corr Error       : {avg_corr_err:.4f}  <-- lower is better")
    print(f"  Max  Corr Error       : {max_corr_err:.4f}  <-- lower is better")
    print(f"  --------------------------------------------------")
    print(f"  Raw Data Global Std   : +/-{std_real_scalar:.4f}")
    print(f"  Gen Data Global Std   : +/-{std_gen_scalar:.4f}")
    print(f"  Time taken            : {elapsed:.1f}s")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Inter-lead Correlation with Raw ECG Baseline")
    
    parser.add_argument("--data_path", type=str, required=True,
                        help="Path to the generated .pt file containing gen_latents/gen_data")
    parser.add_argument("--raw_data_path", type=str,
                        default="data/processed_data_icd",
                        help="Path to the original MIMIC dataset folder")
    parser.add_argument("--vae_path", type=str,
                        default="checkpoints/ecg_vae_ema.pth",
                        help="VAE Decoder checkpoint")
    #'checkpoints/ecg_vae_ema.pth'
    #'checkpoints/ecg_vae_ema.pth'
    
    parser.add_argument("--use_vae", action="store_true", default=True,
                        help="Use VAE decoding (default is True)")
    parser.add_argument("--no_vae", dest="use_vae", action="store_false",
                        help="Skip VAE decoding (for data already in ECG waveform format)")
                        
    parser.add_argument("--batch_size", type=int, default=128,
                        help="Batch size for loading and decoding")
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    evaluate_correlation(args)
