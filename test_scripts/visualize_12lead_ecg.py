

import argparse
import os
import sys
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch


_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from VAE.vae_model import VAE_Decoder


LEAD_NAMES = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


def parse_indices(indices_str: str) -> List[int]:
    if not indices_str:
        return []
    out = []
    for x in indices_str.split(","):
        x = x.strip()
        if x:
            out.append(int(x))
    return out


def decode_latents(latents: torch.Tensor, decoder: torch.nn.Module, device: torch.device, batch_size: int = 64) -> torch.Tensor:
    
    chunks = []
    with torch.no_grad():
        for i in range(0, latents.shape[0], batch_size):
            batch = latents[i : i + batch_size].to(device)
            ecg = decoder(batch)
            chunks.append(ecg.cpu())
    return torch.cat(chunks, dim=0)


def extract_12lead(sample: torch.Tensor) -> np.ndarray:
    
    x = sample.detach().cpu().numpy()

    if x.ndim != 2:
        raise ValueError(f"Unexpected ECG shape for one sample: {x.shape}")

    if x.shape[0] == 12:
        return x
    if x.shape[1] == 12:
        return x.T

    raise ValueError(f"Cannot infer 12-lead axis from shape {x.shape}")


def plot_one_sample(real_ecg_12: np.ndarray, gen_ecg_12: np.ndarray, sample_idx: int, save_path: str, title_prefix: str = "") -> None:
    
    L = min(real_ecg_12.shape[1], gen_ecg_12.shape[1])
    real_ecg_12 = real_ecg_12[:, :L]
    gen_ecg_12 = gen_ecg_12[:, :L]
    t = np.arange(L)

    fig, axes = plt.subplots(6, 2, figsize=(14, 12), sharex=True)
    axes = axes.flatten()

    for i in range(12):
        ax = axes[i]
        r = real_ecg_12[i]
        g = gen_ecg_12[i]

        ymin = min(r.min(), g.min())
        ymax = max(r.max(), g.max())
        margin = (ymax - ymin) * 0.1 + 1e-6

        ax.plot(t, r, color="black", linewidth=1.0, label="Real" if i == 0 else None)
        ax.plot(t, g, color="#1f77b4", linewidth=1.0, alpha=0.9, label="Generated" if i == 0 else None)
        ax.set_title(LEAD_NAMES[i], fontsize=10, pad=4)
        ax.set_ylim(ymin - margin, ymax + margin)
        ax.grid(True, alpha=0.25, linestyle="--")

    axes[0].legend(loc="upper right", fontsize=9)

    fig.suptitle(f"{title_prefix}Sample #{sample_idx}: 12-Lead ECG (Real vs Generated)", fontsize=14, fontweight="bold")
    fig.supxlabel("Time")
    fig.supylabel("Amplitude")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Visualize 12-lead ECG from generated latent dataset")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the generated .pt file")
    parser.add_argument(
        "--vae_path",
        type=str,
        default="checkpoints/ecg_vae_ema.pth",
        help="VAE checkpoint path",
    )
    parser.add_argument("--indices", type=str, default="", help="Sample indices, e.g. 0,5,10")
    parser.add_argument("--num_samples", type=int, default=3, help="Number of random samples when --indices is not set")
    parser.add_argument("--seed", type=int, default=2026, help="Random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="VAE decoding batch size")
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--output_dir", type=str, default="", help="Output directory; defaults to ecg_vis next to data_path")
    parser.add_argument("--save_pdf", action="store_true", help="Also save PDF files")

    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    print("=" * 72)
    print("Visualize Generated ECG (12-lead)")
    print("=" * 72)
    print(f"Data path : {args.data_path}")
    print(f"VAE path  : {args.vae_path}")
    print(f"Device    : {device}")

    data = torch.load(args.data_path, map_location="cpu", weights_only=False)
    if "real_latents" not in data or "gen_latents" not in data:
        raise KeyError("data file must contain keys: real_latents and gen_latents")

    real_latents = data["real_latents"]
    gen_latents = data["gen_latents"]

    if not torch.is_tensor(real_latents) or not torch.is_tensor(gen_latents):
        raise TypeError("real_latents / gen_latents must be torch.Tensor")

    n = min(real_latents.shape[0], gen_latents.shape[0])
    print(f"Total samples available: {n}")
    print(f"real_latents shape: {tuple(real_latents.shape)}")
    print(f"gen_latents  shape: {tuple(gen_latents.shape)}")

    idx_list = parse_indices(args.indices)
    if len(idx_list) == 0:
        rng = np.random.default_rng(args.seed)
        k = min(args.num_samples, n)
        idx_list = rng.choice(n, size=k, replace=False).tolist()
    else:
        idx_list = [i for i in idx_list if 0 <= i < n]

    if len(idx_list) == 0:
        raise ValueError("No valid sample indices selected")

    print(f"Selected indices: {idx_list}")

    vae_ckpt = torch.load(args.vae_path, map_location=device, weights_only=False)
    decoder = VAE_Decoder()
    decoder.load_state_dict(vae_ckpt["decoder"])
    decoder.eval().to(device)

    sel_real_latents = real_latents[idx_list]
    sel_gen_latents = gen_latents[idx_list]

    real_ecgs = decode_latents(sel_real_latents, decoder, device, batch_size=args.batch_size)
    gen_ecgs = decode_latents(sel_gen_latents, decoder, device, batch_size=args.batch_size)

    if args.output_dir.strip():
        out_dir = args.output_dir
    else:
        out_dir = os.path.join(os.path.dirname(args.data_path), "ecg_vis")
    os.makedirs(out_dir, exist_ok=True)

    for local_i, sample_idx in enumerate(idx_list):
        real_12 = extract_12lead(real_ecgs[local_i])
        gen_12 = extract_12lead(gen_ecgs[local_i])

        png_path = os.path.join(out_dir, f"sample_{sample_idx:05d}_12lead_real_vs_gen.png")
        plot_one_sample(real_12, gen_12, sample_idx=sample_idx, save_path=png_path, title_prefix="")
        print(f"Saved: {png_path}")

        if args.save_pdf:
            pdf_path = os.path.join(out_dir, f"sample_{sample_idx:05d}_12lead_real_vs_gen.pdf")
            plot_one_sample(real_12, gen_12, sample_idx=sample_idx, save_path=pdf_path, title_prefix="")
            print(f"Saved: {pdf_path}")

    print("=" * 72)
    print("Done")
    print("=" * 72)


if __name__ == "__main__":
    main()
