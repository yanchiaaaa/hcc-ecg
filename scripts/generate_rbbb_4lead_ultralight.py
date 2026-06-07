#!/usr/bin/env python3
"""
Generate RBBB-conditioned HCC-ECG qualitative 4-lead figures.

Only compact 4x1 figures are saved for paper inspection:
  condition_XX_<split>_<key>/
    condition.json
    real_4lead_4x1_ultralight.png/pdf
    generated_4lead_4x1_ultralight.png/pdf
"""

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in [PROJECT_ROOT, THIS_DIR]:
    text = str(path)
    if text in sys.path:
        sys.path.remove(text)
    sys.path.insert(0, text)

from generate_lbbb_hcc_qualitative import (  # noqa: E402
    generated_sanity,
    has_lbbb,
    has_rbbb,
    json_safe,
    load_hcc_components,
    normalize_waveform,
    parse_code_list,
    prepare_batch_data,
    signal_quality,
)
from generate_multi5_ecg import (  # noqa: E402
    decode_latents,
    forward_full_model,
    normalize_embedding,
    normalize_gender_value,
    normalize_scalar,
)


LEAD_TO_INDEX = {
    "I": 0,
    "II": 1,
    "III": 2,
    "aVR": 3,
    "aVL": 4,
    "aVF": 5,
    "V1": 6,
    "V2": 7,
    "V3": 8,
    "V4": 9,
    "V5": 10,
    "V6": 11,
}
DISPLAY_LEADS = ["I", "V1", "V2", "V6"]

ULTRALIGHT = {
    "face": "#fffdfd",
    "minor": "#f4dada",
    "major": "#eeaaaa",
    "minor_alpha": 0.42,
    "major_alpha": 0.46,
    "minor_lw": 0.20,
    "major_lw": 0.40,
    "wave": "#2b5db5",
    "wave_lw": 0.86,
    "label": "#111827",
}


def iter_files(data_dir: Path, include_train: bool):
    yield "val", data_dir / "mimic_vae_val_icd.pt"
    yield "test", data_dir / "mimic_vae_test_icd.pt"
    if include_train:
        for path in sorted(data_dir.glob("mimic_vae_train_icd_part*.pt")):
            yield "train", path


def get_text(label):
    return str(label.get("text", ""))


def get_diag(label):
    return str(label.get("icd_text", ""))


def has_rbbb_code(icd_codes):
    return any(code.startswith("I451") for code in icd_codes)


def report_group(text: str) -> str:
    lower = text.lower()
    if "atrial fibrillation" in lower or "atrial flutter" in lower:
        return "af_rbbb"
    if "pac" in lower or "premature atrial" in lower:
        return "pac_rbbb"
    if "prolonged pr" in lower or "av block" in lower:
        return "pr_rbbb"
    if "st" in lower or "t wave" in lower:
        return "stt_rbbb"
    if "sinus rhythm" in lower:
        return "sinus_rbbb"
    return "other_rbbb"


def extract_candidate(split, key, item):
    label = item["label"]
    icd_codes = parse_code_list(label.get("icd"))
    if not (has_rbbb_code(icd_codes) or has_rbbb(label)):
        return None
    if has_lbbb(label):
        return None

    hr = normalize_scalar(label.get("hr", label.get("heart rate", None)), default=float("nan"))
    if not math.isfinite(hr) or not (55.0 <= hr <= 100.0):
        return None
    q = signal_quality(item["data"])
    if not q.get("pass", False):
        return None

    age = normalize_scalar(label.get("age", None), default=float("nan"))
    sex = normalize_gender_value(label.get("gender", "F"))
    text = get_text(label)
    diag = get_diag(label)
    waveform = normalize_waveform(item["data"])

    rbbb_in_report = "right bundle branch block" in text.lower() or "rbbb" in text.lower()
    rbbb_in_diag = "right bundle" in diag.lower()
    score = 0.0
    score += 4.0 if rbbb_in_report else 2.0
    score += 1.0 if rbbb_in_diag or has_rbbb_code(icd_codes) else 0.0
    score += 1.0 if 65.0 <= hr <= 85.0 else 0.4
    score += max(0.0, 1.0 - abs(hr - 75.0) / 40.0)
    score += min(q.get("robust_range", 0.0), 3.0) * 0.05
    score -= max(0, len(icd_codes) - 4) * 0.08

    condition = {
        "source_split": split,
        "source_key": str(key),
        "subject_id": str(label.get("subject_id", "")),
        "study_id": str(label.get("study_id", "")),
        "age": age,
        "sex": sex,
        "heart_rate": hr,
        "icd_codes": icd_codes,
        "report": text,
        "diagnosis_text": diag,
        "report_group": report_group(text),
        "selection_score": float(score),
        "real_quality": q,
    }
    cond_label = {
        "text": text,
        "icd_text": diag,
        "source_clinical_diagnoses": diag,
        "icd": icd_codes,
        "icd_codes": icd_codes,
        "age": age,
        "gender": sex,
        "hr": hr,
        "text_embed": normalize_embedding(label["text_embed"], "text_embed"),
        "source_split": split,
        "source_key": str(key),
        "subject_id": str(label.get("subject_id", "")),
        "study_id": str(label.get("study_id", "")),
    }
    return {"condition": condition, "label": cond_label, "real_waveform": waveform}


def select_diverse(candidates, n):
    candidates = sorted(candidates, key=lambda c: c["condition"]["selection_score"], reverse=True)
    target_order = ["sinus_rbbb", "pr_rbbb", "pac_rbbb", "af_rbbb", "stt_rbbb", "other_rbbb"]
    quotas = {"sinus_rbbb": 4, "pr_rbbb": 2, "pac_rbbb": 1, "af_rbbb": 1, "stt_rbbb": 1, "other_rbbb": 2}
    selected = []
    used = set()
    for group in target_order:
        count = 0
        for cand in candidates:
            uid = (cand["condition"]["source_split"], cand["condition"]["source_key"])
            if uid in used or cand["condition"]["report_group"] != group:
                continue
            selected.append(cand)
            used.add(uid)
            count += 1
            if count >= quotas[group] or len(selected) >= n:
                break
        if len(selected) >= n:
            break
    for cand in candidates:
        uid = (cand["condition"]["source_split"], cand["condition"]["source_key"])
        if uid in used:
            continue
        selected.append(cand)
        used.add(uid)
        if len(selected) >= n:
            break
    return selected[:n], candidates


def scan_conditions(args):
    data_dir = Path(args.data_dir)
    candidates = []
    for split, path in iter_files(data_dir, include_train=not args.skip_train):
        print(f"Scanning {split}: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        for key, item in tqdm(data.items(), desc=f"scan {path.name}"):
            try:
                cand = extract_candidate(split, key, item)
            except Exception as exc:
                if args.verbose:
                    print(f"skip {split}/{key}: {exc}")
                cand = None
            if cand is not None:
                candidates.append(cand)
        del data
    selected, all_candidates = select_diverse(candidates, args.num_conditions)
    print(f"Found {len(candidates)} valid RBBB candidates; selected {len(selected)}.")
    group_counts = {}
    for cand in selected:
        group = cand["condition"]["report_group"]
        group_counts[group] = group_counts.get(group, 0) + 1
    print(f"Selected report groups: {group_counts}")
    return selected, all_candidates


def lead_ylim(real, generated, lead_idx):
    vals = np.concatenate([real[:, lead_idx], generated[:, lead_idx]])
    vals = vals[np.isfinite(vals)]
    if len(vals) == 0:
        return 1.0
    vals = vals - np.median(vals)
    lim = float(np.percentile(np.abs(vals), 99.2))
    return max(lim * 1.08, 0.5)


def draw_ecg_grid(ax, xlim, ylim):
    ax.set_facecolor(ULTRALIGHT["face"])
    xmin, xmax = xlim
    ymin, ymax = ylim
    minor_x = np.arange(np.floor(xmin / 0.04) * 0.04, xmax + 0.04, 0.04)
    major_x = np.arange(np.floor(xmin / 0.20) * 0.20, xmax + 0.20, 0.20)
    minor_y = np.arange(np.floor(ymin / 0.10) * 0.10, ymax + 0.10, 0.10)
    major_y = np.arange(np.floor(ymin / 0.50) * 0.50, ymax + 0.50, 0.50)
    for x in minor_x:
        ax.axvline(x, color=ULTRALIGHT["minor"], lw=ULTRALIGHT["minor_lw"], alpha=ULTRALIGHT["minor_alpha"], zorder=0)
    for y in minor_y:
        ax.axhline(y, color=ULTRALIGHT["minor"], lw=ULTRALIGHT["minor_lw"], alpha=ULTRALIGHT["minor_alpha"], zorder=0)
    for x in major_x:
        ax.axvline(x, color=ULTRALIGHT["major"], lw=ULTRALIGHT["major_lw"], alpha=ULTRALIGHT["major_alpha"], zorder=0)
    for y in major_y:
        ax.axhline(y, color=ULTRALIGHT["major"], lw=ULTRALIGHT["major_lw"], alpha=ULTRALIGHT["major_alpha"], zorder=0)


def plot_4x1_ultralight(waveform, save_base, sample_rate=102.4, y_lims=None):
    import matplotlib.pyplot as plt

    arr = normalize_waveform(waveform).numpy()
    t = np.arange(arr.shape[0]) / sample_rate
    fig, axes = plt.subplots(4, 1, figsize=(3.35, 4.35), dpi=450, sharex=True)
    fig.patch.set_facecolor("white")
    for ax, lead in zip(axes, DISPLAY_LEADS):
        idx = LEAD_TO_INDEX[lead]
        sig = arr[:, idx]
        y_lim = y_lims.get(lead) if y_lims else None
        if y_lim is None:
            y_lim = max(float(np.percentile(np.abs(sig - np.median(sig)), 99.2)), 0.5)
        xlim = (float(t[0]), float(t[-1]))
        ylim = (-y_lim, y_lim)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        draw_ecg_grid(ax, xlim, ylim)
        ax.plot(t, sig, color=ULTRALIGHT["wave"], linewidth=ULTRALIGHT["wave_lw"], zorder=3)
        ax.text(0.012, 0.76, lead, transform=ax.transAxes, fontsize=8.4, fontweight="bold", color=ULTRALIGHT["label"])
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.subplots_adjust(left=0.03, right=0.995, top=0.995, bottom=0.015, hspace=0.055)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{save_base}.{ext}", bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def generate_waveforms(args, selected):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    collate, model, decoder, scheduler = load_hcc_components(args, device)
    dummy = torch.zeros(4, 128)
    batch = [(dummy, cand["label"]) for cand in selected]
    _, batch_labels = collate(batch)
    prepared = prepare_batch_data(batch_labels, device)
    curr_bs = len(selected)
    noises = []
    for idx in range(curr_bs):
        g = torch.Generator(device="cpu").manual_seed(args.seed + idx)
        noises.append(torch.randn(4, 128, generator=g))
    xi = torch.stack(noises, dim=0).to(device)

    scheduler.set_timesteps(args.num_sampling_steps)
    with torch.no_grad():
        for t_step in tqdm(scheduler.timesteps, desc="HCC-ECG sampling"):
            xi_in = torch.cat([xi, xi], dim=0)
            t_in = t_step.expand(curr_bs * 2).to(device)
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                noise_pred = forward_full_model(model, xi_in, t_in, prepared, device)
            eps_uncond, eps_cond = noise_pred.chunk(2)
            noise_guided = eps_uncond + args.scale * (eps_cond - eps_uncond)
            if args.rescale_phi > 0.0:
                std_dims = [1, 2, 3] if eps_cond.dim() == 4 else [1, 2]
                std_cond = eps_cond.std(dim=std_dims, keepdim=True)
                std_guided = noise_guided.std(dim=std_dims, keepdim=True)
                noise_rescaled = noise_guided * (std_cond / (std_guided + 1e-8))
                noise_guided = args.rescale_phi * noise_rescaled + (1 - args.rescale_phi) * noise_guided
            xi = scheduler.step(noise_guided, t_step, xi)["prev_sample"]
        ecg = decode_latents(xi, decoder, device, args.decode_batch_size)
    return [ecg[i].float().cpu() for i in range(curr_bs)]


def save_outputs(args, selected, generated):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    index_rows = []
    generated_pt = {}
    for idx, (cand, gen_waveform) in enumerate(zip(selected, generated)):
        cond = cand["condition"]
        folder = out_dir / f"condition_{idx:02d}_{cond['source_split']}_{cond['source_key']}"
        folder.mkdir(parents=True, exist_ok=True)
        real = normalize_waveform(cand["real_waveform"])
        gen = normalize_waveform(gen_waveform)
        sanity = generated_sanity(gen, cond["heart_rate"], sample_rate=args.sample_rate)
        condition_json = {**cond, "generated_seed": args.seed + idx, "generated_sanity": sanity}
        (folder / "condition.json").write_text(json.dumps(json_safe(condition_json), indent=2, ensure_ascii=False), encoding="utf-8")

        real_arr = real.numpy()
        gen_arr = gen.numpy()
        y_lims = {lead: lead_ylim(real_arr, gen_arr, LEAD_TO_INDEX[lead]) for lead in DISPLAY_LEADS}
        plot_4x1_ultralight(real, str(folder / "real_4lead_4x1_ultralight"), sample_rate=args.sample_rate, y_lims=y_lims)
        plot_4x1_ultralight(gen, str(folder / "generated_4lead_4x1_ultralight"), sample_rate=args.sample_rate, y_lims=y_lims)

        key = f"condition_{idx:02d}"
        generated_pt[key] = {"data": gen, "condition": json_safe(condition_json), "real_data": real}
        row = {
            "condition_index": idx,
            "folder": str(folder),
            "source_split": cond["source_split"],
            "source_key": cond["source_key"],
            "subject_id": cond["subject_id"],
            "study_id": cond["study_id"],
            "age": cond["age"],
            "sex": cond["sex"],
            "heart_rate": cond["heart_rate"],
            "icd_codes": ";".join(cond["icd_codes"]),
            "report_group": cond["report_group"],
            "report": cond["report"],
            "diagnosis_text": cond["diagnosis_text"],
            "generated_seed": args.seed + idx,
            **sanity,
        }
        index_rows.append(row)

    torch.save(generated_pt, out_dir / "generated_and_real_4lead_waveforms.pt")
    with open(out_dir / "index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(index_rows[0].keys()))
        writer.writeheader()
        writer.writerows(index_rows)
    with open(out_dir / "index.json", "w", encoding="utf-8") as f:
        json.dump(index_rows, f, indent=2, ensure_ascii=False)
    return index_rows


def main():
    parser = argparse.ArgumentParser(description="Generate compact 4-lead RBBB HCC-ECG paper candidates")
    parser.add_argument("--data_dir", type=str, default="data/processed_data_icd")
    parser.add_argument("--output_dir", type=str, default="outputs/qualitative_rbbb_4lead_ultralight")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/hcc_ecg_full_ema.pth")
    parser.add_argument("--vae_path", type=str, default="checkpoints/ecg_vae_ema.pth")
    parser.add_argument("--icd_graph_path", type=str, default="checkpoints/icd_graph_data.pt")
    parser.add_argument("--icd_embeddings_path", type=str, default="checkpoints/icd_hyperbolic_best.pth")
    parser.add_argument("--num_conditions", type=int, default=10)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--sample_rate", type=float, default=102.4)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--rescale_phi", type=float, default=0.7)
    parser.add_argument("--num_sampling_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=99113)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--icd_embed_dim", type=int, default=768)
    parser.add_argument("--text_embed_dim", type=int, default=768)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    start = time.time()
    selected, all_candidates = scan_conditions(args)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    compact = [{k: json_safe(v) for k, v in c["condition"].items()} for c in all_candidates]
    (out_dir / "all_rbbb_candidates.json").write_text(json.dumps(compact, indent=2, ensure_ascii=False), encoding="utf-8")
    generated = generate_waveforms(args, selected)
    rows = save_outputs(args, selected, generated)
    summary = {
        "output_dir": str(out_dir),
        "num_conditions": len(rows),
        "display_leads": DISPLAY_LEADS,
        "checkpoint_path": args.checkpoint_path,
        "vae_path": args.vae_path,
        "scale": args.scale,
        "rescale_phi": args.rescale_phi,
        "num_sampling_steps": args.num_sampling_steps,
        "elapsed_sec": time.time() - start,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
