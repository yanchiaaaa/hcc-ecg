#!/usr/bin/env python3
"""
Generate HCC-ECG qualitative LBBB candidates from real MIMIC ECG conditions.

This script scans processed MIMIC-IV ECG files for clear left bundle branch
block conditions from HCC-ECG/data_icd, keeps the original ICD code list
(including I447), report text embedding, and tabular conditions, then generates
multiple stochastic 12-lead ECG candidates for paper figure selection.
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
for p in [str(_THIS_DIR), str(_PROJECT_ROOT)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from diffusers import DPMSolverMultistepScheduler
from icd_graph_loader import ICDGraphEmbeddingLoader
from optimized_collate_fn import OptimizedCollateFn

from generate_multi5_ecg import (
    create_model,
    decode_latents,
    forward_full_model,
    load_checkpoint,
    load_vae_decoder,
    normalize_embedding,
    normalize_gender_value,
    normalize_scalar,
    prepare_batch_data,
)


LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
PAPER_LAYOUT = [
    ("I", 0), ("aVR", 3), ("V1", 6), ("V4", 9),
    ("II", 1), ("aVL", 4), ("V2", 7), ("V5", 10),
    ("III", 2), ("aVF", 5), ("V3", 8), ("V6", 11),
]

LBBB_PATTERNS = [
    re.compile(r"\blbbb\b", re.I),
    re.compile(r"\bleft\s+bundle[-\s]+branch\s+block\b", re.I),
    re.compile(r"\bcomplete\s+left\s+bundle[-\s]+branch\s+block\b", re.I),
    re.compile(r"\bleft\s+bundle[-\s]+branch\s+block,\s*unspecified\b", re.I),
    re.compile(r"\blinksschenkelblock\b", re.I),
]
RBBB_PATTERNS = [
    re.compile(r"\brbbb\b", re.I),
    re.compile(r"\bright\s+bundle[-\s]+branch\s+block\b", re.I),
]
SINUS_PATTERNS = [
    re.compile(r"\bsinus\s+rhythm\b", re.I),
    re.compile(r"\bnormal\s+sinus\s+rhythm\b", re.I),
]
CONFUSING_PATTERNS = [
    (re.compile(r"\batrial\s+fibrillation\b|\batrial\s+flutter\b|\bafib\b|\bafl\b", re.I), 2.5),
    (re.compile(r"\bfrequent\s+pvcs?\b|\bventricular\s+premature\b|\bpvc", re.I), 1.2),
    (re.compile(r"\bmyocardial\s+infarction\b|\binfarct\b|\bmi\b", re.I), 1.0),
    (re.compile(r"\blvh\b|left\s+ventricular\s+hypertrophy", re.I), 0.7),
    (re.compile(r"pacemaker|paced\s+rhythm", re.I), 2.5),
    (re.compile(r"ventricular\s+tachycardia|\bvt\b", re.I), 2.0),
]


def json_safe(value):
    if torch.is_tensor(value):
        value = value.detach().cpu()
        if value.numel() == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, np.ndarray):
        if value.size == 1:
            return value.item()
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    try:
        json.dumps(value)
        return value
    except TypeError:
        return str(value)


def text_blob(label):
    parts = []
    for key in ["text", "icd_text", "icd", "diagnosis", "report"]:
        if key in label and label[key] is not None:
            parts.append(str(label[key]))
    return " | ".join(parts)




def parse_code_list(value):
    import ast
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text or text.lower() == "nan":
            return []
        try:
            parsed = ast.literal_eval(text)
            value = parsed
        except Exception:
            value = [text]
    elif not isinstance(value, (list, tuple, set)):
        value = [value]
    codes = []
    for v in value:
        code = str(v).replace(".", "").strip().upper()
        if code and code != "NAN":
            codes.append(code)
    return codes


def has_i447(label):
    return "I447" in parse_code_list(label.get("icd"))

def has_lbbb(label):
    blob = text_blob(label)
    return any(p.search(blob) for p in LBBB_PATTERNS)


def has_rbbb(label):
    blob = text_blob(label)
    return any(p.search(blob) for p in RBBB_PATTERNS)


def has_sinus(label):
    blob = text_blob(label)
    return any(p.search(blob) for p in SINUS_PATTERNS)


def confusion_penalty(label):
    blob = text_blob(label)
    total = 0.0
    matched = []
    for pattern, penalty in CONFUSING_PATTERNS:
        if pattern.search(blob):
            total += penalty
            matched.append(pattern.pattern)
    return total, matched


def normalize_waveform(x):
    x = torch.as_tensor(x, dtype=torch.float32)
    if x.dim() == 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.shape == (12, 1024):
        x = x.transpose(0, 1)
    if x.dim() != 2 or x.shape[1] != 12:
        raise ValueError(f"Expected waveform shape (1024, 12), got {tuple(x.shape)}")
    return x.contiguous()


def signal_quality(x):
    arr = normalize_waveform(x).numpy()
    finite = bool(np.isfinite(arr).all())
    if not finite:
        return {"finite": False, "pass": False}
    lead_std = arr.std(axis=0)
    robust_range = np.percentile(arr, 99) - np.percentile(arr, 1)
    max_abs = float(np.max(np.abs(arr)))
    missing_leads = int(np.sum(lead_std < 1e-5))
    pass_quality = (
        missing_leads == 0
        and 0.01 <= float(np.median(lead_std)) <= 20.0
        and 0.02 <= float(robust_range) <= 80.0
        and max_abs <= 100.0
    )
    return {
        "finite": True,
        "pass": bool(pass_quality),
        "median_lead_std": float(np.median(lead_std)),
        "robust_range": float(robust_range),
        "max_abs": max_abs,
        "missing_leads": missing_leads,
    }


def extract_condition(split, key, item):
    label = item["label"]
    hr = normalize_scalar(label.get("hr", label.get("heart rate", None)), default=float("nan"))
    age = normalize_scalar(label.get("age", None), default=float("nan"))
    gender = normalize_gender_value(label.get("gender", "F"))
    report = str(label.get("text", ""))
    icd_text = str(label.get("icd_text", ""))
    icd_codes = parse_code_list(label.get("icd"))
    clinical = str(label.get("icd_text", ""))
    subject_id = label.get("subject_id", "")
    study_id = label.get("study_id", "")
    q = signal_quality(item["data"])

    diagnosis_count = len(icd_codes) + max(report.count("|") + 1 if report else 0, 0)
    explicit_report = bool(any(p.search(report) for p in LBBB_PATTERNS))
    explicit_diag = bool(any(p.search(clinical + " " + icd_text) for p in LBBB_PATTERNS))
    hr_penalty = abs(hr - 75.0) / 50.0 if math.isfinite(hr) else 5.0
    score = 0.0
    score += 5.0 if explicit_report else 0.0
    score += 3.0 if explicit_diag else 0.0
    score += 1.0 if 55 <= hr <= 100 else -5.0
    score += 0.5 if q.get("pass") else -5.0
    conf_penalty, conf_matches = confusion_penalty(label)
    score += 2.0 if has_sinus(label) else 0.0
    score -= min(diagnosis_count, 12) * 0.08
    score -= hr_penalty
    score -= conf_penalty
    if has_rbbb(label):
        score -= 4.0

    cond_label = {
        "text": report,
        "icd_text": icd_text,
        "source_clinical_diagnoses": clinical,
        "icd": icd_codes,
        "icd_codes": icd_codes,
        "age": age,
        "gender": gender,
        "hr": hr,
        "text_embed": normalize_embedding(label["text_embed"], "text_embed"),
        "source_split": split,
        "source_key": str(key),
        "subject_id": str(subject_id),
        "study_id": str(study_id),
    }
    return {
        "split": split,
        "key": str(key),
        "subject_id": str(subject_id),
        "study_id": str(study_id),
        "age": age,
        "sex": gender,
        "heart_rate": hr,
        "icd_code": "I447",
        "icd_codes": icd_codes,
        "report": report,
        "diagnosis_summary": clinical,
        "icd_text": icd_text,
        "quality": q,
        "selection_score": float(score),
        "confusion_penalty": float(conf_penalty),
        "confusion_matches": conf_matches,
        "condition_label": cond_label,
        "real_waveform": normalize_waveform(item["data"]),
    }


def iter_processed_files(data_dir, include_train=True):
    data_dir = Path(data_dir)
    yield "val", data_dir / "mimic_vae_val_icd.pt"
    yield "test", data_dir / "mimic_vae_test_icd.pt"
    if include_train:
        for path in sorted(data_dir.glob("mimic_vae_train_icd_part*.pt")):
            yield "train", path


def scan_lbbb_conditions(args):
    candidates = []
    for split, path in iter_processed_files(args.data_dir, include_train=not args.skip_train):
        print(f"Scanning {split}: {path}")
        data = torch.load(path, map_location="cpu", weights_only=False)
        for key, item in tqdm(data.items(), desc=f"scan {path.name}"):
            label = item.get("label", {})
            if not has_i447(label):
                continue
            if not has_lbbb(label):
                continue
            if has_rbbb(label):
                continue
            try:
                cond = extract_condition(split, key, item)
            except Exception as exc:
                if args.verbose:
                    print(f"skip {split}/{key}: {exc}")
                continue
            if not (args.min_hr <= cond["heart_rate"] <= args.max_hr):
                continue
            if not cond["quality"].get("pass", False):
                continue
            candidates.append(cond)

    candidates.sort(key=lambda x: x["selection_score"], reverse=True)
    selected = candidates[: args.num_conditions]
    print(f"Found {len(candidates)} valid LBBB candidates; selected {len(selected)}.")
    return selected, candidates


def plot_ecg_grid(waveform, save_base, title=None, sample_rate=102.4):
    import matplotlib.pyplot as plt

    arr = normalize_waveform(waveform).numpy()
    t = np.arange(arr.shape[0]) / sample_rate
    y_lim = np.percentile(np.abs(arr), 99.3)
    y_lim = float(max(y_lim, 0.5))

    fig, axes = plt.subplots(3, 4, figsize=(12.0, 6.2), sharex=True, sharey=True)
    axes = axes.reshape(-1)
    for ax_idx, (lead_name, lead_idx) in enumerate(PAPER_LAYOUT):
        ax = axes[ax_idx]
        ax.plot(t, arr[:, lead_idx], color="#111827", linewidth=0.75)
        ax.text(0.015, 0.86, lead_name, transform=ax.transAxes, fontsize=9, fontweight="bold")
        ax.set_ylim(-y_lim, y_lim)
        ax.grid(True, color="#E5E7EB", linewidth=0.45)
        ax.tick_params(axis="both", labelsize=7, length=2)
    for ax in axes[-4:]:
        ax.set_xlabel("Time (s)", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=10, y=0.995)
    fig.tight_layout(pad=0.7)
    for ext in ["png", "pdf"]:
        fig.savefig(f"{save_base}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def estimate_hr(waveform, sample_rate=102.4):
    arr = normalize_waveform(waveform).numpy()
    lead = arr[:, 1]
    lead = lead - np.median(lead)
    sig = np.abs(lead)
    if not np.isfinite(sig).all() or sig.std() < 1e-8:
        return float("nan"), 0
    min_dist = int(sample_rate * 0.32)
    threshold = np.percentile(sig, 92)
    peaks = []
    last = -min_dist
    for i in range(1, len(sig) - 1):
        if sig[i] >= threshold and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1] and i - last >= min_dist:
            peaks.append(i)
            last = i
    duration = len(sig) / sample_rate
    hr = len(peaks) / duration * 60.0 if duration > 0 else float("nan")
    return float(hr), len(peaks)


def generated_sanity(waveform, target_hr, sample_rate=102.4):
    arr = normalize_waveform(waveform).numpy()
    finite = bool(np.isfinite(arr).all())
    max_abs = float(np.max(np.abs(arr))) if finite else float("nan")
    robust_range = float(np.percentile(arr, 99) - np.percentile(arr, 1)) if finite else float("nan")
    lead_std = arr.std(axis=0) if finite else np.zeros(12)
    missing_leads = int(np.sum(lead_std < 1e-5))
    corr = np.corrcoef(arr.T) if finite else np.full((12, 12), np.nan)
    off_diag = corr[np.triu_indices(12, k=1)]
    mean_abs_corr = float(np.nanmean(np.abs(off_diag)))
    est_hr, peak_count = estimate_hr(waveform, sample_rate=sample_rate)
    hr_abs_error = abs(est_hr - target_hr) if math.isfinite(est_hr) and math.isfinite(target_hr) else float("nan")
    ok = finite and missing_leads == 0 and max_abs < 100.0 and robust_range > 0.02 and mean_abs_corr < 0.985
    score = 0.0
    score += 2.0 if ok else -4.0
    if math.isfinite(hr_abs_error):
        score += max(0.0, 2.0 - hr_abs_error / 15.0)
    score += max(0.0, 1.0 - mean_abs_corr)
    score -= 1.0 if robust_range < 0.05 or max_abs > 50 else 0.0
    return {
        "finite": finite,
        "max_abs": max_abs,
        "robust_range": robust_range,
        "missing_leads": missing_leads,
        "estimated_hr": est_hr,
        "detected_peak_count": peak_count,
        "target_hr": float(target_hr),
        "hr_abs_error": float(hr_abs_error) if math.isfinite(hr_abs_error) else None,
        "mean_abs_lead_corr": mean_abs_corr,
        "auto_quality_score": float(score),
        "auto_pass": bool(ok),
    }


def load_hcc_components(args, device):
    icd_loader = ICDGraphEmbeddingLoader(
        graph_data_path=args.icd_graph_path,
        embeddings_path=args.icd_embeddings_path,
        special_tokens=["NORM"],
        logger=None,
    )
    icd_embeddings, code_to_id = icd_loader.load()
    icd_embeddings = icd_embeddings.to(device)
    collate = OptimizedCollateFn(
        icd_graph_embeddings=icd_embeddings,
        code_to_id=code_to_id,
        use_precomputed_text=True,
        enable_icd_cache=True,
        verbose=False,
    )
    model = create_model(
        {
            "in_channels": 4,
            "seq_length": 128,
            "hidden_size": args.hidden_size,
            "depth": args.depth,
            "num_heads": args.num_heads,
            "icd_embed_dim": args.icd_embed_dim,
            "text_embed_dim": args.text_embed_dim,
            "mlp_ratio": 4.0,
            "use_rope": args.use_rope,
        }
    )
    model = load_checkpoint(model, args.checkpoint_path, use_ema=True).to(device).eval()
    decoder = load_vae_decoder(args.vae_path, device)
    scheduler = DPMSolverMultistepScheduler(
        num_train_timesteps=1000,
        beta_start=0.00085,
        beta_end=0.012,
        algorithm_type="dpmsolver++",
        solver_order=2,
    )
    return collate, model, decoder, scheduler


def generate_candidates(args, selected):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    image_dir = out_dir / "all_candidates"
    real_dir = out_dir / "source_real_ecg"
    shortlist_dir = out_dir / "top_shortlist"
    for d in [image_dir, real_dir, shortlist_dir]:
        d.mkdir(parents=True, exist_ok=True)

    collate, model, decoder, scheduler = load_hcc_components(args, device)

    records = []
    generated_pt = {}
    task_labels = []
    task_meta = []
    for cond_idx, cond in enumerate(selected):
        title = f"LBBB source {cond_idx:02d}; HR {cond['heart_rate']:.0f} bpm; {cond['sex']}, age {cond['age']:.0f}"
        real_base = real_dir / f"source_{cond_idx:02d}_{cond['split']}_{cond['key']}"
        plot_ecg_grid(cond["real_waveform"], str(real_base), title=title, sample_rate=args.sample_rate)
        for rep in range(args.samples_per_condition):
            seed = args.seed + cond_idx * 1000 + rep
            task_labels.append(cond["condition_label"])
            task_meta.append((cond_idx, rep, seed, cond))

    dummy = torch.zeros(4, 128)
    batches = []
    for start in range(0, len(task_labels), args.batch_size):
        batch = [(dummy, task_labels[i]) for i in range(start, min(start + args.batch_size, len(task_labels)))]
        batches.append((start, batch))

    start_time = time.time()
    for start, batch in tqdm(batches, desc="HCC-ECG LBBB generation"):
        _, batch_labels = collate(batch)
        prepared = prepare_batch_data(batch_labels, device)
        curr_bs = len(batch)
        noises = []
        for i in range(curr_bs):
            seed = task_meta[start + i][2]
            g = torch.Generator(device="cpu").manual_seed(seed)
            noises.append(torch.randn(4, 128, generator=g))
        xi = torch.stack(noises, dim=0).to(device)

        scheduler.set_timesteps(args.num_sampling_steps)
        with torch.no_grad():
            for t_step in scheduler.timesteps:
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

        for i in range(curr_bs):
            cond_idx, rep, seed, cond = task_meta[start + i]
            key = f"cond{cond_idx:02d}_sample{rep:02d}_seed{seed}"
            waveform = ecg[i].float().cpu()
            sanity = generated_sanity(waveform, cond["heart_rate"], sample_rate=args.sample_rate)
            title = f"LBBB (I447); HR {cond['heart_rate']:.0f} bpm; {cond['sex']}, age {cond['age']:.0f}"
            save_base = image_dir / key
            plot_ecg_grid(waveform, str(save_base), title=title if not args.no_titles else None, sample_rate=args.sample_rate)
            generated_pt[key] = {"data": waveform, "condition": json_safe(cond["condition_label"]), "sanity": sanity}
            records.append(
                {
                    "generated_key": key,
                    "image_png": str(save_base) + ".png",
                    "image_pdf": str(save_base) + ".pdf",
                    "condition_index": cond_idx,
                    "sample_index": rep,
                    "seed": seed,
                    "source_split": cond["split"],
                    "source_key": cond["key"],
                    "subject_id": cond["subject_id"],
                    "study_id": cond["study_id"],
                    "age": cond["age"],
                    "sex": cond["sex"],
                    "heart_rate": cond["heart_rate"],
                    "icd_code": "I447",
                    "report": cond["report"],
                    "diagnosis_summary": cond["diagnosis_summary"],
                    "selection_score": cond["selection_score"],
                    "confusion_penalty": cond.get("confusion_penalty", 0.0),
                    **sanity,
                }
            )

    torch.save(generated_pt, out_dir / "generated_lbbb_waveforms.pt")
    with open(out_dir / "all_candidates_index.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    with open(out_dir / "all_candidates_index.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    ranked = sorted(records, key=lambda r: r["auto_quality_score"], reverse=True)
    for rank, row in enumerate(ranked[: args.shortlist_k], start=1):
        src_png = Path(row["image_png"])
        src_pdf = Path(row["image_pdf"])
        prefix = f"rank{rank:02d}_{src_png.stem}"
        import shutil
        shutil.copy2(src_png, shortlist_dir / f"{prefix}.png")
        shutil.copy2(src_pdf, shortlist_dir / f"{prefix}.pdf")
        row["shortlist_rank"] = rank
        row["auto_comment"] = make_auto_comment(row)

    with open(out_dir / "top_shortlist.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(ranked[0].keys())
        if "shortlist_rank" not in fields:
            fields = ["shortlist_rank"] + fields
        if "auto_comment" not in fields:
            fields.append("auto_comment")
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(ranked[: args.shortlist_k])
    with open(out_dir / "top_shortlist.json", "w", encoding="utf-8") as f:
        json.dump(ranked[: args.shortlist_k], f, indent=2, ensure_ascii=False)

    condition_rows = []
    for i, cond in enumerate(selected):
        condition_rows.append(
            {
                "condition_index": i,
                "source_split": cond["split"],
                "source_key": cond["key"],
                "subject_id": cond["subject_id"],
                "study_id": cond["study_id"],
                "age": cond["age"],
                "sex": cond["sex"],
                "heart_rate": cond["heart_rate"],
                "icd_code": "I447",
                "selection_score": cond["selection_score"],
                "confusion_penalty": cond.get("confusion_penalty", 0.0),
                "report": cond["report"],
                "diagnosis_summary": cond["diagnosis_summary"],
                "icd_text": cond["icd_text"],
                **{f"real_{k}": v for k, v in cond["quality"].items() if not isinstance(v, dict)},
            }
        )
    with open(out_dir / "selected_lbbb_conditions.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(condition_rows[0].keys()))
        writer.writeheader()
        writer.writerows(condition_rows)

    summary = {
        "output_dir": str(out_dir),
        "num_conditions": len(selected),
        "samples_per_condition": args.samples_per_condition,
        "total_generated": len(records),
        "checkpoint_path": args.checkpoint_path,
        "vae_path": args.vae_path,
        "scale": args.scale,
        "rescale_phi": args.rescale_phi,
        "num_sampling_steps": args.num_sampling_steps,
        "sample_rate": args.sample_rate,
        "elapsed_sec": time.time() - start_time,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return records


def make_auto_comment(row):
    comments = []
    if row.get("auto_pass"):
        comments.append("basic sanity checks passed")
    else:
        comments.append("requires manual review")
    if row.get("hr_abs_error") is not None:
        comments.append(f"estimated HR error {row['hr_abs_error']:.1f} bpm")
    comments.append(f"mean abs lead correlation {row['mean_abs_lead_corr']:.2f}")
    if row.get("max_abs", 0) > 50:
        comments.append("large amplitude range")
    return "; ".join(comments)


def main():
    parser = argparse.ArgumentParser(description="HCC-ECG LBBB qualitative candidate generation")
    parser.add_argument("--data_dir", type=str, default="data/processed_data_icd")
    parser.add_argument("--output_dir", type=str, default="outputs/qualitative_lbbb_hcc")
    parser.add_argument("--checkpoint_path", type=str, default="checkpoints/hcc_ecg_full_ema.pth")
    parser.add_argument("--vae_path", type=str, default="checkpoints/ecg_vae_ema.pth")
    parser.add_argument("--icd_graph_path", type=str, default="checkpoints/icd_graph_data.pt")
    parser.add_argument("--icd_embeddings_path", type=str, default="checkpoints/icd_hyperbolic_best.pth")
    parser.add_argument("--num_conditions", type=int, default=12)
    parser.add_argument("--samples_per_condition", type=int, default=8)
    parser.add_argument("--shortlist_k", type=int, default=20)
    parser.add_argument("--min_hr", type=float, default=55.0)
    parser.add_argument("--max_hr", type=float, default=100.0)
    parser.add_argument("--skip_train", action="store_true", help="scan only val/test for faster debugging")
    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--decode_batch_size", type=int, default=64)
    parser.add_argument("--sample_rate", type=float, default=102.4)
    parser.add_argument("--scale", type=float, default=1.5)
    parser.add_argument("--rescale_phi", type=float, default=0.7)
    parser.add_argument("--num_sampling_steps", type=int, default=35)
    parser.add_argument("--seed", type=int, default=44700)
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--use_amp", action="store_true", default=False)
    parser.add_argument("--no_titles", action="store_true", help="save generated plots without title text")
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--icd_embed_dim", type=int, default=768)
    parser.add_argument("--text_embed_dim", type=int, default=768)
    parser.add_argument("--use_rope", action="store_true", default=False)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selected, all_candidates = scan_lbbb_conditions(args)
    if not selected:
        raise RuntimeError("No valid LBBB conditions found.")

    with open(out_dir / "all_lbbb_condition_candidates.json", "w", encoding="utf-8") as f:
        compact = [{k: json_safe(v) for k, v in c.items() if k not in ["condition_label", "real_waveform"]} for c in all_candidates]
        json.dump(compact, f, indent=2, ensure_ascii=False)

    generate_candidates(args, selected)
    print(f"Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
