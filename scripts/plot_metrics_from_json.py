import argparse
import json
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def extract_precision_at_k(nsm: Dict, baseline_key: str = "count_only") -> Tuple[pd.DataFrame, List[int]]:
    model_pak = nsm.get("model", {}).get("precision_at_k", {})
    base_pak = nsm.get("baselines", {}).get(baseline_key, {}).get("precision_at_k", {})

    if not model_pak:
        raise ValueError("metrics.json missing next_step_metrics.model.precision_at_k")

    ks = sorted([int(k) for k in model_pak.keys()])
    rows = []
    for k in ks:
        k_str = str(k)
        m = model_pak.get(k_str, {})
        b = base_pak.get(k_str, {}) if base_pak else {}
        rows.append({
            "k": k,
            "model_mean": m.get("mean"),
            "model_ci_low": m.get("ci_lower"),
            "model_ci_high": m.get("ci_upper"),
            "baseline_mean": b.get("mean"),
            "baseline_ci_low": b.get("ci_lower"),
            "baseline_ci_high": b.get("ci_upper"),
        })
    df = pd.DataFrame(rows)
    return df, ks


def extract_point_metric(nsm: Dict, name: str, baseline_key: str = "count_only") -> pd.DataFrame:
    m = nsm.get("model", {}).get(name, {})
    b = nsm.get("baselines", {}).get(baseline_key, {}).get(name, {})
    if not m:
        raise ValueError(f"metrics.json missing next_step_metrics.model.{name}")
    return pd.DataFrame([
        {
            "metric": name.upper(),
            "variant": "model",
            "mean": m.get("mean"),
            "ci_low": m.get("ci_lower"),
            "ci_high": m.get("ci_upper"),
        },
        {
            "metric": name.upper(),
            "variant": "baseline",
            "mean": b.get("mean"),
            "ci_low": b.get("ci_lower"),
            "ci_high": b.get("ci_upper"),
        },
    ])


def extract_p_values(nsm: Dict, baseline_key: str = "count_only") -> Dict[str, float]:
    sig = nsm.get("significance", {}).get(baseline_key, {})
    out = {}
    pak = sig.get("precision_at_k", {})
    for k, obj in pak.items():
        out[f"precision_at_{k}"] = obj.get("p_value")
    for met in ("MRR", "accuracy"):
        val = sig.get(met, {}).get("p_value")
        if val is not None:
            out[met] = val
    return out


def plot_precision_at_k(df: pd.DataFrame, pvals: Dict[str, float], outdir: str) -> None:
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(8, 5))

    # Model line + CI
    ax.plot(df["k"], df["model_mean"], marker="o", label="Model", color="#1f77b4")
    if df[["model_ci_low", "model_ci_high"]].notna().all().all():
        ax.fill_between(df["k"], df["model_ci_low"], df["model_ci_high"], color="#1f77b4", alpha=0.15)

    # Baseline line + CI (if present)
    if df["baseline_mean"].notna().any():
        ax.plot(df["k"], df["baseline_mean"], marker="s", label="Baseline (count_only)", color="#ff7f0e")
        if df[["baseline_ci_low", "baseline_ci_high"]].notna().all().all():
            ax.fill_between(df["k"], df["baseline_ci_low"], df["baseline_ci_high"], color="#ff7f0e", alpha=0.15)

    # Annotate p-values
    for i, row in df.iterrows():
        k = int(row["k"])
        pv = pvals.get(f"precision_at_{k}")
        if pv is not None and pd.notna(row["model_mean"]):
            ax.text(k, row["model_mean"] + 0.005, f"p={pv:.3g}", ha="center", va="bottom", fontsize=8)

    ax.set_xlabel("K")
    ax.set_ylabel("Precision@K (Hit@K)")
    ax.set_title("Precision@K with 95% CI")
    ax.set_ylim(0, 1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "precision_at_k.png"), dpi=200)
    plt.close(fig)


def plot_point_metrics(df_acc: pd.DataFrame, df_mrr: pd.DataFrame, pvals: Dict[str, float], outdir: str) -> None:
    df = pd.concat([df_acc, df_mrr], ignore_index=True)
    # Bar plot with error bars
    fig, ax = plt.subplots(figsize=(7, 5))
    order = ["ACCURACY", "MRR"]
    variants = ["model", "baseline"]

    width = 0.35
    x = [0, 1]  # positions for metrics

    for j, var in enumerate(variants):
        means = [df[(df.metric == m) & (df.variant == var)]["mean"].values[0] for m in order]
        cil = [df[(df.metric == m) & (df.variant == var)]["ci_low"].values[0] for m in order]
        cih = [df[(df.metric == m) & (df.variant == var)]["ci_high"].values[0] for m in order]
        errs = [None, None]
        if all(pd.notna(c) for c in cil + cih):
            errs = [
                [[means[i] - cil[i] for i in range(2)], [cih[i] - means[i] for i in range(2)]]
            ]
        # Compute bar positions per variant
        xpos = [p + (j - 0.5) * width for p in x]
        ax.bar(xpos, means, width=width, label=var.title())

    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("Accuracy and MRR (Model vs Baseline)")
    ax.legend()

    # p-values annotation at center of metric groups
    for i, name in enumerate(order):
        key = name if name in ("MRR", "ACCURACY") else name.upper()
        pv = pvals.get(key if key in pvals else name)
        if pv is not None:
            ax.text(i, 0.02, f"p={pv:.3g}", ha="center", va="bottom", fontsize=9, color="#444")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "accuracy_mrr_comparison.png"), dpi=200)
    plt.close(fig)


def plot_path_probability_summary(ppm: Dict, outdir: str) -> None:
    if not ppm:
        return
    paths_eval = ppm.get("paths_evaluated")
    total_steps = ppm.get("total_steps")
    zero_prob_steps = ppm.get("zero_prob_steps")
    mean_logp = ppm.get("mean_avg_log_prob")
    mean_ppl = ppm.get("mean_perplexity")

    rows = [
        ("paths_evaluated", paths_eval),
        ("total_steps", total_steps),
        ("zero_prob_steps", zero_prob_steps),
        ("zero_prob_rate", (zero_prob_steps / total_steps) if total_steps else None),
        ("mean_avg_log_prob", mean_logp),
        ("mean_perplexity", mean_ppl),
    ]
    df = pd.DataFrame(rows, columns=["metric", "value"])
    df.to_csv(os.path.join(outdir, "path_probability_summary.csv"), index=False)

    # Simple bar for zero-prob rate and perplexity
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    zr = (zero_prob_steps / total_steps) if (zero_prob_steps is not None and total_steps) else None
    if zr is not None:
        ax[0].bar(["Zero-Prob Rate"], [zr], color="#d62728")
        ax[0].set_ylim(0, 1)
        ax[0].set_ylabel("Rate")
    ax[0].set_title("Zero-Probability Step Rate")

    if mean_ppl is not None:
        ax[1].bar(["Mean Perplexity"], [mean_ppl], color="#2ca02c")
        ax[1].set_ylabel("Perplexity")
    ax[1].set_title("Path Probability Perplexity")

    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "path_probability_metrics.png"), dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot key charts from outputs/metrics.json (Precision@K, MRR, Accuracy, Baseline)")
    parser.add_argument("--input", default=os.path.join("outputs", "metrics.json"), help="Path to metrics.json")
    parser.add_argument("--out", default=os.path.join("outputs", "evaluation_plots", "metrics_json_plots"), help="Directory to save plots")
    parser.add_argument("--baseline", default="count_only", help="Baseline key under next_step_metrics.baselines")
    args = parser.parse_args()

    ensure_outdir(args.out)

    with open(args.input, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    nsm = metrics.get("next_step_metrics", {})
    if not nsm:
        raise ValueError("metrics.json missing 'next_step_metrics'")

    # Precision@K + p-values
    df_pak, ks = extract_precision_at_k(nsm, baseline_key=args.baseline)
    df_pak.to_csv(os.path.join(args.out, "precision_at_k_from_json.csv"), index=False)

    pvals = extract_p_values(nsm, baseline_key=args.baseline)
    plot_precision_at_k(df_pak, pvals, args.out)

    # Point metrics: Accuracy + MRR
    df_acc = extract_point_metric(nsm, "accuracy", baseline_key=args.baseline)
    df_mrr = extract_point_metric(nsm, "MRR", baseline_key=args.baseline)
    plot_point_metrics(df_acc, df_mrr, pvals, args.out)

    # Path probability metrics summary (if present)
    ppm = metrics.get("path_probability_metrics", {})
    plot_path_probability_summary(ppm, args.out)

    print(f"Saved plots to: {args.out}")


if __name__ == "__main__":
    main()
