"""Stage 5, 6 & 7: Ablation Study, Noise Robustness Stress Test & Automated Plotting.

Dataset-agnostic: works seamlessly on default California Housing OR any custom CSV
passed via `--csv` (and optional `--target`), matching `kd_california.py`.

Usage:
    python kd_california.py --csv custom.csv --target price
    python ablation_and_robustness.py --csv custom.csv --target price
"""
import argparse
import copy
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F


from data_prep import load_dataset, prepare
from knowledge_distillation import (
    ACTIVATIONS,
    Expert,
    FrozenEnsemble,
    Student,
    banner,
    cache_teacher,
    format_table,
    gaussian_kl,
    inverse_target,
    regression_metrics,
    relational_loss,
    set_seed,
    teacher_gate,
    train_expert,
    train_student,
)
# Exploratory Data Analysis (EDA) and plotting libraries
def generate_eda_plots(out_dir, X, y, feature_names):
    """Generate 3 publication-grade 300-DPI EDA figures for the dataset."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })

    # ------------------------------------------------------------------
    # EDA PLOT 1: Target Distribution (Capping) & Heavy-Tail Outlier Boxplot
    # ------------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.0))

    ax1.hist(y, bins=45, color="#2b5c8f", edgecolor="white", alpha=0.9)
    ax1.axvline(5.0, color="#c0392b", linestyle="--", lw=1.6, label="Ceiling Cap ($500k / 5.0)")
    ax1.axvline(np.median(y), color="#16a085", linestyle="-.", lw=1.6, label=f"Median ({np.median(y):.2f})")
    ax1.set_title("(a) Target Distribution & Upper Capping")
    ax1.set_xlabel("Median House Value ($100k)")
    ax1.set_ylabel("Block Group Count")
    ax1.grid(True, linestyle="--", alpha=0.25)
    ax1.legend(frameon=True, facecolor="white")

    # Standardized features before clipping to highlight extreme raw outliers
    X_z = (X - np.nanmean(X, axis=0)) / (np.nanstd(X, axis=0) + 1e-8)
    bp = ax2.boxplot(
        [X_z[:, i] for i in range(X.shape[1])],
        labels=feature_names,
        vert=True,
        patch_artist=True,
        flierprops=dict(marker="o", markersize=2, alpha=0.35, markeredgecolor="#e67e22"),
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#85c1e9")
        patch.set_edgecolor("#2b5c8f")
    ax2.set_ylim(-5, 35)
    ax2.set_title("(b) Raw Feature Outliers (Z-Score Scale)")
    ax2.set_ylabel("Standard Deviations from Mean")
    ax2.tick_params(axis="x", rotation=30)
    ax2.grid(True, linestyle="--", alpha=0.25)

    plt.tight_layout()
    plt.savefig(out_dir / "eda1_target_and_skew.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # EDA PLOT 2: Feature & Target Correlation Heatmap
    # ------------------------------------------------------------------
    data_all = np.column_stack([X, y])
    labels_all = list(feature_names) + ["MedHouseVal"]
    corr = np.corrcoef(data_all, rowvar=False)

    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Pearson Correlation ($r$)", fontweight="bold")

    ax.set_xticks(np.arange(len(labels_all)))
    ax.set_yticks(np.arange(len(labels_all)))
    ax.set_xticklabels(labels_all, rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(labels_all, fontsize=9)
    ax.set_title("Feature & Target Pearson Correlation Matrix")

    for i in range(len(labels_all)):
        for j in range(len(labels_all)):
            val = corr[i, j]
            color = "white" if abs(val) > 0.55 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.tight_layout()
    plt.savefig(out_dir / "eda2_correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    # ------------------------------------------------------------------
    # EDA PLOT 3: Spatial Distribution & Regional Heteroscedasticity
    # ------------------------------------------------------------------
    if "Latitude" in feature_names and "Longitude" in feature_names:
        lat_idx = feature_names.index("Latitude")
        lon_idx = feature_names.index("Longitude")
        inc_idx = feature_names.index("MedInc") if "MedInc" in feature_names else 0

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))

        sc1 = ax1.scatter(
            X[:, lon_idx], X[:, lat_idx], c=y, cmap="viridis",
            s=8, alpha=0.65, edgecolors="none"
        )
        cbar1 = plt.colorbar(sc1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label("MedHouseVal ($100k)", fontweight="bold")
        ax1.set_title("(a) Geospatial Distribution of House Values")
        ax1.set_xlabel("Longitude")
        ax1.set_ylabel("Latitude")
        ax1.grid(True, linestyle="--", alpha=0.25)

        # Income vs. House Value showing widening variance (heteroscedasticity)
        ax2.scatter(X[:, inc_idx], y, s=6, alpha=0.25, color="#16a085", edgecolors="none")
        ax2.axhline(5.0, color="#c0392b", linestyle="--", lw=1.3, label="Target Cap ($500k)")
        ax2.set_title("(b) Income vs. Value Heteroscedastic Spread")
        ax2.set_xlabel("Median Income ($10k units)")
        ax2.set_ylabel("Median House Value ($100k)")
        ax2.grid(True, linestyle="--", alpha=0.25)
        ax2.legend(frameon=True, facecolor="white")

        plt.tight_layout()
        plt.savefig(out_dir / "eda3_spatial_heteroscedasticity.png", bbox_inches="tight")
        plt.close()

def train_ablation_variant(seed, D, C, args, lam_kd, lam_rel, tau, beta):
    """Train a Student instance under a specific UAR-BD ablation configuration."""
    set_seed(seed)
    in_dim = D["X_train"].size(1)
    model = Student(in_dim=in_dim)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.s_lr, weight_decay=args.s_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.s_epochs)
    n = D["X_train"].size(0)
    bs = min(args.s_bs, max(16, n // 4))
    best, best_state, bad = float("inf"), None, 0

    for ep in range(args.s_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if b.numel() < 8:
                continue
            xb, yb = D["X_train"][b], D["y_train"][b]
            mu, lv, zs = model(xb, with_latent=True)

            task = F.mse_loss(mu, yb)
            loss = task

            if lam_kd > 0.0:
                gate = teacher_gate(yb, C["mu"][b], C["logvar"][b], tau, beta)
                kl = (gate * gaussian_kl(C["mu"][b], C["logvar"][b], mu, lv)).mean()
                loss = loss + lam_kd * kl

            if lam_rel > 0.0:
                rel = relational_loss(zs, C["h"][b])
                loss = loss + lam_rel * rel

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        sched.step()
        model.eval()
        with torch.no_grad():
            val = F.mse_loss(model(D["X_val"])[0], D["y_val"]).item()
        if val < best - 1e-6:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.s_patience:
                break

    model.load_state_dict(best_state)
    model.eval()
    return model


def generate_plots(out_dir, main_data, ablation_results, robustness_results, sigmas, meta):
    """Dynamically extract metrics from results.json and current runs to save 4 figures."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 10,
        "axes.labelweight": "bold",
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "legend.fontsize": 9,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
    })

    COLORS = {
        "Teacher": "#2b5c8f",
        "Stand-alone": "#7f8c8d",
        "Naive KD": "#e67e22",
        "UAR-BD": "#16a085",
        "Ablation1": "#2980b9",
        "Ablation2": "#8e44ad",
        "TaskMSE": "#95a5a6",
    }

    res = main_data["results"]
    prof = main_data["profile"]
    t_log = main_data["teacher_training"]
    unit_lbl = meta.get("target_unit", "target units")

    model_keys = [
        "Teacher ensemble",
        "Arm 1: Stand-alone (NLL)",
        "Arm 2: Naive KD (MSE mix)",
        "Arm 3: UAR-BD",
    ]
    model_labels = [
        "Teacher\nEnsemble",
        "Arm 1:\nStand-Alone",
        "Arm 2:\nNaive KD",
        "Arm 3:\nUAR-BD (Ours)",
    ]

    def get_stat(m_key, metric):
        vals = np.array([entry[metric] for entry in res[m_key]["metrics"]])
        return float(vals.mean()), (float(vals.std()) if len(vals) > 1 else 0.0)

    r2_means, r2_stds = zip(*[get_stat(k, "R2") for k in model_keys])
    mae_means, mae_stds = zip(*[get_stat(k, "MAE") for k in model_keys])

    latencies = [
        prof["teacher"]["median_ms"],
        prof["Arm 1: Stand-alone (NLL)"]["median_ms"],
        prof["Arm 2: Naive KD (MSE mix)"]["median_ms"],
        prof["Arm 3: UAR-BD"]["median_ms"],
    ]
    t_params = prof["teacher"]["params"]
    s_deploy = prof["Arm 3: UAR-BD"]["params_deploy"]
    comp_ratio = main_data["compression"]["deploy"]
    speedup_ratio = latencies[0] / latencies[3]

    # ==========================================================================
    # FIGURE 1: MAIN PREDICTIVE ACCURACY & KNOWLEDGE RECOVERY
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    bar_colors = [COLORS["Teacher"], COLORS["Stand-alone"], COLORS["Naive KD"], COLORS["UAR-BD"]]
    x = np.arange(len(model_labels))

    bars1 = ax1.bar(
        x, r2_means, yerr=r2_stds, capsize=5, color=bar_colors, width=0.58,
        edgecolor="black", linewidth=0.7, error_kw={"elinewidth": 1.2, "ecolor": "#2c3e50"}
    )
    r2_span = max(0.015, max(r2_means) - min(r2_means))
    ax1.set_ylim(min(r2_means) - 0.6 * r2_span, max(r2_means) + 0.45 * r2_span)
    ax1.set_ylabel("Test $R^2$ Score (Higher is Better)")
    ax1.set_title("(a) Predictive Accuracy ($R^2$) & Seed Stability")
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_labels)

    ax1.axhline(r2_means[1], color=COLORS["Stand-alone"], linestyle=":", linewidth=1.1, alpha=0.8)
    ax1.axhline(r2_means[0], color=COLORS["Teacher"], linestyle="--", linewidth=1.1, alpha=0.7)

    for idx, (bar, mean, std) in enumerate(zip(bars1, r2_means, r2_stds)):
        label = f"{mean:.4f}" if std == 0 else f"{mean:.4f}\n±{std:.4f}"
        ax1.text(
            bar.get_x() + bar.get_width() / 2, mean + std + 0.04 * r2_span, label,
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold" if idx == 3 else "normal",
            color="#16a085" if idx == 3 else "#2c3e50",
        )

    gap = r2_means[0] - r2_means[1]
    if gap > 1e-6:
        recovery = (r2_means[3] - r2_means[1]) / gap * 100.0
        ax1.annotate(
            f"Recovers {recovery:.1f}% of\nTeacher-Student Gap",
            xy=(3, r2_means[3]), xytext=(1.75, r2_means[0] - 0.2 * r2_span),
            arrowprops=dict(arrowstyle="->", color="#16a085", lw=1.3),
            fontsize=8.5, fontweight="bold", color="#117864",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085", lw=0.8),
        )

    bars2 = ax2.bar(
        x, mae_means, yerr=mae_stds, capsize=5, color=bar_colors, width=0.58,
        edgecolor="black", linewidth=0.7, error_kw={"elinewidth": 1.2, "ecolor": "#2c3e50"}
    )
    mae_span = max(0.015, max(mae_means) - min(mae_means))
    ax2.set_ylim(max(0.0, min(mae_means) - 0.7 * mae_span), max(mae_means) + 0.65 * mae_span)
    ax2.set_ylabel(f"Test MAE in {unit_lbl} (Lower is Better)")
    ax2.set_title("(b) Mean Absolute Error (MAE) Comparison")
    ax2.set_xticks(x)
    ax2.set_xticklabels(model_labels)

    for idx, (bar, mean, std) in enumerate(zip(bars2, mae_means, mae_stds)):
        label = f"{mean:.4f}" if std == 0 else f"{mean:.4f}\n±{std:.4f}"
        ax2.text(
            bar.get_x() + bar.get_width() / 2, mean + std + 0.04 * mae_span, label,
            ha="center", va="bottom", fontsize=8.5,
            fontweight="bold" if idx == 3 else "normal",
            color="#16a085" if idx == 3 else "#2c3e50",
        )

    plt.tight_layout()
    plt.savefig(out_dir / "fig1_main_accuracy.png", bbox_inches="tight")
    plt.close()

    # ==========================================================================
    # FIGURE 2: STEPWISE ABLATION STUDY
    # ==========================================================================
    abl_order = [
        ("Task MSE Only (No KD)", "Task MSE Only\n(No KD)"),
        ("w/o Error Gating (tau=100)", "w/o Error Gating\n(tau=100)"),
        ("w/o Relational Loss", "w/o Relational\nLoss (lam_rel=0)"),
        ("Full UAR-BD (Proposed)", "Full UAR-BD\n(Proposed)"),
    ]
    abl_r2_m, abl_r2_s, abl_mae_m, abl_mae_s, abl_xlabels = [], [], [], [], []
    for raw_k, disp_lbl in abl_order:
        entries = ablation_results[raw_k]
        r2_v = np.array([e["R2"] for e in entries])
        mae_v = np.array([e["MAE"] for e in entries])
        abl_r2_m.append(float(r2_v.mean()))
        abl_r2_s.append(float(r2_v.std()))
        abl_mae_m.append(float(mae_v.mean()))
        abl_mae_s.append(float(mae_v.std()))
        abl_xlabels.append(disp_lbl)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    abl_colors = [COLORS["TaskMSE"], COLORS["Ablation2"], COLORS["Ablation1"], COLORS["UAR-BD"]]
    xa = np.arange(len(abl_xlabels))

    bars_a1 = ax1.bar(xa, abl_r2_m, yerr=abl_r2_s, capsize=5, color=abl_colors, width=0.55, edgecolor="black", linewidth=0.7)
    a1_span = max(0.01, max(abl_r2_m) - min(abl_r2_m))
    ax1.set_ylim(min(abl_r2_m) - 0.8 * a1_span, max(abl_r2_m) + 0.85 * a1_span)
    ax1.set_ylabel("Test $R^2$ Score (Higher is Better)")
    ax1.set_title("(a) Stepwise Ablation Impact on $R^2$")
    ax1.set_xticks(xa)
    ax1.set_xticklabels(abl_xlabels, fontsize=8.5)

    for idx, (bar, mean, std) in enumerate(zip(bars_a1, abl_r2_m, abl_r2_s)):
        ax1.text(
            bar.get_x() + bar.get_width() / 2, mean + std + 0.06 * a1_span, f"{mean:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold" if idx == 3 else "normal"
        )

    bars_a2 = ax2.bar(xa, abl_mae_m, yerr=abl_mae_s, capsize=5, color=abl_colors, width=0.55, edgecolor="black", linewidth=0.7)
    a2_span = max(0.01, max(abl_mae_m) - min(abl_mae_m))
    ax2.set_ylim(max(0.0, min(abl_mae_m) - 0.6 * a2_span), max(abl_mae_m) + 0.75 * a2_span)
    ax2.set_ylabel(f"Test MAE in {unit_lbl} (Lower is Better)")
    ax2.set_title("(b) Stepwise Ablation Impact on MAE")
    ax2.set_xticks(xa)
    ax2.set_xticklabels(abl_xlabels, fontsize=8.5)

    for idx, (bar, mean, std) in enumerate(zip(bars_a2, abl_mae_m, abl_mae_s)):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, mean + std + 0.05 * a2_span, f"{mean:.4f}",
            ha="center", va="bottom", fontsize=8.5, fontweight="bold" if idx == 3 else "normal"
        )

    plt.tight_layout()
    plt.savefig(out_dir / "fig2_ablation_study.png", bbox_inches="tight")
    plt.close()

    # ==========================================================================
    # FIGURE 3: GAUSSIAN NOISE ROBUSTNESS STRESS TEST
    # ==========================================================================
    rob_teacher = [robustness_results["Teacher ensemble"][str(s)]["R2"] for s in sigmas]

    def extract_rob(arm_label):
        means, stds = [], []
        for s in sigmas:
            vals = np.array([m["R2"] for m in robustness_results[arm_label][str(s)]])
            means.append(float(vals.mean()))
            stds.append(float(vals.std()))
        return np.array(means), np.array(stds)

    rob_sa_m, rob_sa_s = extract_rob("Arm 1: Stand-alone (NLL)")
    rob_nk_m, rob_nk_s = extract_rob("Arm 2: Naive KD (MSE mix)")
    rob_ub_m, rob_ub_s = extract_rob("Arm 3: UAR-BD (Proposed)")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.6))

    ax1.plot(sigmas, rob_teacher, "o--", color=COLORS["Teacher"], lw=2, markersize=6, label=f"Teacher Ensemble ({t_params / 1000:.1f}k params)")
    ax1.errorbar(sigmas, rob_ub_m, yerr=rob_ub_s, fmt="s-", color=COLORS["UAR-BD"], lw=2.3, markersize=6.5, capsize=4, label=f"Arm 3: UAR-BD Student ({s_deploy / 1000:.1f}k params)")
    ax1.errorbar(sigmas, rob_nk_m, yerr=rob_nk_s, fmt="^-", color=COLORS["Naive KD"], lw=1.8, markersize=6, capsize=4, label=f"Arm 2: Naive KD Student ({s_deploy / 1000:.1f}k params)")
    ax1.errorbar(sigmas, rob_sa_m, yerr=rob_sa_s, fmt="d:", color=COLORS["Stand-alone"], lw=1.8, markersize=5.5, capsize=4, label=f"Arm 1: Stand-Alone Student ({s_deploy / 1000:.1f}k params)")

    ax1.set_xlabel("Gaussian Feature Noise Level ($\\sigma$ in std units)")
    ax1.set_ylabel("Test $R^2$ Score")
    ax1.set_title("(a) Predictive Degradation Under Input Noise")
    ax1.set_xticks(sigmas)
    ax1.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.9)

    diff_vs_sa = (rob_ub_m - rob_sa_m) * 100.0
    diff_vs_nk = (rob_ub_m - rob_nk_m) * 100.0
    x_s = np.arange(len(sigmas))
    w = 0.35

    b1 = ax2.bar(x_s - w / 2, diff_vs_sa, width=w, color="#34495e", edgecolor="black", linewidth=0.6, label="Gain over Stand-Alone ($\\Delta R^2 \\times 100$)")
    b2 = ax2.bar(x_s + w / 2, diff_vs_nk, width=w, color=COLORS["UAR-BD"], edgecolor="black", linewidth=0.6, label="Gain over Naive KD ($\\Delta R^2 \\times 100$)")

    ax2.axhline(0, color="black", linewidth=0.8)
    ax2.set_xlabel("Gaussian Feature Noise Level ($\\sigma$ in std units)")
    ax2.set_ylabel("$R^2$ Advantage ($\\Delta R^2 \\times 100$ points)")
    ax2.set_title("(b) UAR-BD Net $R^2$ Gain Under Increasing Noise")
    ax2.set_xticks(x_s)
    ax2.set_xticklabels([f"$\\sigma={s:.2f}$" for s in sigmas])
    y_min_b = min(0.0, float(diff_vs_nk.min()) - 0.35, float(diff_vs_sa.min()) - 0.35)
    y_max_b = max(0.5, float(diff_vs_sa.max()), float(diff_vs_nk.max())) + 0.45
    ax2.set_ylim(y_min_b, y_max_b)
    ax2.legend(loc="upper left", frameon=True, facecolor="white", framealpha=0.9)

    for bar in b1:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.05 if h >= 0 else -0.05
        sign = "+" if h >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{sign}{h:.2f}", ha="center", va=va, fontsize=8)
    for bar in b2:
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        offset = 0.05 if h >= 0 else -0.05
        sign = "+" if h >= 0 else ""
        ax2.text(bar.get_x() + bar.get_width() / 2, h + offset, f"{sign}{h:.2f}", ha="center", va=va, fontsize=8, fontweight="bold", color="#117864")

    plt.tight_layout()
    plt.savefig(out_dir / "fig3_noise_robustness.png", bbox_inches="tight")
    plt.close()

    # ==========================================================================
    # FIGURE 4: PARETO FRONTIER & TEACHER ENSEMBLE CALIBRATION
    # ==========================================================================
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    p_colors = [COLORS["Teacher"], COLORS["Stand-alone"], COLORS["Naive KD"], COLORS["UAR-BD"]]
    sizes = [450, 140, 140, 200]

    for i in range(len(model_labels)):
        ax1.scatter(latencies[i], r2_means[i], s=sizes[i], color=p_colors[i], edgecolors="black", linewidth=1.2, zorder=4)

    p_span = max(0.015, max(r2_means) - min(r2_means))
    ax1.annotate(
        f"Teacher Ensemble\n({t_params:,} params | {latencies[0]:.3f} ms)",
        xy=(latencies[0], r2_means[0]), xytext=(latencies[0] * 0.54, r2_means[0] - 0.12 * p_span),
        arrowprops=dict(arrowstyle="->", color="#2b5c8f", lw=1),
        fontsize=8.5, fontweight="bold", color="#2b5c8f",
    )
    ax1.annotate(
        f"Arm 3: UAR-BD (Ours)\n({s_deploy:,} params | {latencies[3]:.3f} ms)\n[{comp_ratio:.1f}x smaller, {speedup_ratio:.1f}x faster]",
        xy=(latencies[3], r2_means[3]), xytext=(latencies[0] * 0.19, r2_means[3]),
        arrowprops=dict(arrowstyle="->", color="#16a085", lw=1.2),
        fontsize=8.5, fontweight="bold", color="#117864",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#e8f8f5", edgecolor="#16a085", lw=0.8),
    )
    ax1.annotate(
        f"Arm 2: Naive KD ({latencies[2]:.3f} ms)",
        xy=(latencies[2], r2_means[2]), xytext=(latencies[0] * 0.19, r2_means[2] - 0.12 * p_span),
        arrowprops=dict(arrowstyle="->", color="#e67e22", lw=1),
        fontsize=8, color="#b9770e",
    )
    ax1.annotate(
        f"Arm 1: Stand-Alone ({latencies[1]:.3f} ms)",
        xy=(latencies[1], r2_means[1]), xytext=(latencies[0] * 0.19, r2_means[1] - 0.05 * p_span),
        arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1),
        fontsize=8, color="#566573",
    )

    ax1.set_xlim(-0.02, latencies[0] * 1.13)
    ax1.set_ylim(min(r2_means) - 0.2 * p_span, max(r2_means) + 0.2 * p_span)
    ax1.set_xlabel("Single-Sample CPU Inference Latency (ms, Median)")
    ax1.set_ylabel("Test $R^2$ Score")
    ax1.set_title("(a) Accuracy vs. Latency Trade-Off (Pareto Frontier)")

    exp_labels = [f"Expert {i + 1}\n({e['expert']})" for i, e in enumerate(t_log)]
    exp_nlls = [e["val_nll"] for e in t_log]
    exp_eps = [e["epochs"] for e in t_log]
    xe = np.arange(len(exp_labels))
    e_colors = ["#3498db", "#2ecc71", "#1abc9c", "#9b59b6", "#e67e22"]

    bars_e = ax2.bar(xe, exp_nlls, color=e_colors, width=0.55, edgecolor="black", linewidth=0.7)
    nll_span = max(0.05, max(exp_nlls) - min(exp_nlls))
    ax2.set_ylim(min(exp_nlls) - 0.6 * nll_span, max(exp_nlls) + 0.6 * nll_span)
    ax2.set_ylabel("Best Validation Gaussian NLL (Lower is Better)")
    ax2.set_title("(b) Heterogeneous Teacher Ensemble Calibration")
    ax2.set_xticks(xe)
    ax2.set_xticklabels(exp_labels, fontsize=8.5)

    for bar, nll, ep in zip(bars_e, exp_nlls, exp_eps):
        ax2.text(
            bar.get_x() + bar.get_width() / 2, nll - 0.12 * nll_span, f"{nll:.4f}\n({ep} ep)",
            ha="center", va="top", fontsize=8, color="white", fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(out_dir / "fig4_efficiency_and_teacher.png", bbox_inches="tight")
    plt.close()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=None, help="local CSV instead of California Housing")
    p.add_argument("--target", default=None, help="target column name in CSV (defaults to last col)")
    p.add_argument("--outdir", default="kd_results")
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--teacher-seed", type=int, default=100)
    p.add_argument("--student-seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--threads", type=int, default=2)
    # teacher
    p.add_argument("--t-epochs", type=int, default=80)
    p.add_argument("--t-patience", type=int, default=15)
    p.add_argument("--t-lr", type=float, default=2e-3)
    p.add_argument("--t-wd", type=float, default=1e-4)
    p.add_argument("--t-bs", type=int, default=256)
    # student
    p.add_argument("--s-epochs", type=int, default=150)
    p.add_argument("--s-patience", type=int, default=25)
    p.add_argument("--s-lr", type=float, default=3e-3)
    p.add_argument("--s-wd", type=float, default=1e-4)
    p.add_argument("--s-bs", type=int, default=256)
    # UAR-BD defaults
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--lam-kd", type=float, default=2.0)
    p.add_argument("--lam-rel", type=float, default=0.05)
    p.add_argument("--tau", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=4.0)
    # Noise stress test levels (in standardized feature units)
    p.add_argument("--noise-sigmas", type=float, nargs="+", default=[0.0, 0.05, 0.10, 0.20])
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1 & 2: Data & Frozen Teacher
    banner("INITIALIZING DATA & FROZEN TEACHER ENSEMBLE")
    X, y, meta = load_dataset(args.csv, target_col=args.target)
    
    generate_eda_plots(out, X, y, meta["feature_names"])
    P = prepare(X, y, seed=args.split_seed, scale_target=meta["scale_target"])
    D = {k: torch.from_numpy(v) for k, v in P.items() if isinstance(v, np.ndarray) and k.startswith(("X_", "y_"))}    
    in_dim = D["X_train"].size(1)

    experts = []
    t0 = time.time()
    for m, (name, act) in enumerate(ACTIVATIONS):
        e = Expert(act, in_dim=in_dim)
        train_expert(e, D, args, seed=args.teacher_seed + m)
        experts.append(e)
    teacher = FrozenEnsemble(experts)
    C = cache_teacher(teacher, D["X_train"])
    print(f"  Dataset [{meta['dataset_name']}] (D={in_dim}) | Teacher cached in {time.time() - t0:.1f}s")

    # ---- Stage 5: Ablation Study
    banner(f"STAGE 5: UAR-BD ABLATION STUDY [{meta['dataset_name']}] (Seeds={args.student_seeds})")
    ablation_configs = [
        ("Full UAR-BD (Proposed)",     {"lam_kd": args.lam_kd, "lam_rel": args.lam_rel, "tau": args.tau, "beta": args.beta}),
        ("w/o Relational Loss",        {"lam_kd": args.lam_kd, "lam_rel": 0.0,          "tau": args.tau, "beta": args.beta}),
        ("w/o Error Gating (tau=100)", {"lam_kd": args.lam_kd, "lam_rel": args.lam_rel, "tau": 100.0,    "beta": args.beta}),
        ("Task MSE Only (No KD)",      {"lam_kd": 0.0,         "lam_rel": 0.0,          "tau": args.tau, "beta": args.beta}),
    ]

    ablation_results = {}
    ablation_rows = []
    uarbd_models = {}

    for label, cfg in ablation_configs:
        metrics_list = []
        for seed in args.student_seeds:
            mdl = train_ablation_variant(seed, D, C, args, **cfg)
            if label == "Full UAR-BD (Proposed)":
                uarbd_models[seed] = mdl
            with torch.no_grad():
                mu = inverse_target(mdl(D["X_test"])[0].numpy(), P)
            metrics_list.append(regression_metrics(P["y_test_orig"], mu))

        ablation_results[label] = metrics_list
        row = [label]
        for k in ["MAE", "MSE", "RMSE", "R2"]:
            vals = np.array([m[k] for m in metrics_list])
            row.append(f"{vals.mean():.4f} ± {vals.std():.4f}")
        ablation_rows.append(row)

    ablation_table = format_table(["Configuration", "MAE", "MSE", "RMSE", "R2"], ablation_rows)
    print("\n" + ablation_table)

    # ---- Stage 6: Feature Noise Robustness Stress Test
    banner(f"STAGE 6: GAUSSIAN NOISE ROBUSTNESS STRESS TEST (Sigmas={args.noise_sigmas})")
    arm1_models, arm2_models = {}, {}
    for seed in args.student_seeds:
        arm1_models[seed], _ = train_student(1, seed, D, C, args)
        arm2_models[seed], _ = train_student(2, seed, D, C, args)

    robustness_results = {
        "Teacher ensemble": {},
        "Arm 1: Stand-alone (NLL)": {},
        "Arm 2: Naive KD (MSE mix)": {},
        "Arm 3: UAR-BD (Proposed)": {},
    }

    noise_rows = []
    for sigma in args.noise_sigmas:
        gen_noise = torch.Generator().manual_seed(999 + int(sigma * 1000))
        noise = torch.randn(D["X_test"].shape, generator=gen_noise) * sigma
        X_noisy = D["X_test"] + noise

        with torch.no_grad():
            t_mu = inverse_target(teacher(X_noisy)[0].numpy(), P)
        t_m = regression_metrics(P["y_test_orig"], t_mu)
        robustness_results["Teacher ensemble"][str(sigma)] = t_m

        arm_r2_strs = {}
        for arm_name, mdl_dict in [
            ("Arm 1: Stand-alone (NLL)", arm1_models),
            ("Arm 2: Naive KD (MSE mix)", arm2_models),
            ("Arm 3: UAR-BD (Proposed)", uarbd_models),
        ]:
            seed_ms = []
            for seed in args.student_seeds:
                with torch.no_grad():
                    s_mu = inverse_target(mdl_dict[seed](X_noisy)[0].numpy(), P)
                seed_ms.append(regression_metrics(P["y_test_orig"], s_mu))
            robustness_results[arm_name][str(sigma)] = seed_ms
            r2_vals = np.array([m["R2"] for m in seed_ms])
            arm_r2_strs[arm_name] = f"{r2_vals.mean():.4f} ± {r2_vals.std():.4f}"

        noise_rows.append([
            f"sigma = {sigma:.2f}",
            f"{t_m['R2']:.4f}",
            arm_r2_strs["Arm 1: Stand-alone (NLL)"],
            arm_r2_strs["Arm 2: Naive KD (MSE mix)"],
            arm_r2_strs["Arm 3: UAR-BD (Proposed)"],
        ])

    noise_table = format_table(
        ["Noise Level", "Teacher R2", "Stand-alone R2", "Naive KD R2", "UAR-BD R2"],
        noise_rows,
    )
    print("\n" + noise_table)

    # Save report & JSON
    report = "\n".join([
        f"# Ablation & Noise Robustness Report: {meta['dataset_name']} (Test N={D['X_test'].size(0)}, Seeds={args.student_seeds})",
        "\n## 1. UAR-BD Ablation Study\n",
        ablation_table,
        "\n## 2. Feature Perturbation Robustness (Test R2 under Gaussian Noise)\n",
        noise_table + "\n",
    ])
    (out / "ablation_robustness.md").write_text(report)
    (out / "ablation_robustness.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "meta": meta,
                "ablation": ablation_results,
                "robustness": robustness_results,
            },
            indent=2,
        )
    )

    # ---- Stage 7: Dynamic Visualization Generation
    banner("STAGE 7: GENERATING PUBLICATION-GRADE VISUALIZATION FIGURES")
    results_json_path = out / "results.json"
    if results_json_path.exists():
        main_data = json.loads(results_json_path.read_text())
        generate_plots(out, main_data, ablation_results, robustness_results, args.noise_sigmas, meta)
        print(f"  Generated 4 high-resolution (300 DPI) figures in: {out.resolve()}")
        for fname in [
            "fig1_main_accuracy.png",
            "fig2_ablation_study.png",
            "fig3_noise_robustness.png",
            "fig4_efficiency_and_teacher.png",
        ]:
            print(f"    - {fname}")
    else:
        print(f"  [Warning] {results_json_path} not found. Run `python kd_california.py` first to generate main benchmark plots.")

    print(f"\nSaved all ablation, robustness & plot artifacts to: {out.resolve()}")


if __name__ == "__main__":
    main()