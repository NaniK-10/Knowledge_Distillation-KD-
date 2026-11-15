"""Knowledge distillation for continuous tabular regression (dataset-agnostic).

Teacher: heterogeneous deep ensemble (5 experts, heteroscedastic Gaussian NLL),
         frozen, aggregated by moment matching, cached once on the train split.
Students (same architecture, same init seed, same batch order):
    Arm 1  stand-alone, student NLL on ground truth
    Arm 2  naive KD, alpha * MSE(gt) + (1 - alpha) * MSE(teacher mean)
    Arm 3  UAR-BD: MSE task loss + teacher-gated Gaussian KL + relational distance loss

Usage:
    python kd_california.py                                   # California Housing
    python kd_california.py --csv custom.csv                  # Any CSV (last col = target)
    python kd_california.py --csv custom.csv --target price   # Any CSV with named target
"""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_prep import load_dataset, prepare

LOGVAR_MIN, LOGVAR_MAX = -10.0, 5.0


# --------------------------------------------------------------------------- utils
def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def gaussian_nll(mu, logvar, y):
    """Heteroscedastic Gaussian NLL (constant 0.5*log(2*pi) dropped)."""
    return (0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))).mean()


def regression_metrics(y_true, y_pred):
    y_true, y_pred = y_true.ravel().astype(np.float64), y_pred.ravel().astype(np.float64)
    err = y_pred - y_true
    mse = float(np.mean(err ** 2))
    sst = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = 1.0 - float(np.sum(err ** 2)) / max(sst, 1e-12)
    return {
        "MAE": float(np.mean(np.abs(err))),
        "MSE": mse,
        "RMSE": mse ** 0.5,
        "R2": r2,
    }


def inverse_target(y_scaled, P):
    """Map predictions from standardized training space back to original target units."""
    return y_scaled * P["y_std"] + P["y_mean"]


def banner(title):
    line = "=" * 78
    print(f"\n{line}\n  {title}\n{line}")


# --------------------------------------------------------------------------- teacher
ACTIVATIONS = [
    ("SiLU", nn.SiLU),
    ("GELU", nn.GELU),
    ("Mish", nn.Mish),
    ("LeakyReLU", lambda: nn.LeakyReLU(0.01)),
    ("ELU", nn.ELU),
]


class Expert(nn.Module):
    """D -> [256, 128, 64] (Linear, LayerNorm, act, Dropout) -> 64 -> (mu, logvar)."""

    def __init__(self, act_factory, in_dim, hidden=(256, 128, 64), p=0.05):
        super().__init__()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), act_factory(), nn.Dropout(p)]
            d = h
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(hidden[-1], 2)

    def forward(self, x):
        h = self.backbone(x)                       # latent features h_t, [B, 64]
        out = self.head(h)
        mu = out[:, :1]
        logvar = out[:, 1:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        return mu, logvar, h


def train_expert(model, D, args, seed):
    set_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=args.t_lr, weight_decay=args.t_wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.t_epochs)
    n = D["X_train"].size(0)
    bs = min(args.t_bs, max(16, n // 4))
    best, best_state, bad = float("inf"), None, 0
    for ep in range(args.t_epochs):
        model.train()
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if b.numel() < 2:
                continue
            mu, lv, _ = model(D["X_train"][b])
            loss = gaussian_nll(mu, lv, D["y_train"][b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        sched.step()
        model.eval()
        with torch.no_grad():
            mu, lv, _ = model(D["X_val"])
            val = gaussian_nll(mu, lv, D["y_val"]).item()
        if val < best - 1e-5:
            best, best_state, bad = val, copy.deepcopy(model.state_dict()), 0
        else:
            bad += 1
            if bad >= args.t_patience:
                break
    model.load_state_dict(best_state)
    return best, ep + 1


class FrozenEnsemble(nn.Module):
    """Analytic moment matching of M Gaussian experts."""

    def __init__(self, experts):
        super().__init__()
        self.experts = nn.ModuleList(experts)
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def forward(self, x):
        outs = [e(x) for e in self.experts]
        mus = torch.stack([o[0] for o in outs])    # [M, B, 1]
        lvs = torch.stack([o[1] for o in outs])
        hs = torch.stack([o[2] for o in outs])     # [M, B, 64]
        mu = mus.mean(0)
        aleatoric = lvs.exp().mean(0)              # mean of expert variances
        epistemic = (mus - mu).pow(2).mean(0)      # disagreement = E[mu_m^2] - mu^2
        total_var = aleatoric + epistemic
        logvar = total_var.clamp_min(1e-10).log()
        return mu, logvar, hs.mean(0), aleatoric, epistemic


@torch.no_grad()
def cache_teacher(ens, X, chunk=4096):
    parts = [ens(X[i:i + chunk]) for i in range(0, X.size(0), chunk)]
    cat = lambda k: torch.cat([p[k] for p in parts])
    return {"mu": cat(0), "logvar": cat(1), "h": cat(2), "alea": cat(3), "epi": cat(4)}


# --------------------------------------------------------------------------- student
class Student(nn.Module):
    """D -> 64 -> 32 -> heads. `proj` (32 -> 64) is training-only (latent alignment)."""

    def __init__(self, in_dim, h1=64, h2=32, latent=64):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU()
        )
        self.head = nn.Linear(h2, 2)
        self.proj = nn.Linear(h2, latent)

    def forward(self, x, with_latent=False):
        z = self.trunk(x)
        out = self.head(z)
        mu = out[:, :1]
        logvar = out[:, 1:].clamp(LOGVAR_MIN, LOGVAR_MAX)
        if with_latent:
            return mu, logvar, self.proj(z)
        return mu, logvar

    def deploy_params(self):
        return n_params(self.trunk) + n_params(self.head)


# --------------------------------------------------------------------------- UAR-BD losses
def gaussian_kl(mu_t, lv_t, mu_s, lv_s):
    """KL( N(mu_t, e^lv_t) || N(mu_s, e^lv_s) ), per sample."""
    return 0.5 * (lv_s - lv_t + (torch.exp(lv_t) + (mu_t - mu_s) ** 2) * torch.exp(-lv_s) - 1.0)


def teacher_gate(y, mu_t, lv_t, tau, beta, target_keep=0.80):
    """Automatically sets cutoff at the `target_keep` quantile of normalized teacher errors
    if tau <= 0, otherwise uses the fixed user-specified tau."""
    z = (y - mu_t).abs() / torch.exp(0.5 * lv_t)
    cutoff = torch.quantile(z, target_keep) if tau <= 0 else tau
    return torch.sigmoid(beta * (cutoff - z)).detach()


def pairwise_dist(z, eps=1e-8):
    sq = (z * z).sum(1, keepdim=True) + (z * z).sum(1).unsqueeze(0) - 2.0 * z @ z.t()
    return torch.sqrt(sq.clamp_min(0.0) + eps)


def relational_loss(z_s, h_t):
    """Mini-batch pairwise-distance (RKD-style) alignment. Each distance matrix is
    divided by its own mean off-diagonal distance, then Huber-compared."""
    B = z_s.size(0)
    mask = ~torch.eye(B, dtype=torch.bool, device=z_s.device)
    Ds, Dt = pairwise_dist(z_s)[mask], pairwise_dist(h_t)[mask]
    return F.smooth_l1_loss(Ds / Ds.mean().clamp_min(1e-8), Dt / Dt.mean().clamp_min(1e-8))


# --------------------------------------------------------------------------- student training
ARM_NAMES = {
    1: "Arm 1: Stand-alone (NLL)",
    2: "Arm 2: Naive KD (MSE mix)",
    3: "Arm 3: UAR-BD",
}


def train_student(arm, seed, D, C, args):
    set_seed(seed)                                  # identical init for every arm
    in_dim = D["X_train"].size(1)
    model = Student(in_dim=in_dim)
    fingerprint = sum(p.double().sum().item() for p in model.parameters())
    gen = torch.Generator().manual_seed(seed)       # identical batch order for every arm
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
            if b.numel() < 8:                       # too small for pairwise terms
                continue
            xb, yb = D["X_train"][b], D["y_train"][b]
            if arm == 1:
                mu, lv = model(xb)
                loss = gaussian_nll(mu, lv, yb)
            elif arm == 2:
                mu, _ = model(xb)
                loss = args.alpha * F.mse_loss(mu, yb) + (1 - args.alpha) * F.mse_loss(mu, C["mu"][b])
            else:
                mu, lv, zs = model(xb, with_latent=True)
                task = F.mse_loss(mu, yb)
                gate = teacher_gate(yb, C["mu"][b], C["logvar"][b], args.tau, args.beta)
                kl = (gate * gaussian_kl(C["mu"][b], C["logvar"][b], mu, lv)).mean()
                rel = relational_loss(zs, C["h"][b])
                loss = task + args.lam_kd * kl + args.lam_rel * rel
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
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
    return model, {"best_val_mse": best, "epochs_run": ep + 1, "init_fingerprint": fingerprint}


# --------------------------------------------------------------------------- profiling
def latency_ms(fn, x, warmup=200, iters=2000):
    ts = []
    with torch.inference_mode():
        for _ in range(warmup):
            fn(x)
        for _ in range(iters):
            t0 = time.perf_counter()
            fn(x)
            ts.append((time.perf_counter() - t0) * 1e3)
    ts = np.asarray(ts)
    return {
        "median_ms": float(np.median(ts)),
        "mean_ms": float(ts.mean()),
        "p95_ms": float(np.percentile(ts, 95)),
    }


def format_table(headers, rows, alignments=None):
    """Render a cleanly aligned ASCII/Markdown table."""
    if alignments is None:
        alignments = ["<"] + [">"] * (len(headers) - 1)
    widths = [
        max(len(str(h)), max((len(str(r[i])) for r in rows), default=0))
        for i, h in enumerate(headers)
    ]

    def fmt_row(vals):
        cells = [f"{str(v):{alignments[i]}{widths[i]}}" for i, v in enumerate(vals)]
        return "| " + " | ".join(cells) + " |"

    sep_cells = []
    for i, w in enumerate(widths):
        if alignments[i] == ">":
            sep_cells.append("-" * (w - 1) + ":")
        else:
            sep_cells.append("-" * w)
    sep = "| " + " | ".join(sep_cells) + " |"
    return "\n".join([fmt_row(headers), sep] + [fmt_row(r) for r in rows])


# --------------------------------------------------------------------------- main
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
    # losses
    p.add_argument("--alpha", type=float, default=0.5, help="Arm 2: weight on ground-truth MSE")
    p.add_argument("--lam-kd", type=float, default=2.0, help="Arm 3: gated Gaussian KL weight")
    p.add_argument("--lam-rel", type=float, default=0.05, help="Arm 3: relational loss weight")
    p.add_argument("--tau", type=float, default=1.0, help="gate: tolerated |error| in teacher stds")
    p.add_argument("--beta", type=float, default=4.0, help="gate: sigmoid sharpness")
    return p.parse_args()


def main():
    args = parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    # ---- Stage 1
    banner("STAGE 1: DATA INGESTION & PREPROCESSING")
    X, y, meta = load_dataset(args.csv, target_col=args.target)
    P = prepare(X, y, seed=args.split_seed, scale_target=meta["scale_target"])
    D = {k: torch.from_numpy(v) for k, v in P.items() if isinstance(v, np.ndarray) and k.startswith(("X_", "y_"))}
    in_dim = D["X_train"].size(1)

    print(f"  Dataset     : {meta['dataset_name']} (Target: {meta['target_unit']})")
    print(f"  Raw Shape   : X={X.shape}, y={y.shape} (Input Features D={in_dim})")
    print(f"  Splits      : Train={D['X_train'].shape[0]:,} | Val={D['X_val'].shape[0]:,} | Test={D['X_test'].shape[0]:,}")

    # ---- Stage 2
    banner("STAGE 2: TEACHER ENSEMBLE TRAINING (5 HETEROGENEOUS EXPERTS)")
    experts, t_log, t_rows = [], [], []
    for m, (name, act) in enumerate(ACTIVATIONS):
        e = Expert(act, in_dim=in_dim)
        t0 = time.time()
        val_nll, ep = train_expert(e, D, args, seed=args.teacher_seed + m)
        elapsed = time.time() - t0
        experts.append(e)
        t_log.append({"expert": name, "val_nll": val_nll, "epochs": ep})
        t_rows.append([f"Expert {m + 1}", name, f"{n_params(e):,}", f"{val_nll:.4f}", ep, f"{elapsed:.1f}s"])
    print(format_table(["Expert", "Activation", "Params", "Best Val NLL", "Epochs", "Time"], t_rows, ["<", "<", ">", ">", ">", ">"]))

    teacher = FrozenEnsemble(experts)
    assert not any(p.requires_grad for p in teacher.parameters())
    teacher_params = n_params(teacher)

    banner("STAGE 3: OFFLINE REPRESENTATION CACHING & UNCERTAINTY GATE DIAGNOSTICS")
    C = cache_teacher(teacher, D["X_train"])
    mb = sum(t.numel() * t.element_size() for t in C.values()) / 1e6
    gate_all = teacher_gate(D["y_train"], C["mu"], C["logvar"], args.tau, args.beta)
    kept_pct = (gate_all > 0.5).float().mean().item() * 100.0
    print(f"  Cached Tensors   : mu={tuple(C['mu'].shape)}, h={tuple(C['h'].shape)} ({mb:.2f} MB)")
    print(f"  Uncertainty      : Aleatoric Var={C['alea'].mean():.4f} | Epistemic Var={C['epi'].mean():.4f}")
    print(f"  Gate Config      : tau={args.tau}, beta={args.beta}")
    print(f"  Gate Diagnostics : Mean Gate={gate_all.mean():.3f} | Samples Kept (>0.5)={kept_pct:.1f}%")

    with torch.no_grad():
        t_test_mu = inverse_target(teacher(D["X_test"])[0].numpy(), P)
    results = {"Teacher ensemble": {"metrics": [regression_metrics(P["y_test_orig"], t_test_mu)]}}

    # ---- Stage 4
    banner("STAGE 4: MULTI-ARM STUDENT TRAINING")
    models = {}
    for arm in (1, 2, 3):
        results[ARM_NAMES[arm]] = {"metrics": [], "train": []}

    for seed in args.student_seeds:
        fps, seed_rows = [], []
        for arm in (1, 2, 3):
            t0 = time.time()
            model, info = train_student(arm, seed, D, C, args)
            elapsed = time.time() - t0
            fps.append(info["init_fingerprint"])
            with torch.no_grad():
                mu = inverse_target(model(D["X_test"])[0].numpy(), P)
            m = regression_metrics(P["y_test_orig"], mu)
            results[ARM_NAMES[arm]]["metrics"].append(m)
            results[ARM_NAMES[arm]]["train"].append(info)
            models[(arm, seed)] = model
            seed_rows.append([
                ARM_NAMES[arm], f"{m['MAE']:.4f}", f"{m['MSE']:.4f}",
                f"{m['RMSE']:.4f}", f"{m['R2']:.4f}", info["epochs_run"], f"{elapsed:.1f}s"
            ])
        assert max(fps) - min(fps) < 1e-9, "student arms did not share an identical init"
        print(f"\n[Seed {seed}]")
        print(format_table(["Student Arm", "MAE", "MSE", "RMSE", "R2", "Epochs", "Time"], seed_rows))

    # ---- Stage 5: efficiency
    torch.set_num_threads(1)
    x1 = D["X_test"][:1]
    ref = models[(1, args.student_seeds[0])]
    s_total, s_deploy = n_params(ref), ref.deploy_params()
    prof = {"teacher": {"params": teacher_params, **latency_ms(lambda x: teacher(x), x1)}}
    for arm in (1, 2, 3):
        mdl = models[(arm, args.student_seeds[0])]
        prof[ARM_NAMES[arm]] = {
            "params_total": s_total,
            "params_deploy": s_deploy,
            **latency_ms(lambda x, m=mdl: m(x), x1),
        }
    comp_deploy, comp_total = teacher_params / s_deploy, teacher_params / s_total

    # ---- Final Report
    banner(f"FINAL EVALUATION SUMMARY [{meta['dataset_name']}] (Test N={D['X_test'].size(0):,} | Seeds={args.student_seeds})")
    keys = ["MAE", "MSE", "RMSE", "R2"]
    acc_rows = []
    for name, r in results.items():
        cells = [name]
        for k in keys:
            v = np.array([mm[k] for mm in r["metrics"]])
            cells.append(f"{v.mean():.4f}" + (f" ± {v.std():.4f}" if len(v) > 1 else ""))
        acc_rows.append(cells)
    acc_table = format_table(["Model"] + keys, acc_rows)
    print("\n1. Predictive Accuracy\n")
    print(acc_table)

    eff_rows = [[
        "Teacher ensemble (5 experts)",
        f"{teacher_params:,}",
        f"{prof['teacher']['median_ms']:.3f}",
        f"{prof['teacher']['mean_ms']:.3f}",
        f"{prof['teacher']['p95_ms']:.3f}",
    ]]
    for arm in (1, 2, 3):
        q = prof[ARM_NAMES[arm]]
        eff_rows.append([
            ARM_NAMES[arm],
            f"{s_deploy:,} deploy / {s_total:,} train",
            f"{q['median_ms']:.3f}",
            f"{q['mean_ms']:.3f}",
            f"{q['p95_ms']:.3f}",
        ])
    eff_table = format_table(
        ["Model", "Parameters", "Median (ms)", "Mean (ms)", "p95 (ms)"],
        eff_rows,
    )
    comp_line = (
        f"Compression Ratio (Teacher / Student): {comp_deploy:.1f}x deploy ({s_deploy:,} params) | "
        f"{comp_total:.1f}x train ({s_total:,} params)"
    )
    print("\n2. Computational Efficiency\n")
    print(eff_table)
    print(f"\n{comp_line}\n")

    report = "\n".join([
        f"# Knowledge Distillation Evaluation: {meta['dataset_name']} (Test N={D['X_test'].size(0)}, Seeds={args.student_seeds})",
        "\n## Predictive Accuracy\n",
        acc_table,
        "\n## Computational Efficiency\n",
        eff_table,
        f"\n{comp_line}\n",
    ])
    (out / "results.md").write_text(report)
    (out / "results.json").write_text(json.dumps(
        {
            "args": vars(args),
            "meta": meta,
            "teacher_training": t_log,
            "results": results,
            "profile": prof,
            "compression": {"deploy": comp_deploy, "total": comp_total},
        },
        indent=2,
    ))
    print(f"Saved artifacts to: {out.resolve()}")


if __name__ == "__main__":
    main()