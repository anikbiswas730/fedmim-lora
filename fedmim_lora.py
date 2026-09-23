# FedMIM-LoRA pipeline in percent format (same code as fedmim_lora.ipynb).
# Run top to bottom with `python fedmim_lora.py`, or open it as a notebook in VS Code / Jupytext.

# %% [markdown]
# # FedMIM-LoRA: masked image modeling keeps federated LoRA stable under client heterogeneity
#
# Frozen ViT-B/16 (ImageNet-21k), LoRA on q/v, 20 Dirichlet non-IID clients, CIFAR-100. Runs in one Kaggle session on 2 x T4.
#
# ## Question
#
# Federated fine-tuning usually gets worse as client data become more different from each other. This notebook asks whether that happens when the local objective is masked image modeling (MIM) instead of supervised cross-entropy, in the parameter-efficient setting: a frozen ViT, LoRA adapters, and only the adapters (plus a small decoder) sent to the server.
#
# Both objectives are trained on exactly the same partitions, client sampling and budget, so every comparison is paired by seed.
#
# ## Hypotheses and decision rules
#
# These were fixed before the runs.
#
# | id | hypothesis | measurement | supported if |
# |---|---|---|---|
# | H1 | MIM accuracy does not change with label skew | linear probe, $\alpha\in\{0.05,0.1,0.5\}$ vs IID, 3 seeds | 90% CI of $\mathrm{acc}_{IID}-\mathrm{acc}_{\alpha=0.05}$ inside $\pm0.5$ pt |
# | H2 | MIM is less sensitive to skew than supervised FedLoRA | difference-in-differences of that penalty | DiD $>0$ in every seed and 95% CI excludes 0 |
# | H3 | MIM client gradients are less dissimilar | noise-debiased $\kappa^2$ at a common parameter point | $\kappa^2$(MIM) < $\kappa^2$(SUP) at every non-IID $\alpha$; paired 95% CI of the difference excludes 0 at $\alpha=0.05$ |
# | H4 | MIM trains more smoothly under skew | per-round validation curve | backtracking(MIM) < backtracking(SUP) at $\alpha=0.05$ |
# | H5 | MIM helps supervised FL as an auxiliary loss | penalty of CE + MIM vs CE | penalty of the federated classifier lower at $\alpha=0.05$ |
# | P1 | the method is cheap | peak memory, throughput, uplink | < 4 GB at batch 64; LoRA uplink >= 100x and total uplink >= 25x below full fine-tuning |
#
# ## MIM with feature targets
#
# The student (frozen backbone + LoRA) sees 25% of the patches. A one-block decoder predicts, at the masked positions, the normalised features that the frozen backbone produces on the full image (mean of the last two blocks). A second term aligns the student [CLS] with the frozen [CLS] through a small projector. Pixel targets hurt a pretrained ViT in our earlier tests, so they are not used. Clients never see labels.
#
# ## Related work
#
# Self-supervised objectives have been reported to be more robust to non-IID data when training from scratch or fine-tuning the whole network (Wang et al., ICLR 2023; Yan et al., IEEE TMI 2023). Here the backbone is frozen and only LoRA adapters are trained. Freezing the LoRA $A$ matrix (FFA-LoRA, LoRA-FA) is available as an option but not used. Reference points: the unadapted backbone (about 85.6% probe / 84.4% kNN) and ILoRA (about 87.5% kNN on CIFAR-100).
#
# The supervised baseline uses labels on every client, so its absolute accuracy is expected to be higher. The comparison is about how much each method loses as the split becomes more skewed.
#
# ## Running it
#
# Enable Internet and the GPU T4 x2 accelerator, then Run All. `PRESET="full"` fits one session: every run is costed before launch and the least important runs are dropped first if time runs out. Runs are cached in `metrics/`, so attaching a previous output as input resumes instead of restarting. `PRESET="smoke"` takes about 15 minutes and touches every code path.

# %%
# ============================================================================================
# 0 · Environment
# ============================================================================================
import os, sys, subprocess, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# FEDMIM_CPU_TEST=1 is a unit-test harness only (tiny random ViT, synthetic data, CPU). Never set it on Kaggle.
TEST_MODE = os.environ.get("FEDMIM_CPU_TEST", "0") == "1"

TRY_INSTALL_BNB = True          # only used for one optional NF4 row of the cost table
HAS_BNB = False
if TRY_INSTALL_BNB and not TEST_MODE:
    try:
        import bitsandbytes as bnb  # noqa
        HAS_BNB = True
    except Exception:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"],
                           check=True, capture_output=True, timeout=600)
            import bitsandbytes as bnb  # noqa
            HAS_BNB = True
        except Exception as e:
            print(f"[warn] bitsandbytes unavailable ({type(e).__name__}); NF4 cost row skipped.")

import torch
print(f"python       : {sys.version.split()[0]}")
print(f"torch        : {torch.__version__}")
try:
    import transformers
    print(f"transformers : {transformers.__version__}  (only used as a weight-loading fallback)")
except Exception:
    print("transformers : not installed (not required)")
print(f"bitsandbytes : {'yes' if HAS_BNB else 'no'}")
print(f"CUDA         : {torch.version.cuda} | devices: {torch.cuda.device_count()}")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  [{i}] {p.name}  {p.total_memory/2**30:.1f} GiB  sm_{p.major}{p.minor}")
if not TEST_MODE:
    assert torch.cuda.is_available(), "Enable the GPU accelerator (T4 x2 recommended)."

# %%
import json, math, time, random, copy, gc, pickle, tarfile, urllib.request, re, shutil, glob
import threading, queue, hashlib, contextlib, traceback, textwrap, zipfile
from dataclasses import dataclass, asdict, field, replace as dc_replace
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import matplotlib
if TEST_MODE:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
try:
    from IPython.display import display
except Exception:                                   # plain-python execution (tests)
    def display(x):
        print(x)

# ---------------------------------------------------------------- output tree
ROOT = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path("./fedmim_out")
OUT = ROOT / "fedmim_lora"
DIRS = {k: OUT / k for k in ["checkpoints", "figures", "tables", "metrics", "logs"]}
for d in DIRS.values():
    d.mkdir(parents=True, exist_ok=True)
print("artifacts ->", OUT)

# ---------------------------------------------------------------- plot style (Okabe-Ito, greyscale-safe)
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#8C564B", "#000000"]
plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 400, "savefig.bbox": "tight",
    "font.family": "serif", "font.size": 11,
    "axes.titlesize": 12, "axes.labelsize": 11, "axes.grid": True,
    "grid.alpha": 0.25, "grid.linestyle": "--", "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": False, "legend.fontsize": 9.5,
    "lines.linewidth": 1.9, "lines.markersize": 5,
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
})

FIG_CAPTIONS: Dict[str, str] = {}
TAB_CAPTIONS: Dict[str, str] = {}

def savefig(fig, name: str, caption: str = ""):
    """400-dpi PNG + vector PDF + caption file: manuscript-ready as saved."""
    fig.savefig(DIRS["figures"] / f"{name}.png")
    fig.savefig(DIRS["figures"] / f"{name}.pdf")
    (DIRS["figures"] / f"{name}.caption.txt").write_text(caption)
    FIG_CAPTIONS[name] = caption
    if TEST_MODE:
        plt.close(fig)
    else:
        plt.show()

def save_table(df: pd.DataFrame, name: str, caption: str = "", label: str = "", floatfmt="%.2f"):
    """CSV for analysis + a LaTeX booktabs fragment ready to \\input{} into the manuscript."""
    df.to_csv(DIRS["tables"] / f"{name}.csv", index=False)
    try:
        tex = df.to_latex(index=False, escape=False, float_format=floatfmt,
                          column_format="l" + "r" * (df.shape[1] - 1),
                          caption=caption, label=f"tab:{label or name}")
    except Exception:
        tex = df.to_string(index=False)
    (DIRS["tables"] / f"{name}.tex").write_text(tex)
    (DIRS["tables"] / f"{name}.caption.txt").write_text(caption)
    TAB_CAPTIONS[name] = caption
    return df

def set_seed(s: int):
    random.seed(s); np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

def fmt_pct(x, nd=2):
    try:
        return f"{100*float(x):.{nd}f}"
    except Exception:
        return "--"

# ---------------------------------------------------------------- device-agnostic helpers
def is_cuda(dev) -> bool:
    return torch.device(dev).type == "cuda"

def amp_ctx(dev, enabled: bool = True):
    if enabled and is_cuda(dev):
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()

def dev_ctx(dev):
    return torch.cuda.device(dev) if is_cuda(dev) else contextlib.nullcontext()

def dsync(dev):
    if is_cuda(dev):
        torch.cuda.synchronize(dev)

def peak_reset(dev):
    if is_cuda(dev):
        torch.cuda.reset_peak_memory_stats(dev)

def peak_mb(dev) -> float:
    return torch.cuda.max_memory_allocated(dev) / 2 ** 20 if is_cuda(dev) else float("nan")

def alloc_mb(dev) -> float:
    return torch.cuda.memory_allocated(dev) / 2 ** 20 if is_cuda(dev) else float("nan")

def free_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
# %% [markdown]
# ## 1. Configuration
#
# A frozen dataclass describes a run; its JSON is stored with every metrics file and checkpoint.
#
# * `objective`: `"mim"` (masked feature modeling, no labels), `"sup"` (cross-entropy with a shared linear head, labels on every client) or `"sup_mim"` (cross-entropy + $\lambda_{mim}\cdot$MIM).
# * `lora_r=16` with trainable $A$ (standard FedLoRA aggregation). `fixed_a=True` freezes $A$.
# * `mask_ratio=0.75`, the best of {0.25, 0.5, 0.75, 0.9} in preliminary runs and 1.9x faster than no masking.
# * `share_heads=True`: the MIM decoder and projector are averaged like the LoRA weights. Uplink is reported for LoRA alone and for LoRA + heads.
# * `dirichlet_alpha=inf` is an exact IID split.
# * Model selection uses kNN on a 5,000-image server validation split that no client holds. The last-round result is reported as well.

# %%
PRESET = "full"               # <<<<<<  "smoke" (~15 min, every code path) | "full" (one 12-h session)
MODE = "train_and_test"       # "train_and_test" | "test_only" (no training: re-analyse cached runs)
IID = float("inf")

@dataclass
class Config:
    # ---- identity ---------------------------------------------------------
    name: str = "fedmim"
    objective: str = "mim"                   # "mim" | "sup" | "sup_mim"
    seed: int = 42

    # ---- backbone ---------------------------------------------------------
    model_name: str = "google/vit-base-patch16-224-in21k"
    img_size: int = 224
    patch_size: int = 16
    use_4bit: bool = False                   # NF4 frozen backbone (cost table only)
    grad_ckpt: bool = False

    # ---- masked image modeling head ---------------------------------------
    mask_ratio: float = 0.75
    dec_dim: int = 192
    dec_depth: int = 1
    dec_heads: int = 3
    norm_target: bool = True                 # per-token normalisation of the regression target
    tgt_layers: int = 2                      # average of the last L frozen-teacher blocks (instance-normed)
    lambda_cls: float = 0.5                  # frozen-teacher [CLS] cosine distillation weight
    proj_hidden: int = 512
    proj_dim: int = 256
    asym_aug: bool = True                    # photometric jitter on the student view only
    lambda_mim: float = 1.0                  # weight of the MIM term in the hybrid objective

    # ---- LoRA -------------------------------------------------------------
    lora_r: int = 16
    lora_alpha: float = 16.0
    lora_targets: Tuple[str, ...] = ("attn.q", "attn.v")
    fixed_a: bool = False                    # True -> FFA-LoRA-style frozen A (ablation only)
    use_gate: bool = True                    # per-site learnable scalar on the LoRA branch

    # ---- federation -------------------------------------------------------
    num_clients: int = 20
    participation: float = 0.4
    rounds: int = 10
    local_epochs: int = 1
    dirichlet_alpha: float = 0.1
    share_heads: bool = True                 # aggregate the MIM decoder/projector (see text)
    fedprox_mu: float = 0.0                  # > 0 -> FedProx proximal term on the shared tensors

    # ---- optimisation -----------------------------------------------------
    batch_size: int = 64
    lr: float = 1e-3
    min_lr_frac: float = 0.05
    wd: float = 0.05
    grad_clip: float = 1.0
    amp: bool = True
    warmup_rounds: int = 2
    label_smoothing: float = 0.1

    # ---- data / evaluation ------------------------------------------------
    val_n: int = 5000                        # server-only validation split (never on a client)
    val_bank_per_class: int = 50             # kNN bank for the per-round validation curve
    knn_k: int = 20
    probe_epochs: int = 40
    fewshot_frac: float = 0.1
    max_local_steps: int = 0
    select_on: str = "val_knn"               # "val_knn" | "last"

def alpha_tag(a: float) -> str:
    return "iid" if not np.isfinite(a) else f"{a:g}"

def alpha_label(a: float) -> str:
    return "IID" if not np.isfinite(a) else f"α={a:g}"

OBJ_LABEL = {"mim": "FedMIM-LoRA (no labels)", "sup": "Supervised FedLoRA (labels)",
             "sup_mim": "Supervised + MIM hybrid", "sup_prox": "Supervised FedLoRA + FedProx",
             "mim_fixA": "FedMIM-LoRA, Fixed-A"}

# ------------------------------------------------------------------ presets
if TEST_MODE:
    BASE = Config(img_size=32, patch_size=8, rounds=2, num_clients=4, participation=1.0,
                  max_local_steps=2, probe_epochs=2, val_n=200, val_bank_per_class=2, batch_size=16,
                  warmup_rounds=1, dec_dim=32, dec_heads=2, proj_hidden=32, proj_dim=16, lora_r=4,
                  lora_alpha=4.0)
    ALPHAS, SEEDS = [0.1, IID], [42]
    GEO = dict(warm_steps=2, batches=2, bs=8)
    TIME_BUDGET_H, RESERVE_MIN = 1.0, 1.0
elif PRESET == "smoke":
    BASE = Config(rounds=2, num_clients=4, participation=1.0, max_local_steps=3, probe_epochs=3,
                  val_n=1000, val_bank_per_class=10, batch_size=32, warmup_rounds=1)
    ALPHAS, SEEDS = [0.1, IID], [42]
    GEO = dict(warm_steps=5, batches=2, bs=32)
    TIME_BUDGET_H, RESERVE_MIN = 1.5, 5.0
else:                                         # "full"
    BASE = Config(rounds=10)
    ALPHAS, SEEDS = [0.05, 0.1, 0.5, IID], [42, 43, 44]
    GEO = dict(warm_steps=150, batches=6, bs=64)
    TIME_BUDGET_H, RESERVE_MIN = 11.25, 25.0

# ------------------------------------------------------------- wall-clock budget
# A study that does not finish does not exist. Every run is costed before launch and skipped if it
# will not fit. The per-round cost (training + per-round validation) starts from priors measured on
# 2xT4 in earlier runs and is re-calibrated after every completed run from its measured wall-clock.
T_START = time.time()
COST_PATH = DIRS["metrics"] / "_calibration.json"
COST = {"mim": 1.05, "sup": 1.40, "sup_mim": 2.10, "final": 3.0, "build": 0.6}   # minutes
if COST_PATH.exists():
    try:
        COST.update(json.loads(COST_PATH.read_text()))
    except Exception:
        pass

def elapsed_min():
    return (time.time() - T_START) / 60.0

def left_min():
    return TIME_BUDGET_H * 60.0 - elapsed_min() - RESERVE_MIN

def cost_min(cfg: Config) -> float:
    per = COST.get(cfg.objective, 1.4) * max(cfg.local_epochs, 1)
    if cfg.fedprox_mu > 0:
        per *= 1.03
    return per * cfg.rounds + COST["final"] + COST["build"]

def calibrate(cfg: Config, hist: Dict[str, Any]):
    """Blend the prior with the measured per-round and final-evaluation minutes of a finished run."""
    try:
        rs = hist["rounds"]
        per = float(np.mean([r["time_s"] + r.get("eval_s", 0.0) for r in rs])) / 60.0
        per /= max(cfg.local_epochs, 1) * (1.03 if cfg.fedprox_mu > 0 else 1.0)
        COST[cfg.objective] = 0.3 * COST.get(cfg.objective, per) + 0.7 * per
        if "final_eval_s" in hist:
            COST["final"] = 0.3 * COST["final"] + 0.7 * hist["final_eval_s"] / 60.0
        COST_PATH.write_text(json.dumps(COST, indent=1))
    except Exception as e:
        print(f"[calibrate] skipped: {e}")

# ------------------------------------------------------------- resume from an attached previous output
_resumed = 0
for src in glob.glob("/kaggle/input/**/fedmim_lora/metrics/*.json", recursive=True):
    dst = DIRS["metrics"] / Path(src).name
    if not dst.exists():
        shutil.copy(src, dst); _resumed += 1
for src in glob.glob("/kaggle/input/**/fedmim_lora/checkpoints/*.pth", recursive=True):
    dst = DIRS["checkpoints"] / Path(src).name
    if not dst.exists():
        shutil.copy(src, dst)
if _resumed:
    print(f"[resume] copied {_resumed} cached metric files from an attached previous run")

if torch.cuda.is_available() and not TEST_MODE:
    DEVICES = [torch.device(f"cuda:{i}") for i in range(torch.cuda.device_count())]
else:
    DEVICES = [torch.device("cpu"), torch.device("cpu")]      # two CPU "replicas" exercise the threading
set_seed(BASE.seed)
(OUT / "base_config.json").write_text(json.dumps(asdict(BASE), indent=2, default=str))

print(f"PRESET={PRESET}  MODE={MODE}  rounds={BASE.rounds}  clients={BASE.num_clients} "
      f"@ {BASE.participation:.0%}  alphas={[alpha_label(a) for a in ALPHAS]}  seeds={SEEDS}")
print(f"devices: {[str(d) for d in DEVICES]}")
print(f"wall-clock budget: {TIME_BUDGET_H:.2f} h (reserve {RESERVE_MIN:.0f} min for analysis)")
print(f"FedMIM-LoRA recipe: objective={BASE.objective} m={BASE.mask_ratio} r={BASE.lora_r} "
      f"fixed_A={BASE.fixed_a} local heads={not BASE.share_heads}")
# %% [markdown]
# ## 2. Data
#
# The 50,000 training images are split once into a 45,000-image client pool and a 5,000-image server validation split. The 10,000 test images are used only for final evaluation.
#
# For each class $c$ we draw $\mathbf{p}_c\sim\mathrm{Dir}(\alpha\mathbf{1}_K)$ over $K=20$ clients and split the class accordingly, with a soft capacity limit per client. We use $\alpha\in\{0.05, 0.1, 0.5\}$ and an exact IID split. Partitions are redrawn for each seed; within a seed all methods get the same partition and the same client sampling order.

# %%
CIFAR100_URL = "https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"

def _find_local_cifar() -> Optional[Path]:
    cands = [Path("/kaggle/input/datasets/shuvobiswas730/cifar-100-fed"),
             Path("/kaggle/input/cifar-100-fed")]
    for t in cands:
        if (t / "train").exists() and (t / "test").exists():
            return t
        if (t / "cifar-100-python" / "train").exists():
            return t / "cifar-100-python"
    base = Path("/kaggle/input")
    if base.exists():
        for p in list(base.glob("**/*cifar*"))[:200]:
            if p.is_dir():
                if (p / "train").exists() and (p / "test").exists() and (p / "meta").exists():
                    return p
                if (p / "cifar-100-python" / "train").exists():
                    return p / "cifar-100-python"
    return None

def load_cifar100():
    if TEST_MODE:                                   # synthetic stand-in with the same structure
        rng = np.random.RandomState(0)
        ytr = np.repeat(np.arange(100), 14); yte = np.repeat(np.arange(100), 4)
        proto = rng.randint(0, 255, size=(100, 32, 32, 3))
        xtr = np.clip(proto[ytr] + rng.randint(-40, 40, size=(len(ytr), 32, 32, 3)), 0, 255).astype(np.uint8)
        xte = np.clip(proto[yte] + rng.randint(-40, 40, size=(len(yte), 32, 32, 3)), 0, 255).astype(np.uint8)
        return xtr, ytr.astype(np.int64), xte, yte.astype(np.int64), [f"c{i}" for i in range(100)]
    d = _find_local_cifar()
    if d is not None:
        print(f"CIFAR-100 found locally: {d}")
    else:
        print("CIFAR-100 not mounted; downloading.")
        root = ROOT / "cifar100"; root.mkdir(parents=True, exist_ok=True)
        d = root / "cifar-100-python"
        if not (d / "train").exists():
            tgz = root / "cifar-100-python.tar.gz"
            if not tgz.exists():
                urllib.request.urlretrieve(CIFAR100_URL, filename=tgz)
            with tarfile.open(tgz) as t:
                t.extractall(path=root)

    def rd(split):
        with open(d / split, "rb") as f:
            o = pickle.load(f, encoding="latin1")
        x = o["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)      # NHWC uint8
        return np.ascontiguousarray(x), np.array(o["fine_labels"], dtype=np.int64)

    (xtr, ytr), (xte, yte) = rd("train"), rd("test")
    names = pickle.load(open(d / "meta", "rb"), encoding="latin1")["fine_label_names"]
    return xtr, ytr, xte, yte, names

XTR, YTR, XTE, YTE, CLASSES = load_cifar100()
NUM_CLASSES = len(CLASSES)
XTR_T, XTE_T = torch.from_numpy(XTR), torch.from_numpy(XTE)
YTR_T, YTE_T = torch.from_numpy(YTR), torch.from_numpy(YTE)

_rng = np.random.RandomState(12345)                  # the split is fixed across every seed and run
_perm = _rng.permutation(len(XTR))
VAL_IDX = np.sort(_perm[:BASE.val_n])
POOL_IDX = np.sort(_perm[BASE.val_n:])
TEST_IDX = np.arange(len(XTE))

def stratified_subset(idx: np.ndarray, labels: np.ndarray, per_class: int, seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    out = []
    for c in range(NUM_CLASSES):
        ic = idx[labels[idx] == c]
        if len(ic):
            out.append(rng.choice(ic, size=min(per_class, len(ic)), replace=False))
    return np.sort(np.concatenate(out))

VAL_BANK = stratified_subset(POOL_IDX, YTR, BASE.val_bank_per_class)

print(f"CIFAR-100  train={XTR.shape}  test={XTE.shape}  classes={NUM_CLASSES}")
print(f"split      client pool={len(POOL_IDX)}  server val={len(VAL_IDX)}  test={len(TEST_IDX)}  "
      f"val kNN bank={len(VAL_BANK)}")

# %%
def dirichlet_partition(labels: np.ndarray, pool: np.ndarray, num_clients: int, alpha: float,
                        num_classes: int, seed: int, min_per_client: int = 32, tries: int = 400):
    """Per-class Dirichlet label skew; alpha = inf gives an exact IID (uniform random) split."""
    rng = np.random.RandomState(seed)
    if not np.isfinite(alpha):
        perm = rng.permutation(pool)
        return [np.sort(p).astype(np.int64) for p in np.array_split(perm, num_clients)]
    N, cap = len(pool), len(pool) / num_clients
    lab_pool = labels[pool]
    best = None
    for _ in range(tries):
        parts = [[] for _ in range(num_clients)]
        for c in range(num_classes):
            idx = pool[np.where(lab_pool == c)[0]]
            if len(idx) == 0:
                continue
            rng.shuffle(idx)
            p = rng.dirichlet(np.repeat(alpha, num_clients))
            p = np.array([pi * (len(parts[i]) < cap) for i, pi in enumerate(p)])
            if p.sum() == 0:
                p = np.ones(num_clients)
            p = p / p.sum()
            cuts = (np.cumsum(p) * len(idx)).astype(int)[:-1]
            for i, chunk in enumerate(np.split(idx, cuts)):
                parts[i].extend(chunk.tolist())
        sizes = [len(x) for x in parts]
        if best is None or min(sizes) > best[0]:
            best = (min(sizes), parts)
        if min(sizes) >= min_per_client:
            break
    return [np.array(sorted(p), dtype=np.int64) for p in best[1]]

def partition_stats(parts, labels, num_classes):
    M = np.zeros((len(parts), num_classes), dtype=np.int64)
    for i, p in enumerate(parts):
        u, c = np.unique(labels[p], return_counts=True)
        M[i, u] = c
    sizes = M.sum(1)
    q = M / np.maximum(sizes[:, None], 1)
    eff = np.exp(-(q * np.log(q + 1e-12)).sum(1))            # effective number of classes exp(H)
    top5 = np.sort(q, axis=1)[:, ::-1][:, :5].sum(1)
    return M, sizes, eff, top5

def matched_test_partition(train_parts, ytr, yte, seed=0, size=1000):
    """Client i's local test set follows client i's own label marginal, so per-client accuracy
    measures the metric that client actually cares about."""
    rng = np.random.RandomState(seed)
    by_class = {c: np.where(yte == c)[0] for c in range(NUM_CLASSES)}
    out = []
    for p in train_parts:
        u, c = np.unique(ytr[p], return_counts=True)
        want = np.maximum((c / c.sum() * size).astype(int), 0)
        sel = [rng.choice(by_class[k], size=min(n, len(by_class[k])), replace=False)
               for k, n in zip(u, want) if n > 0 and len(by_class[k])]
        out.append(np.sort(np.concatenate(sel)) if sel else np.array([], dtype=np.int64))
    return out

PARTITIONS: Dict[Tuple[float, int], List[np.ndarray]] = {}
TEST_PARTS: Dict[Tuple[float, int], List[np.ndarray]] = {}

def parts_for(alpha: float, seed: int, num_clients: Optional[int] = None):
    key = (alpha, seed)
    if key not in PARTITIONS:
        PARTITIONS[key] = dirichlet_partition(YTR, POOL_IDX, num_clients or BASE.num_clients, alpha,
                                              NUM_CLASSES, seed)
        TEST_PARTS[key] = matched_test_partition(PARTITIONS[key], YTR, YTE, seed=seed)
    return PARTITIONS[key]

_rows = []
for a in ALPHAS:
    for sd in SEEDS:
        M, sizes, eff, top5 = partition_stats(parts_for(a, sd), YTR, NUM_CLASSES)
        _rows.append(dict(alpha=alpha_label(a), seed=sd, n_min=int(sizes.min()),
                          n_median=int(np.median(sizes)), n_max=int(sizes.max()),
                          eff_classes_mean=float(eff.mean()), eff_classes_min=float(eff.min()),
                          top5_mass_mean=float(top5.mean())))
PART_STATS_ALL = pd.DataFrame(_rows)
PART_STATS = save_table(
    PART_STATS_ALL.drop(columns="seed").groupby("alpha", sort=False).mean().reset_index(),
    "table1_partitions",
    f"Table 1. Client partitions of the {len(POOL_IDX):,}-image pool over {BASE.num_clients} clients, "
    f"averaged over seeds {SEEDS}. 'Effective classes' is exp(H) of a client's label marginal "
    "(100 = uniform); 'top-5 mass' is the fraction of a client's data in its five dominant classes.",
    "partitions")
display(PART_STATS.round(2))
EFF_CLASSES = {a: float(PART_STATS_ALL[PART_STATS_ALL.alpha == alpha_label(a)].eff_classes_mean.mean())
               for a in ALPHAS}

# %%
# ---- Figure 1: what the partitions look like -----------------------------------------------
try:
    fig, axes = plt.subplots(1, len(ALPHAS), figsize=(3.6 * len(ALPHAS), 3.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    for ax, a in zip(axes, ALPHAS):
        M, sizes, eff, top5 = partition_stats(parts_for(a, SEEDS[0]), YTR, NUM_CLASSES)
        q = M / np.maximum(sizes[:, None], 1)
        im = ax.imshow(q, aspect="auto", cmap="magma", vmin=0, vmax=max(0.05, float(q.max()) * 0.9),
                       interpolation="nearest")
        ax.set_title(f"{alpha_label(a)}  (eff. classes {eff.mean():.1f})", fontsize=10.5)
        ax.set_xlabel("class"); ax.set_ylabel("client" if a == ALPHAS[0] else "")
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.01, label="P(class | client)")
    savefig(fig, "fig1_partitions",
            "Figure 1. Per-client label marginals of the client pool under Dirichlet label skew "
            f"(seed {SEEDS[0]}). Titles give the mean effective number of classes per client, exp(H). "
            "At α=0.05 a typical client holds a handful of classes; the IID split is uniform.")
except Exception as e:
    print(f"[fig1 skipped] {type(e).__name__}: {e}")

# %%
IMNET_MEAN, IMNET_STD = 0.5, 0.5      # the in21k ViT preprocessing uses mean = std = 0.5

class GPUBatcher:
    """uint8 images stay on the CPU; resize / crop / flip / jitter / normalise all happen on the GPU."""

    def __init__(self, images_u8: torch.Tensor, labels: torch.Tensor, img_size: int):
        self.x, self.y, self.S = images_u8, labels, img_size

    def _fetch(self, idx, device):
        if isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        xb = self.x[idx].to(device, non_blocking=True)
        return xb.permute(0, 3, 1, 2).float().div_(255.0)

    def labels_of(self, idx, device):
        if isinstance(idx, np.ndarray):
            idx = torch.from_numpy(idx)
        return self.y[idx].to(device, non_blocking=True)

    def geo_batch(self, idx, device, gen, scale=(0.35, 1.0), ratio=(3 / 4, 4 / 3)):
        """Random-resized-crop + flip via one affine grid; returned in [0,1], un-normalised."""
        x = self._fetch(idx, device)
        B = x.shape[0]
        u = lambda a, b: torch.rand(B, generator=gen) * (b - a) + a
        area = u(scale[0], scale[1])
        ar = torch.exp(u(math.log(ratio[0]), math.log(ratio[1])))
        w = torch.sqrt(area * ar).clamp(max=1.0)
        h = torch.sqrt(area / ar).clamp(max=1.0)
        cx = (torch.rand(B, generator=gen) * 2 - 1) * (1 - w)
        cy = (torch.rand(B, generator=gen) * 2 - 1) * (1 - h)
        flip = (torch.rand(B, generator=gen) < 0.5).float() * (-2.0) + 1.0
        th = torch.zeros(B, 2, 3)
        th[:, 0, 0] = w * flip; th[:, 0, 2] = cx
        th[:, 1, 1] = h;        th[:, 1, 2] = cy
        grid = F.affine_grid(th.to(device), (B, 3, self.S, self.S), align_corners=False)
        return F.grid_sample(x, grid, mode="bilinear", padding_mode="reflection", align_corners=False)

    @staticmethod
    def photometric(x, gen, p_jitter=0.8, p_gray=0.2):
        B, dev = x.shape[0], x.device
        r = lambda a, b: (torch.rand(B, 1, 1, 1, generator=gen) * (b - a) + a).to(dev)
        m = (torch.rand(B, 1, 1, 1, generator=gen) < p_jitter).float().to(dev)
        x = x * (1 + m * (r(0.6, 1.4) - 1))
        mu = x.mean(dim=(1, 2, 3), keepdim=True)
        x = mu + (x - mu) * (1 + m * (r(0.6, 1.4) - 1))
        g = (x * torch.tensor([0.299, 0.587, 0.114], device=dev).view(1, 3, 1, 1)).sum(1, keepdim=True)
        x = g + (x - g) * (1 + m * (r(0.6, 1.4) - 1))
        gm = (torch.rand(B, 1, 1, 1, generator=gen) < p_gray).float().to(dev)
        x = gm * g.expand_as(x) + (1 - gm) * x
        return x.clamp(0, 1)

    @staticmethod
    def norm(x):
        return (x - IMNET_MEAN) / IMNET_STD

    def train_views(self, idx, device, gen, jitter: bool = True):
        """(student view, clean teacher view) with identical geometry."""
        x = self.geo_batch(idx, device, gen)
        xs = self.photometric(x, gen) if jitter else x
        return self.norm(xs), self.norm(x)

    def eval_batch(self, idx, device):
        x = self._fetch(idx, device)
        x = F.interpolate(x, size=(self.S, self.S), mode="bilinear", align_corners=False)
        return self.norm(x)

BATCH_TR = GPUBatcher(XTR_T, YTR_T, BASE.img_size)
BATCH_TE = GPUBatcher(XTE_T, YTE_T, BASE.img_size)

_g = torch.Generator().manual_seed(0)
_s, _t = BATCH_TR.train_views(np.arange(4), DEVICES[0], _g)
print(f"student view {tuple(_s.shape)} range=[{_s.min():.2f},{_s.max():.2f}] | "
      f"views share geometry, differ photometrically (mean |Δ|={float((_s-_t).abs().mean()):.4f})")
del _s, _t; free_mem()
# %% [markdown]
# ## 3. Model
#
# ### 3.1 ViT-B/16
#
# The ViT forward pass is implemented here (pre-LN blocks, exact GELU, LayerNorm eps 1e-12, SDPA attention) and loads `google/vit-base-patch16-224-in21k` directly from `model.safetensors`. Both the old and the new Hugging Face parameter names are mapped, so LoRA injection does not depend on the installed `transformers` version. Section 3.4 checks the port against `ViTModel`.
#
# ### 3.2 LoRA
#
# $W_0x + g\cdot\tfrac{\alpha}{r}BAx$ on the query and value projections of all 12 blocks (24 sites). $B=0$ at initialisation, so round 0 is the pretrained model. $A$ is a seeded semi-orthogonal matrix, identical on every client.
#
# ### 3.3 MIM objective
#
# MAE-style asymmetric encoder: the student encodes only the visible tokens, and a one-block decoder re-inserts mask tokens and regresses the per-token normalised average of the frozen backbone's last two blocks on the clean image. A projector/predictor pair aligns the student [CLS] with a fixed random projection of the teacher [CLS] (cosine loss, weight 0.5). The target network is frozen, so there is no collapse mode.

# %%
class Attn(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v = nn.Linear(dim, dim), nn.Linear(dim, dim), nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape
        sh = lambda t: t.view(B, N, self.h, C // self.h).transpose(1, 2)
        o = F.scaled_dot_product_attention(sh(self.q(x)), sh(self.k(x)), sh(self.v(x)))
        return self.proj(o.transpose(1, 2).reshape(B, N, C))

class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_dim: int, eps: float):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(dim, eps=eps), nn.LayerNorm(dim, eps=eps)
        self.attn = Attn(dim, heads)
        self.fc1, self.fc2 = nn.Linear(dim, mlp_dim), nn.Linear(mlp_dim, dim)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.fc2(F.gelu(self.fc1(self.ln2(x))))

class ViTBackbone(nn.Module):
    def __init__(self, img=224, patch=16, dim=768, depth=12, heads=12, mlp_dim=3072, eps=1e-12):
        super().__init__()
        self.patch_embed = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, (img // patch) ** 2 + 1, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_dim, eps) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim, eps=eps)
        self.dim, self.depth, self.grad_ckpt = dim, depth, False

    def embed(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2) + self.pos_embed[:, 1:]

    def cls(self, B):
        return (self.cls_token + self.pos_embed[:, :1]).expand(B, -1, -1)

    def run_blocks(self, x, keep_last: int = 0):
        hidden = []
        for i, blk in enumerate(self.blocks):
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(blk, x, use_reentrant=False)
            else:
                x = blk(x)
            if keep_last and i >= self.depth - keep_last:
                hidden.append(x)
        return x, hidden

    def forward(self, x):                                   # plain forward, used for the parity check
        t = self.embed(x)
        t = torch.cat([self.cls(t.shape[0]).to(t.dtype), t], dim=1)
        return self.norm(self.run_blocks(t)[0])

VIT_ARCH = (dict(dim=64, depth=2, heads=4, mlp_dim=128) if TEST_MODE
            else dict(dim=768, depth=12, heads=12, mlp_dim=3072))

# ---------------------------------------------------------------- checkpoint loading + key mapping
_MAP_RULES = [
    (r"embeddings\.cls_token", "cls_token"),
    (r"embeddings\.position_embeddings", "pos_embed"),
    (r"embeddings\.patch_embeddings\.projection\.(weight|bias)", r"patch_embed.\1"),
    # transformers <= 4.x layout (and the layout stored in the hub checkpoint)
    (r"encoder\.layer\.(\d+)\.attention\.attention\.query\.(weight|bias)", r"blocks.\1.attn.q.\2"),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.key\.(weight|bias)", r"blocks.\1.attn.k.\2"),
    (r"encoder\.layer\.(\d+)\.attention\.attention\.value\.(weight|bias)", r"blocks.\1.attn.v.\2"),
    (r"encoder\.layer\.(\d+)\.attention\.output\.dense\.(weight|bias)", r"blocks.\1.attn.proj.\2"),
    (r"encoder\.layer\.(\d+)\.intermediate\.dense\.(weight|bias)", r"blocks.\1.fc1.\2"),
    (r"encoder\.layer\.(\d+)\.output\.dense\.(weight|bias)", r"blocks.\1.fc2.\2"),
    (r"encoder\.layer\.(\d+)\.layernorm_before\.(weight|bias)", r"blocks.\1.ln1.\2"),
    (r"encoder\.layer\.(\d+)\.layernorm_after\.(weight|bias)", r"blocks.\1.ln2.\2"),
    # transformers >= 5 layout
    (r"layers\.(\d+)\.attention\.q_proj\.(weight|bias)", r"blocks.\1.attn.q.\2"),
    (r"layers\.(\d+)\.attention\.k_proj\.(weight|bias)", r"blocks.\1.attn.k.\2"),
    (r"layers\.(\d+)\.attention\.v_proj\.(weight|bias)", r"blocks.\1.attn.v.\2"),
    (r"layers\.(\d+)\.attention\.o_proj\.(weight|bias)", r"blocks.\1.attn.proj.\2"),
    (r"layers\.(\d+)\.mlp\.fc1\.(weight|bias)", r"blocks.\1.fc1.\2"),
    (r"layers\.(\d+)\.mlp\.fc2\.(weight|bias)", r"blocks.\1.fc2.\2"),
    (r"layers\.(\d+)\.layernorm_before\.(weight|bias)", r"blocks.\1.ln1.\2"),
    (r"layers\.(\d+)\.layernorm_after\.(weight|bias)", r"blocks.\1.ln2.\2"),
    (r"layernorm\.(weight|bias)", r"norm.\1"),
]

def map_vit_keys(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    out = {}
    for k, v in sd.items():
        k2 = re.sub(r"^(vit\.|model\.)", "", k)
        for pat, rep in _MAP_RULES:
            if re.fullmatch(pat, k2):
                out[re.sub(pat, rep, k2)] = v.float()
                break
    return out

def fetch_backbone_state(model_name: str) -> Tuple[Dict[str, torch.Tensor], str]:
    errs = []
    try:
        from huggingface_hub import hf_hub_download
        from safetensors.torch import load_file
        return load_file(hf_hub_download(model_name, "model.safetensors")), "hub:model.safetensors"
    except Exception as e:
        errs.append(f"safetensors: {type(e).__name__}: {e}")
    try:
        from huggingface_hub import hf_hub_download
        p = hf_hub_download(model_name, "pytorch_model.bin")
        try:
            return torch.load(p, map_location="cpu", weights_only=True), "hub:pytorch_model.bin"
        except TypeError:
            return torch.load(p, map_location="cpu"), "hub:pytorch_model.bin"
    except Exception as e:
        errs.append(f"bin: {type(e).__name__}: {e}")
    try:
        from transformers import ViTModel
        return ViTModel.from_pretrained(model_name).state_dict(), "transformers.ViTModel"
    except Exception as e:
        errs.append(f"transformers: {type(e).__name__}: {e}")
    raise RuntimeError("could not load the backbone weights (is Internet enabled?):\n  " + "\n  ".join(errs))

_BACKBONE_SD: Dict[str, torch.Tensor] = {}

def backbone_state() -> Dict[str, torch.Tensor]:
    global _BACKBONE_SD
    if TEST_MODE:
        return {}
    if not _BACKBONE_SD:
        t0 = time.time()
        raw, src = fetch_backbone_state(BASE.model_name)
        _BACKBONE_SD = map_vit_keys(raw)
        need = set(ViTBackbone(BASE.img_size, BASE.patch_size, **VIT_ARCH).state_dict().keys())
        missing = sorted(need - set(_BACKBONE_SD))
        if missing:
            raise RuntimeError(f"backbone key mapping incomplete: {len(missing)} missing, e.g. {missing[:5]}")
        print(f"[backbone] {BASE.model_name} loaded from {src} in {time.time()-t0:.1f}s "
              f"({len(_BACKBONE_SD)} tensors mapped, 0 missing)")
    return _BACKBONE_SD

# %%
def seeded_semi_orthogonal(r: int, k: int, seed: int) -> torch.Tensor:
    """A in R^{r x k} with A A^T = I_r, deterministic given `seed`."""
    g = torch.Generator().manual_seed(int(seed))
    q, _ = torch.linalg.qr(torch.randn(k, r, generator=g, dtype=torch.float32))
    return q.t().contiguous()

class LoRALinear(nn.Module):
    """Frozen linear layer + gated low-rank branch. mode='off' disables the branch (frozen teacher)."""
    def __init__(self, base: nn.Module, r: int, alpha: float, fixed_a: bool, seed: int, use_gate=True):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        in_f, out_f = int(base.in_features), int(base.out_features)
        self.r, self.scaling, self.fixed_a, self.mode = r, alpha / r, fixed_a, "student"
        A = seeded_semi_orthogonal(r, in_f, seed)
        if fixed_a:
            self.register_buffer("lora_A", A)
        else:
            self.lora_A = nn.Parameter(A)
        self.lora_B = nn.Parameter(torch.zeros(out_f, r))
        self.lora_g = nn.Parameter(torch.ones(1)) if use_gate else None

    @property
    def in_features(self):
        return self.base.in_features

    @property
    def out_features(self):
        return self.base.out_features

    def forward(self, x):
        y = self.base(x)
        if self.mode == "off":
            return y
        z = F.linear(F.linear(x, self.lora_A.to(x.dtype)), self.lora_B.to(x.dtype)) * self.scaling
        return y + (z if self.lora_g is None else z * self.lora_g.to(z.dtype))

def inject_lora(root: nn.Module, targets, r, alpha, fixed_a, seed, use_gate=True) -> List[str]:
    replaced, named = [], dict(root.named_modules())
    for name, mod in list(named.items()):
        if not isinstance(mod, nn.Linear) or not any(name.endswith(t) for t in targets):
            continue
        parent_name, leaf = name.rsplit(".", 1) if "." in name else ("", name)
        parent = named[parent_name] if parent_name else root
        lseed = (seed * 1000003 + int(hashlib.md5(name.encode()).hexdigest()[:8], 16)) % (2 ** 31 - 1)
        setattr(parent, leaf, LoRALinear(mod, r, alpha, fixed_a, lseed, use_gate))
        replaced.append(name)
    return replaced

@contextlib.contextmanager
def lora_mode(root: nn.Module, mode: str):
    prev = [(m, m.mode) for m in root.modules() if isinstance(m, LoRALinear)]
    for m, _ in prev:
        m.mode = mode
    try:
        yield
    finally:
        for m, p in prev:
            m.mode = p

def quantize_backbone_nf4(model: nn.Module, device):
    """Swap every frozen nn.Linear inside the ViT blocks for a bitsandbytes NF4 Linear4bit."""
    import bitsandbytes as bnb
    def q4(lin: nn.Linear):
        new = bnb.nn.Linear4bit(lin.in_features, lin.out_features, bias=lin.bias is not None,
                                compute_dtype=torch.float16, quant_type="nf4")
        new.weight = bnb.nn.Params4bit(lin.weight.data.detach().cpu().contiguous(), requires_grad=False,
                                       quant_type="nf4")
        if lin.bias is not None:
            new.bias = nn.Parameter(lin.bias.data.detach().cpu().clone(), requires_grad=False)
        return new.to(device)
    for blk in model.vit.blocks:
        for parent, leafs in [(blk.attn, ["q", "k", "v", "proj"]), (blk, ["fc1", "fc2"])]:
            for leaf in leafs:
                mod = getattr(parent, leaf)
                if isinstance(mod, LoRALinear):
                    mod.base = q4(mod.base)
                elif isinstance(mod, nn.Linear):
                    setattr(parent, leaf, q4(mod))

# %%
def get_2d_sincos_pos_embed(dim: int, grid: int) -> torch.Tensor:
    def emb_1d(d, pos):
        omega = 1.0 / (10000 ** (np.arange(d // 2, dtype=np.float64) / (d / 2.0)))
        out = np.einsum("m,d->md", pos.reshape(-1), omega)
        return np.concatenate([np.sin(out), np.cos(out)], axis=1)
    gh, gw = np.meshgrid(np.arange(grid, dtype=np.float32), np.arange(grid, dtype=np.float32), indexing="ij")
    pe = np.concatenate([emb_1d(dim // 2, gw), emb_1d(dim // 2, gh)], axis=1)
    pe = np.concatenate([np.zeros([1, dim]), pe], axis=0)
    return torch.from_numpy(pe).float().unsqueeze(0)

class DecBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(nn.Linear(dim, h), nn.GELU(), nn.Linear(h, dim))

    def forward(self, x):
        y = self.n1(x)
        x = x + self.attn(y, y, y, need_weights=False)[0]
        return x + self.mlp(self.n2(x))

class MaskedDecoder(nn.Module):
    """Asymmetric MAE decoder: re-insert mask tokens, add 2-D sin-cos positions, predict every patch."""
    def __init__(self, enc_dim, dim, depth, heads, num_patches, tgt_dim):
        super().__init__()
        self.proj = nn.Linear(enc_dim, dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.register_buffer("pos", get_2d_sincos_pos_embed(dim, int(round(num_patches ** 0.5))))
        self.blocks = nn.ModuleList([DecBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.pred = nn.Linear(dim, tgt_dim)
        nn.init.normal_(self.mask_token, std=0.02)

    def forward(self, h, ids_restore):
        x = self.proj(h)
        B, N, D = ids_restore.shape[0], ids_restore.shape[1], x.shape[-1]
        mt = self.mask_token.to(x.dtype).expand(B, N + 1 - x.shape[1], -1)
        x_ = torch.gather(torch.cat([x[:, 1:, :], mt], dim=1), 1, ids_restore.unsqueeze(-1).expand(-1, -1, D))
        x = torch.cat([x[:, :1, :], x_], dim=1) + self.pos.to(x.dtype)
        for blk in self.blocks:
            x = blk(x)
        return self.pred(self.norm(x))[:, 1:, :]

def mlp_head(din, dh, dout):
    """LayerNorm, never BatchNorm: BN running statistics are buffers that aggregation never reconciles."""
    return nn.Sequential(nn.Linear(din, dh), nn.LayerNorm(dh), nn.GELU(), nn.Linear(dh, dout))

class FedMIMViT(nn.Module):
    """Frozen ViT-B/16 + gated LoRA + (MIM decoder & CLS projector) and/or (linear classifier)."""

    def __init__(self, cfg: Config, num_classes: int = 100):
        super().__init__()
        self.cfg = cfg
        self.vit = ViTBackbone(cfg.img_size, cfg.patch_size, **VIT_ARCH)
        sd = backbone_state()
        if sd:
            self.vit.load_state_dict(sd, strict=True)
        for p in self.vit.parameters():
            p.requires_grad_(False)
        self.vit.grad_ckpt = cfg.grad_ckpt
        self.enc_dim = self.vit.dim
        self.num_patches = (cfg.img_size // cfg.patch_size) ** 2
        self.lora_sites = inject_lora(self.vit.blocks, tuple(cfg.lora_targets), cfg.lora_r, cfg.lora_alpha,
                                      cfg.fixed_a, cfg.seed, cfg.use_gate)
        self.has_mim = cfg.objective in ("mim", "sup_mim")
        self.has_sup = cfg.objective in ("sup", "sup_mim")
        self.decoder = self.proj = self.pred_head = self.tgt_proj = self.head = None
        if self.has_mim:
            self.decoder = MaskedDecoder(self.enc_dim, cfg.dec_dim, cfg.dec_depth, cfg.dec_heads,
                                         self.num_patches, self.enc_dim)
            if cfg.lambda_cls > 0:
                self.proj = mlp_head(self.enc_dim, cfg.proj_hidden, cfg.proj_dim)
                self.pred_head = mlp_head(cfg.proj_dim, cfg.proj_hidden, cfg.proj_dim)
                self.tgt_proj = nn.Linear(self.enc_dim, cfg.proj_dim, bias=False)
                with torch.no_grad():               # fixed, shared, untrained projection of the teacher CLS
                    g = torch.Generator().manual_seed(cfg.seed + 99991)
                    self.tgt_proj.weight.copy_(torch.randn(cfg.proj_dim, self.enc_dim, generator=g)
                                               / math.sqrt(self.enc_dim))
                for p in self.tgt_proj.parameters():
                    p.requires_grad_(False)
        if self.has_sup:
            self.head = nn.Linear(self.enc_dim, num_classes)
            nn.init.trunc_normal_(self.head.weight, std=0.01); nn.init.zeros_(self.head.bias)

    # ------------------------------------------------------------- encoder
    @staticmethod
    def random_masking(x, mask_ratio, gen):
        B, N, D = x.shape
        keep = max(1, int(round(N * (1.0 - mask_ratio))))
        noise = torch.rand(B, N, generator=gen).to(x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)
        xk = torch.gather(x, 1, ids_shuffle[:, :keep].unsqueeze(-1).expand(-1, -1, D))
        base = torch.cat([torch.zeros(B, keep, device=x.device), torch.ones(B, N - keep, device=x.device)], 1)
        return xk, torch.gather(base, 1, ids_restore), ids_restore

    def forward_encoder(self, x, mask_ratio: float = 0.0, gen=None, keep_last: int = 0):
        t = self.vit.embed(x)
        mask = ids_restore = None
        if mask_ratio and mask_ratio > 0:
            t, mask, ids_restore = self.random_masking(t, mask_ratio, gen)
        t = torch.cat([self.vit.cls(t.shape[0]).to(t.dtype), t], dim=1)
        last, hidden = self.vit.run_blocks(t, keep_last=keep_last)
        return self.vit.norm(last), mask, ids_restore, hidden

    # ------------------------------------------------------------- MIM loss
    @torch.no_grad()
    def teacher_targets(self, x_clean):
        cfg = self.cfg
        with lora_mode(self, "off"):
            h_t, _, _, hid = self.forward_encoder(x_clean, 0.0, None, keep_last=cfg.tgt_layers)
        tgt = torch.stack([F.instance_norm(z.transpose(1, 2).float()).transpose(1, 2) for z in hid], 0).mean(0)
        cls_t = self.tgt_proj(h_t[:, 0].float()) if self.tgt_proj is not None else None
        return tgt[:, 1:, :], cls_t

    def loss_mim(self, xs, xt, gen):
        cfg = self.cfg
        tgt, cls_t = self.teacher_targets(xt)
        if cfg.norm_target:
            tgt = (tgt - tgt.mean(-1, keepdim=True)) / (tgt.var(-1, keepdim=True) + 1e-6).sqrt()
        h, mask, ids_restore, _ = self.forward_encoder(xs, cfg.mask_ratio, gen)
        if mask is None:                                    # m = 0: regress every token (cost-table row)
            B_, N_ = h.shape[0], h.shape[1] - 1
            ids_restore = torch.arange(N_, device=h.device).unsqueeze(0).expand(B_, -1)
            mask = torch.ones(B_, N_, device=h.device)
        pred = self.decoder(h, ids_restore)
        l_patch = (((pred.float() - tgt) ** 2).mean(-1) * mask).sum() / mask.sum().clamp(min=1)
        parts = {"l_patch": float(l_patch.detach())}
        loss = l_patch
        if cls_t is not None and cfg.lambda_cls > 0:
            z = self.pred_head(self.proj(h[:, 0]))
            l_cls = (2.0 - 2.0 * F.cosine_similarity(z.float(), cls_t.float(), dim=-1)).mean()
            loss = loss + cfg.lambda_cls * l_cls
            parts["l_cls"] = float(l_cls.detach())
        return loss, parts

    # ------------------------------------------------------------- supervised / read-out
    def logits(self, x):
        h, _, _, _ = self.forward_encoder(x, 0.0, None)
        return self.head(h[:, 1:, :].mean(1))

    @torch.no_grad()
    def features_logits(self, x, want_logits: bool = False):
        """Evaluation read-out: [CLS] || mean-patch of the last layer (identical for every method)."""
        h, _, _, _ = self.forward_encoder(x, 0.0, None)
        f = torch.cat([h[:, 0], h[:, 1:].mean(1)], dim=-1)
        lg = self.head(h[:, 1:, :].mean(1)) if (want_logits and self.head is not None) else None
        return f, lg

# %%
def shared_keys(model: nn.Module, cfg: Config) -> List[str]:
    """Tensors that are communicated and aggregated (LoRA A/B, gates, and the supervised head)."""
    ks = []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if (not cfg.share_heads) and n.split(".")[0] in ("decoder", "proj", "pred_head"):
            continue
        ks.append(n)
    return sorted(ks)

def get_state(model, keys) -> Dict[str, torch.Tensor]:
    P = dict(model.named_parameters())
    return {k: P[k].detach().float().cpu().clone() for k in keys}

@torch.no_grad()
def set_state(model, state: Dict[str, torch.Tensor]):
    P = dict(model.named_parameters())
    for k, v in state.items():
        if k in P:
            P[k].copy_(v.to(P[k].device, P[k].dtype))

def state_mb(state, dtype_bytes=2) -> float:
    """Uplink payload in MB assuming fp16 transmission."""
    return sum(v.numel() for v in state.values()) * dtype_bytes / 2 ** 20

MODEL_LOCK = threading.Lock()

def build_model(cfg: Config, device) -> FedMIMViT:
    with MODEL_LOCK:
        m = FedMIMViT(cfg, NUM_CLASSES)
    m = m.to(device)
    if cfg.use_4bit:
        quantize_backbone_nf4(m, device)
    return m
# %% [markdown]
# ### 3.4 Checks before training
#
# 1. The ViT port reproduces Hugging Face `ViTModel` on the same weights.
# 2. With $B=0$ the adapted model equals the frozen backbone.
# 3. Two independently built clients generate the same $A$.
# 4. Every objective returns a finite loss and gradient.

# %%
_t0 = time.time()
_m = build_model(BASE, DEVICES[0])
print(f"built in {time.time()-_t0:.1f}s | LoRA at {len(_m.lora_sites)} projections "
      f"(r={BASE.lora_r}, fixed_A={BASE.fixed_a}, gate={BASE.use_gate})")

# ---- 1. parity with Hugging Face ------------------------------------------------------------
HF_PARITY = None
try:
    from transformers import ViTModel, ViTConfig
    if TEST_MODE:
        _hf = ViTModel(ViTConfig(hidden_size=VIT_ARCH["dim"], num_hidden_layers=VIT_ARCH["depth"],
                                 num_attention_heads=VIT_ARCH["heads"], intermediate_size=VIT_ARCH["mlp_dim"],
                                 image_size=BASE.img_size, patch_size=BASE.patch_size), add_pooling_layer=False)
        _ref = ViTBackbone(BASE.img_size, BASE.patch_size, **VIT_ARCH)
        _ref.load_state_dict(map_vit_keys(_hf.state_dict()), strict=True)
    else:
        _hf = ViTModel.from_pretrained(BASE.model_name, add_pooling_layer=False)
        _ref = _m.vit
    _dev = DEVICES[0]
    _hf = _hf.to(_dev).eval()
    _x = torch.randn(4, 3, BASE.img_size, BASE.img_size, generator=torch.Generator().manual_seed(1)).to(_dev)
    with torch.no_grad(), lora_mode(_m, "off"):
        _a = _hf(pixel_values=_x).last_hidden_state.float()
        _b = _ref.eval()(_x).float()
    HF_PARITY = float((_a - _b).abs().max())
    print(f"[parity] max |HF ViTModel - this port| = {HF_PARITY:.2e}  (fp32, 4 random images)")
    del _hf, _a, _b, _x
    if HF_PARITY > 5e-3:
        raise RuntimeError(f"ViT port disagrees with Hugging Face by {HF_PARITY:.2e}; refusing to train.")
except RuntimeError:
    raise
except Exception as e:
    print(f"[parity] Hugging Face reference unavailable ({type(e).__name__}: {e}); relying on the "
          "zero-shot-floor cross-check in Sec. 5 instead.")
free_mem()

# ---- parameter / payload accounting -----------------------------------------------------------
def count_params(model):
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))

TOTAL_PARAMS, TRAIN_PARAMS = count_params(_m)
KEYS0 = shared_keys(_m, BASE)
UPLINK_MB = state_mb(get_state(_m, KEYS0))
VIT_FULL_MB = sum(p.numel() for n, p in _m.vit.named_parameters() if "lora_" not in n) * 2 / 2 ** 20
LORA_P = sum(p.numel() for n, p in _m.named_parameters() if "lora_" in n and p.requires_grad)
DEC_P = sum(p.numel() for n, p in _m.named_parameters() if n.split(".")[0] in ("decoder", "proj", "pred_head")
            and p.requires_grad)
print(f"total params {TOTAL_PARAMS/1e6:.2f} M | trainable {TRAIN_PARAMS/1e6:.3f} M "
      f"({100*TRAIN_PARAMS/TOTAL_PARAMS:.2f} %)")
print(f"  LoRA (A,B,gates)               : {LORA_P/1e3:8.1f} K -> {LORA_P*2/2**20:.3f} MB")
print(f"  uplink per client per round    : {UPLINK_MB:.3f} MB (fp16, everything aggregated)")
print(f"  MIM decoder + projector        : {DEC_P/1e3:8.1f} K -> {DEC_P*2/2**20:.3f} MB "
      f"({'aggregated' if BASE.share_heads else 'kept local'})")
LORA_ONLY_MB = LORA_P * 2 / 2 ** 20
print(f"  full fine-tuning uplink        : {VIT_FULL_MB:.1f} MB -> compression x{VIT_FULL_MB/max(UPLINK_MB,1e-9):.0f}")

# ---- 2. identity at initialisation ---------------------------------------------------------------
_g = torch.Generator().manual_seed(0)
_xs, _xt = BATCH_TR.train_views(np.arange(8), DEVICES[0], _g)
with torch.no_grad():
    _on, _ = _m.features_logits(_xt)
    with lora_mode(_m, "off"):
        _off, _ = _m.features_logits(_xt)
INIT_GAP = float((_on - _off).abs().max())
print(f"[invariant] max |f_adapted - f_frozen| at init = {INIT_GAP:.2e}  (expect 0)")
assert INIT_GAP < 1e-5

# ---- 3. shared A ----------------------------------------------------------------------------
_m2 = build_model(dc_replace(BASE, name="dup"), DEVICES[0])
_dA = max(float((a.lora_A - b.lora_A).abs().max())
          for a, b in zip([x for x in _m.modules() if isinstance(x, LoRALinear)],
                          [x for x in _m2.modules() if isinstance(x, LoRALinear)]))
print(f"[invariant] max |A_client1 - A_client2| = {_dA:.2e}  (expect 0)")
assert _dA == 0.0
del _m2; free_mem()

# ---- 4. every objective runs ------------------------------------------------------------------
for _obj in ["mim", "sup", "sup_mim"]:
    _mm = _m if _obj == "mim" else build_model(dc_replace(BASE, objective=_obj), DEVICES[0])
    _mm.train()
    with amp_ctx(DEVICES[0], BASE.amp):
        _loss = torch.zeros((), device=DEVICES[0])
        if _mm.has_sup:
            _loss = _loss + F.cross_entropy(_mm.logits(_xs), BATCH_TR.labels_of(np.arange(8), DEVICES[0]))
        if _mm.has_mim:
            _l, _parts = _mm.loss_mim(_xs, _xt, _g)
            _loss = _loss + _l
    _loss.backward()
    _gn = sum(float(p.grad.norm()) for n, p in _mm.named_parameters() if "lora_B" in n and p.grad is not None)
    print(f"[{_obj:>7}] loss={float(_loss):.4f}  ||grad LoRA-B||={_gn:.3e}  finite={bool(torch.isfinite(_loss))}")
    assert torch.isfinite(_loss)
    if _mm is not _m:
        del _mm
del _m, _xs, _xt, _on, _off, _loss
free_mem()

# %% [markdown]
# ## 4. Federated training
#
# FedAvg over the communicated tensors, weighted by client sample counts. One model replica per GPU; clients are dispatched to the replicas by a thread pool (8 clients per round, 4 per T4).
#
# Logged every round, on the LoRA-$B$ coordinates:
#
# * $\hat\kappa_t^2=\sum_k w_k\|\Delta_k-\bar\Delta\|^2/\|\bar\Delta\|^2$, the client drift of the round;
# * mean pairwise cosine between client updates and $\|\bar\Delta\|$;
# * the aggregation error of averaging $A$ and $B$ separately, $\varepsilon_t=\|\bar B\bar A-\overline{BA}\|_F/\|\overline{BA}\|_F$.
#
# FedProx adds $\tfrac{\mu}{2}\|w-w_{global}\|^2$ over the communicated tensors as a gradient term.

# %%
def flat_delta(state, ref, prefix="lora_B"):
    v = [(state[k] - ref[k]).reshape(-1) for k in sorted(ref) if prefix in k]
    return torch.cat(v) if v else torch.zeros(1)

def dissimilarity_metrics(states, ref, weights):
    D = torch.stack([flat_delta(s, ref) for s in states]).double()
    w = torch.tensor(weights, dtype=D.dtype); w = w / w.sum()
    mean = (w[:, None] * D).sum(0)
    num = (w * ((D - mean) ** 2).sum(1)).sum()
    den = (mean ** 2).sum().clamp(min=1e-18)
    Dn = F.normalize(D, dim=1)
    C_ = Dn @ Dn.t()
    K = len(states)
    cos = (C_.sum() - C_.diag().sum()) / max(K * (K - 1), 1)
    return dict(kappa2=float(num / den), kappa=float((num / den).sqrt()), pairwise_cos=float(cos),
                update_norm=float(mean.norm()), client_norm_mean=float(D.norm(dim=1).mean()))

def abo_error(states, weights):
    """Relative || mean(B) mean(A) - mean(BA) ||_F averaged over LoRA sites; exactly 0 when A is shared."""
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    errs = []
    for kB in sorted(states[0]):
        if "lora_B" not in kB:
            continue
        kA = kB.replace("lora_B", "lora_A")
        if kA not in states[0]:
            return 0.0
        Bm = sum(wi * s[kB].double() for wi, s in zip(w, states))
        Am = sum(wi * s[kA].double() for wi, s in zip(w, states))
        true = sum(wi * (s[kB].double() @ s[kA].double()) for wi, s in zip(w, states))
        errs.append(float((Bm @ Am - true).norm() / true.norm().clamp(min=1e-12)))
    return float(np.mean(errs)) if errs else 0.0

def fedavg(states, weights, ref):
    w = np.asarray(weights, dtype=np.float64); w = w / w.sum()
    return {k: ref[k] + sum(float(wi) * (s[k] - ref[k]) for wi, s in zip(w, states)) for k in ref}

def round_lr(cfg: Config, rnd: int) -> float:
    """Linear warm-up over `warmup_rounds`, then cosine decay to `min_lr_frac * lr`."""
    if cfg.rounds <= 1:
        return cfg.lr
    if rnd < cfg.warmup_rounds:
        return cfg.lr * (rnd + 1) / max(cfg.warmup_rounds, 1)
    t = (rnd - cfg.warmup_rounds) / max(cfg.rounds - cfg.warmup_rounds - 1, 1)
    return cfg.lr * (cfg.min_lr_frac + (1 - cfg.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * min(t, 1.0))))

def objective_loss(model: FedMIMViT, cfg: Config, bidx, device, gen):
    """The local objective. Returns (loss, parts, n_correct)."""
    xs, xt = BATCH_TR.train_views(bidx, device, gen, jitter=cfg.asym_aug)
    loss = torch.zeros((), device=device)
    parts, correct = {}, 0
    if model.has_sup:
        y = BATCH_TR.labels_of(bidx, device)
        lg = model.logits(xs)
        ce = F.cross_entropy(lg.float(), y, label_smoothing=cfg.label_smoothing)
        loss = loss + ce
        parts["l_ce"] = float(ce.detach())
        correct = int((lg.argmax(-1) == y).sum())
    if model.has_mim:
        lm, pm = model.loss_mim(xs, xt, gen)
        loss = loss + (cfg.lambda_mim if model.has_sup else 1.0) * lm
        parts.update(pm)
    return loss, parts, correct

def local_train(model: FedMIMViT, cfg: Config, idxs: np.ndarray, device, lr: float, seed: int,
                global_state: Dict[str, torch.Tensor], keys: List[str]):
    model.train()
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    params = list(named.values())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=cfg.wd, betas=(0.9, 0.95))
    scaler = make_scaler(cfg.amp and is_cuda(device))
    gen = torch.Generator().manual_seed(seed)
    npg = np.random.RandomState(seed % (2 ** 31 - 1))
    prox = ({k: global_state[k].to(device) for k in keys if k in named} if cfg.fedprox_mu > 0 else None)

    tot, nst, correct, nseen = 0.0, 0, 0, 0
    acc_parts: Dict[str, float] = {}
    bs = cfg.batch_size
    for _ in range(cfg.local_epochs):
        perm = npg.permutation(len(idxs))
        nb = max(1, len(idxs) // bs)
        if cfg.max_local_steps:
            nb = min(nb, cfg.max_local_steps)
        for b in range(nb):
            bidx = idxs[perm[b * bs:(b + 1) * bs]]
            if len(bidx) < 2:
                continue
            with amp_ctx(device, cfg.amp):
                loss, parts, c = objective_loss(model, cfg, bidx, device, gen)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            if prox is not None:
                for k, p0 in prox.items():
                    p = named[k]
                    if p.grad is not None:
                        p.grad.add_(p.detach() - p0.to(p.dtype), alpha=cfg.fedprox_mu)
            torch.nn.utils.clip_grad_norm_(params, cfg.grad_clip)
            scaler.step(opt); scaler.update()
            tot += float(loss.detach()); nst += 1
            correct += c; nseen += len(bidx)
            for k, v in parts.items():
                acc_parts[k] = acc_parts.get(k, 0.0) + v
    extra = {k: v / max(nst, 1) for k, v in acc_parts.items()}
    if model.has_sup:
        extra["train_acc"] = correct / max(nseen, 1)
    return tot / max(nst, 1), nst, len(idxs), extra

class GPUPool:
    """One persistent model replica per device, handed to worker threads."""
    def __init__(self, cfg: Config, devices):
        self.q = queue.Queue()
        self.replicas = []
        for d in devices:
            if is_cuda(d):
                torch.cuda.set_device(d)
            self.replicas.append((build_model(cfg, d), d))
            self.q.put(len(self.replicas) - 1)
        self.keys = shared_keys(self.replicas[0][0], cfg)
        # the local heads start identical on every replica (same seed) and are re-initialised per client
        self.head_init = {n: p.detach().cpu().clone() for n, p in self.replicas[0][0].named_parameters()
                          if p.requires_grad and n not in self.keys}

    def acquire(self):
        i = self.q.get(); return i, self.replicas[i][0], self.replicas[i][1]

    def release(self, i):
        self.q.put(i)

    def close(self):
        for m, _ in self.replicas:
            m.to("cpu")
        self.replicas.clear()
        free_mem()

def run_round(pool: GPUPool, cfg: Config, global_state, client_ids, parts, rnd, lr, local_heads):
    """Local heads (MIM decoder/projector) persist per client across the rounds it is sampled in."""
    results, lock = {}, threading.Lock()

    def work(cid):
        i, model, dev = pool.acquire()
        try:
            with dev_ctx(dev):
                set_state(model, global_state)
                set_state(model, local_heads.get(cid, pool.head_init))
                seed = (cfg.seed * 7919 + rnd * 131 + cid) % (2 ** 31 - 1)
                loss, nst, nsamp, extra = local_train(model, cfg, parts[cid], dev, lr, seed,
                                                      global_state, pool.keys)
                st = get_state(model, pool.keys)
                hd = {n: p.detach().cpu().clone() for n, p in model.named_parameters()
                      if p.requires_grad and n not in pool.keys}
            with lock:
                results[cid] = dict(state=st, heads=hd, n=nsamp, loss=loss, steps=nst, **extra)
        finally:
            pool.release(i)

    with ThreadPoolExecutor(max_workers=len(pool.replicas)) as ex:
        list(ex.map(work, client_ids))
    for cid in client_ids:
        local_heads[cid] = results[cid].pop("heads")
    return [results[c] for c in client_ids]
# %% [markdown]
# ## 5. Evaluation
#
# All models, including the unadapted backbone, use the same frozen read-out ([CLS] concatenated with the mean patch token, 1,536-d):
#
# * kNN (k = 20, cosine, temperature 0.07) with the 45k client pool as the bank;
# * linear probe (40 epochs, AdamW, standardised features), the main metric;
# * 10% few-shot probe;
# * for supervised runs, the federated classifier head itself;
# * per-client accuracy on test subsets that follow each client's label distribution.
#
# Every round, kNN on the server validation split (bank of 50 images per class) gives the training curve and selects the reported round. Supervised runs also log the validation accuracy of the global head.

# %%
@torch.no_grad()
def pool_extract(replicas, state, batcher, idx: np.ndarray, want_logits=False, bs=256, amp=True):
    """Features (and head logits) for `idx`, split across every replica/GPU in parallel."""
    chunks = [c for c in np.array_split(idx, len(replicas))]
    outs: List[Any] = [None] * len(replicas)

    def work(j):
        model, dev = replicas[j]
        c = chunks[j]
        with dev_ctx(dev):
            if state is not None:
                set_state(model, state)
            model.eval()
            fs, ls = [], []
            for s in range(0, len(c), bs):
                x = batcher.eval_batch(c[s:s + bs], dev)
                with amp_ctx(dev, amp):
                    f, lg = model.features_logits(x, want_logits)
                fs.append(f.float())
                if lg is not None:
                    ls.append(lg.float())
            outs[j] = (torch.cat(fs).to(DEVICES[0]) if fs else None,
                       torch.cat(ls).to(DEVICES[0]) if ls else None)

    with ThreadPoolExecutor(max_workers=len(replicas)) as ex:
        list(ex.map(work, range(len(replicas))))
    feats = torch.cat([o[0] for o in outs if o[0] is not None])
    lgs = [o[1] for o in outs if o[1] is not None]
    return feats, (torch.cat(lgs) if want_logits and lgs else None)

@torch.no_grad()
def knn_classify(fb, yb, fq, yq, k=20, T=0.07, chunk=512, return_pred=False):
    fb, fq = F.normalize(fb, dim=1), F.normalize(fq, dim=1)
    preds = []
    for s in range(0, len(fq), chunk):
        d, i = (fq[s:s + chunk] @ fb.t()).topk(min(k, fb.shape[0]), dim=1)
        scores = torch.zeros(d.shape[0], NUM_CLASSES, device=fb.device)
        scores.scatter_add_(1, yb[i], (d / T).exp())
        preds.append(scores.argmax(1))
    pred = torch.cat(preds)
    acc = float((pred == yq).float().mean())
    return (acc, pred) if return_pred else acc

def linear_probe(ftr, ytr, fte, yte, epochs=40, lr=1e-3, wd=1e-4, bs=1024, seed=0):
    torch.manual_seed(seed)
    g = torch.Generator(device="cpu").manual_seed(seed)
    mu, sd = ftr.mean(0, keepdim=True), ftr.std(0, keepdim=True) + 1e-6
    Xtr, Xte = (ftr - mu) / sd, (fte - mu) / sd
    clf = nn.Linear(Xtr.shape[1], NUM_CLASSES).to(Xtr.device)
    nn.init.zeros_(clf.bias); nn.init.trunc_normal_(clf.weight, std=0.01)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=wd)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    n = len(Xtr)
    for _ in range(epochs):
        perm = torch.randperm(n, generator=g).to(Xtr.device)
        for s in range(0, n, bs):
            b = perm[s:s + bs]
            loss = F.cross_entropy(clf(Xtr[b]), ytr[b])
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        sch.step()
    with torch.no_grad():
        logits = clf(Xte)
        pred = logits.argmax(1)
        acc = float((pred == yte).float().mean())
        top5 = float((logits.topk(min(5, NUM_CLASSES), 1).indices == yte[:, None]).any(1).float().mean())
    return acc, top5, pred

def per_client_acc(pred: torch.Tensor, y: torch.Tensor, test_parts) -> np.ndarray:
    out = []
    for t in test_parts:
        if len(t):
            ti = torch.from_numpy(t).to(pred.device)
            out.append(float((pred[ti] == y[ti]).float().mean()))
    return np.array(out)

@torch.no_grad()
def val_eval(replicas, state, cfg: Config, has_head: bool) -> Dict[str, float]:
    idx = np.concatenate([VAL_BANK, VAL_IDX])
    f, lg = pool_extract(replicas, state, BATCH_TR, idx, want_logits=has_head, amp=cfg.amp)
    nb = len(VAL_BANK)
    yv = YTR_T[VAL_IDX].to(DEVICES[0])
    out = {"val_knn": knn_classify(f[:nb], YTR_T[VAL_BANK].to(DEVICES[0]), f[nb:], yv, k=cfg.knn_k)}
    if lg is not None:
        out["val_head"] = float((lg[nb:].argmax(1) == yv).float().mean())
    del f, lg
    return out

def full_eval(replicas, state, cfg: Config, has_head: bool, test_parts=None, seed=0) -> Dict[str, Any]:
    ytr = YTR_T[POOL_IDX].to(DEVICES[0]); yte = YTE_T[TEST_IDX].to(DEVICES[0])
    ftr, _ = pool_extract(replicas, state, BATCH_TR, POOL_IDX, amp=cfg.amp)
    fte, lte = pool_extract(replicas, state, BATCH_TE, TEST_IDX, want_logits=has_head, amp=cfg.amp)
    out: Dict[str, Any] = {}
    out["knn_top1"], kpred = knn_classify(ftr, ytr, fte, yte, k=cfg.knn_k, return_pred=True)
    out["probe_top1"], out["probe_top5"], ppred = linear_probe(ftr, ytr, fte, yte, epochs=cfg.probe_epochs,
                                                               seed=seed)
    k_fs = max(NUM_CLASSES, int(len(POOL_IDX) * cfg.fewshot_frac))
    sub = torch.from_numpy(np.sort(np.random.RandomState(7).choice(len(POOL_IDX), k_fs, replace=False)))
    sub = sub.to(DEVICES[0])
    out["fewshot_top1"] = linear_probe(ftr[sub], ytr[sub], fte, yte, epochs=cfg.probe_epochs, seed=seed)[0]
    out["_probe_pred"] = ppred.cpu().numpy().astype(int).tolist()
    out["_knn_pred"] = kpred.cpu().numpy().astype(int).tolist()
    if lte is not None:
        hpred = lte.argmax(1)
        out["head_top1"] = float((hpred == yte).float().mean())
        out["_head_pred"] = hpred.cpu().numpy().astype(int).tolist()
    if test_parts is not None:
        pc = per_client_acc(ppred, yte, test_parts)
        out.update(client_probe=pc.tolist(), client_probe_mean=float(pc.mean()),
                   client_probe_std=float(pc.std()), client_probe_worst10=float(np.percentile(pc, 10)))
        if lte is not None:
            ph = per_client_acc(lte.argmax(1), yte, test_parts)
            out.update(client_head=ph.tolist(), client_head_mean=float(ph.mean()),
                       client_head_std=float(ph.std()), client_head_worst10=float(np.percentile(ph, 10)))
    del ftr, fte, lte
    free_mem()
    return out

# %% [markdown]
# ### 5.1 Unadapted backbone
#
# The ImageNet-21k backbone with no adaptation, evaluated the same way. An earlier run with the Hugging Face implementation gave 84.36% kNN, which doubles as a check on the port.

# %%
ZS_PATH = DIRS["metrics"] / "zeroshot.json"
if ZS_PATH.exists():
    ZERO_SHOT = json.loads(ZS_PATH.read_text())
else:
    _t0 = time.time()
    _zr = [(build_model(BASE, d), d) for d in DEVICES]
    for _mm, _ in _zr:
        for m in _mm.modules():
            if isinstance(m, LoRALinear):
                m.mode = "off"
    ZERO_SHOT = full_eval(_zr, None, BASE, has_head=False, seed=BASE.seed)
    yte_ = YTE_T[TEST_IDX].to(DEVICES[0])
    _pp = torch.tensor(ZERO_SHOT["_probe_pred"], device=DEVICES[0])
    ZERO_SHOT["client_probe_by_partition"] = {
        f"{alpha_tag(a)}_s{sd}": per_client_acc(_pp, yte_, TEST_PARTS[(a, sd)]).tolist()
        for a in ALPHAS for sd in SEEDS}
    ZERO_SHOT["eval_s"] = time.time() - _t0
    ZS_PATH.write_text(json.dumps(ZERO_SHOT))
    for _mm, _ in _zr:
        _mm.to("cpu")
    del _zr; free_mem()

print("frozen ViT-B/16-in21k, no adaptation :")
print(f"  kNN {fmt_pct(ZERO_SHOT['knn_top1'])}%   probe {fmt_pct(ZERO_SHOT['probe_top1'])}%   "
      f"top-5 {fmt_pct(ZERO_SHOT['probe_top5'])}%   few-shot {fmt_pct(ZERO_SHOT['fewshot_top1'])}%")
if not TEST_MODE and PRESET == "full":
    _d = 100 * ZERO_SHOT["knn_top1"] - 84.36
    print(f"  cross-check vs earlier HF-implementation run: kNN differs by {_d:+.2f} pt "
          f"{'(OK)' if abs(_d) < 1.0 else '(large difference: check the backbone port)'}")
# %% [markdown]
# ## 6. Cost (P1)
#
# Peak memory from `torch.cuda.max_memory_allocated` over real optimisation steps (batch 64 unless stated), throughput in images/s on one T4, and uplink as the exact fp16 size of the aggregated tensors.

# %%
def bench_step(cfg: Config, device, steps=6, warmup=2):
    free_mem(); peak_reset(device)
    m = build_model(cfg, device)
    weight_mb = alloc_mb(device)
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-4)
    scaler = make_scaler(cfg.amp and is_cuda(device))
    gen = torch.Generator().manual_seed(0)
    idx = np.arange(cfg.batch_size); m.train(); t = time.time()
    for i in range(steps + warmup):
        if i == warmup:
            dsync(device); t = time.time(); peak_reset(device)
        with amp_ctx(device, cfg.amp):
            loss, _, _ = objective_loss(m, cfg, idx, device, gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    dsync(device)
    dt = (time.time() - t) / steps
    n_tr = sum(p.numel() for p in params)
    keys = shared_keys(m, cfg)
    row = dict(step_s=dt, img_s=cfg.batch_size / dt, peak_mb=peak_mb(device), weight_mb=weight_mb,
               opt_state_mb=8 * n_tr / 2 ** 20, trainable_M=n_tr / 1e6,
               uplink_mb=state_mb(get_state(m, keys)),
               uplink_lora_mb=state_mb({k: v for k, v in get_state(m, keys).items() if "lora_" in k}),
               enc_tokens=1 + int(round((cfg.img_size // cfg.patch_size) ** 2 *
                                        (1 - (cfg.mask_ratio if m.has_mim and not m.has_sup else 0)))))
    del m, opt; free_mem()
    return row

BENCH_PATH = DIRS["metrics"] / "bench.json"
if BENCH_PATH.exists():
    BENCH = pd.DataFrame(json.loads(BENCH_PATH.read_text()))
else:
    grid = [("FedMIM-LoRA (m=0.75, r=16)", {}),
            ("FedMIM-LoRA without input sparsity (m=0)", dict(mask_ratio=0.0))]
    if PRESET == "full" and not TEST_MODE:
        grid += [("FedMIM-LoRA + gradient checkpointing", dict(grad_ckpt=True)),
                 ("Supervised FedLoRA (r=16)", dict(objective="sup")),
                 ("Supervised + MIM hybrid", dict(objective="sup_mim"))]
        if HAS_BNB:
            grid += [("FedMIM-LoRA + NF4 backbone", dict(use_4bit=True)),
                     ("FedMIM-LoRA minimal (NF4, m=0.9, r=4, bs=16, ckpt)",
                      dict(use_4bit=True, mask_ratio=0.9, lora_r=4, lora_alpha=4.0, batch_size=16,
                           grad_ckpt=True))]
    rows = []
    for label, over in grid:
        try:
            r = bench_step(dc_replace(BASE, **over), DEVICES[0])
            r["configuration"] = label
            rows.append(r)
            print(f"  {label:<52} peak={r['peak_mb']:7.0f} MB  {r['img_s']:6.1f} img/s  "
                  f"uplink {r['uplink_mb']:.2f} MB (LoRA {r['uplink_lora_mb']:.2f})")
        except Exception as e:
            print(f"  [skip] {label}: {type(e).__name__}: {e}")
            free_mem()
    BENCH = pd.DataFrame(rows)
    BENCH_PATH.write_text(json.dumps(rows, indent=1))

if len(BENCH):
    _cols = ["configuration", "enc_tokens", "peak_mb", "weight_mb", "img_s", "trainable_M", "opt_state_mb",
             "uplink_lora_mb", "uplink_mb"]
    T_COST = BENCH[_cols].copy()
    T_COST["x below full-FT uplink"] = VIT_FULL_MB / T_COST["uplink_mb"]
    display(save_table(T_COST.round(2), "table2_cost",
        "Table 2. Measured cost of one local optimisation step on a single T4 (batch 64 unless stated). "
        f"Peak = torch.cuda.max_memory_allocated; uplink = exact fp16 payload per client per round; full "
        f"fine-tuning of the backbone would send {VIT_FULL_MB:.1f} MB.", "cost"))
# %% [markdown]
# ## 7. Client-gradient dissimilarity (H3)
#
# Standard non-convex FL analyses bound the stationarity gap by $\mathcal{O}(1/\sqrt{KT}+\kappa^2/T)$, where $\kappa^2$ bounds $\|\nabla F_k-\nabla F\|^2$. If MIM keeps clients closer together, $\kappa^2_{MIM}<\kappa^2_{SUP}$. The per-round update statistics in Sec. 4 mix gradient disagreement with optimiser drift and the learning-rate schedule, so here $\kappa^2$ is measured directly:
#
# 1. Both objectives are evaluated at the same parameters: pretrained backbone, $B=0$, the same $A$. Each objective's head is first warmed up centrally with the backbone and LoRA frozen.
# 2. Per-client gradients $g_k$ with respect to LoRA-$B$ (the only non-zero LoRA gradient at $B=0$), using the same images for both objectives.
# 3. Mini-batch noise makes even IID clients look different. With $s_k$ the variance of client $k$'s mean gradient (from its per-batch gradients),
# $$\kappa^2_{het}=\frac{\sum_k w_k\|g_k-\bar g\|^2-\sum_k w_k(1-w_k)s_k}{\|\bar g\|^2-\sum_k w_k^2 s_k}.$$
# Under IID it should be close to 0.
# 4. Confidence intervals by leave-one-client-out jackknife. Because both objectives are measured on the same clients, the difference $\kappa^2_{SUP}-\kappa^2_{MIM}$ is jackknifed jointly.
# 5. The gradient of CE + $\lambda\cdot$MIM is $g^{SUP}_k+\lambda g^{MIM}_k$, so $\kappa^2$ of the hybrid objective follows from the same gradients for a grid of $\lambda$.
#
# The client Gram matrices are saved to `metrics/geometry_grams.npz`, so the statistics can be recomputed without a GPU.

# %%
GEO_PATH = DIRS["metrics"] / "geometry.json"
GRAD_SCALE = 1024.0

def warm_heads(model: FedMIMViT, cfg: Config, dev, steps: int, bs: int, seed: int = 0):
    lora = [p for n, p in model.named_parameters() if "lora_" in n and p.requires_grad]
    for p in lora:
        p.requires_grad_(False)
    heads = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(heads, lr=1e-3, weight_decay=0.0)
    scaler = make_scaler(cfg.amp and is_cuda(dev))
    rng = np.random.RandomState(seed); gen = torch.Generator().manual_seed(seed)
    model.train(); last = float("nan")
    for _ in range(steps):
        bidx = rng.choice(POOL_IDX, bs, replace=False)
        with amp_ctx(dev, cfg.amp):
            loss, _, _ = objective_loss(model, cfg, bidx, dev, gen)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        last = float(loss.detach())
    for p in lora:
        p.requires_grad_(True)
    return last

def per_batch_grads(model: FedMIMViT, cfg: Config, dev, batches, seed: int) -> torch.Tensor:
    """[n_batches, P] gradients of the local objective w.r.t. every LoRA-B tensor."""
    model.train()
    Bp = [p for n, p in sorted(model.named_parameters()) if "lora_B" in n]
    out = []
    for b, bidx in enumerate(batches):
        gen = torch.Generator().manual_seed(seed * 1000 + b)
        model.zero_grad(set_to_none=True)
        with amp_ctx(dev, cfg.amp):
            loss, _, _ = objective_loss(model, cfg, bidx, dev, gen)
        (loss.float() * GRAD_SCALE).backward()
        g = torch.cat([p.grad.detach().float().reshape(-1) for p in Bp]) / GRAD_SCALE
        if not torch.isfinite(g).all():
            raise FloatingPointError("non-finite gradient in the geometry probe")
        out.append(g.cpu())
    model.zero_grad(set_to_none=True)
    return torch.stack(out)

def kappa_from_gram(M: np.ndarray, s: np.ndarray, w: np.ndarray) -> Dict[str, float]:
    """κ² (raw and noise-debiased) and mean pairwise cosine from the client Gram matrix M = g g^T."""
    w = w / w.sum()
    Mw = M @ w; wMw = float(w @ Mw)
    d2 = np.diag(M) - 2 * Mw + wMw                          # ||g_k - ḡ||²
    num, den = float(w @ d2), wMw
    num_h = num - float(np.sum(w * (1 - w) * s)); den_h = den - float(np.sum(w ** 2 * s))
    nrm = np.sqrt(np.clip(np.diag(M), 1e-30, None))
    C = M / np.outer(nrm, nrm); K = len(w)
    cos = float((C.sum() - np.trace(C)) / max(K * (K - 1), 1))
    return dict(kappa2_raw=num / max(den, 1e-30), kappa2_het=max(num_h, 0.0) / max(den_h, 1e-30),
                cos=cos, gbar_norm=float(np.sqrt(max(den, 0.0))), noise_frac=float(np.sum(w * s)) / max(num, 1e-30))

def client_gram(Gs: List[torch.Tensor]):
    """Client-mean gradient Gram matrix M = g g^T and per-client mini-batch noise s_k = tr Var / nb."""
    gk = torch.stack([g.mean(0) for g in Gs]).double()
    nb = Gs[0].shape[0]
    s = np.array([float(((g.double() - g.double().mean(0)) ** 2).sum()) / (nb * max(nb - 1, 1)) for g in Gs])
    return (gk @ gk.t()).numpy(), s

def _loo(M, s, w, k):
    keep = np.r_[0:k, k + 1:len(w)]
    return kappa_from_gram(M[np.ix_(keep, keep)], s[keep], w[keep])["kappa2_het"]

def jackknife(theta, thetas_loo):
    """Leave-one-client-out jackknife: standard error and normal 95% interval around the full estimate."""
    t = np.asarray(thetas_loo, dtype=np.float64); K = len(t)
    se = float(np.sqrt((K - 1) / K * np.sum((t - t.mean()) ** 2)))
    return se, theta - 1.96 * se, theta + 1.96 * se

def geometry_stats(M, s, w) -> Dict[str, Any]:
    w = w.astype(np.float64)
    res = kappa_from_gram(M, s, w)
    se, lo, hi = jackknife(res["kappa2_het"], [_loo(M, s, w, k) for k in range(len(w))])
    res.update(kappa2_het_se=se, kappa2_het_lo=max(lo, 0.0), kappa2_het_hi=hi)
    return res

def paired_stats(Mm, sm, Ms, ss, w) -> Dict[str, float]:
    """SUP vs MIM on the same clients: difference and log-ratio of κ²_het, jackknifed jointly over clients."""
    w = w.astype(np.float64)
    km, ks = kappa_from_gram(Mm, sm, w)["kappa2_het"], kappa_from_gram(Ms, ss, w)["kappa2_het"]
    lm = np.array([_loo(Mm, sm, w, k) for k in range(len(w))])
    ls = np.array([_loo(Ms, ss, w, k) for k in range(len(w))])
    se_d, d_lo, d_hi = jackknife(ks - km, ls - lm)
    out = dict(diff=ks - km, diff_se=se_d, diff_lo=d_lo, diff_hi=d_hi)
    if km > 1e-6 and ks > 1e-6 and np.all(lm > 1e-6) and np.all(ls > 1e-6):
        se_r, r_lo, r_hi = jackknife(np.log(ks / km), np.log(ls / lm))
        out.update(ratio=ks / km, ratio_lo=float(np.exp(r_lo)), ratio_hi=float(np.exp(r_hi)))
    return out

def run_geometry():
    t0 = time.time()
    objs = ["mim", "sup"]
    devs = [DEVICES[0], DEVICES[1 % len(DEVICES)]]
    models = {}
    for o, d in zip(objs, devs):
        cfg_o = dc_replace(BASE, objective=o, seed=SEEDS[0])
        models[o] = (build_model(cfg_o, d), d, cfg_o)
        with dev_ctx(d):
            l = warm_heads(models[o][0], cfg_o, d, GEO["warm_steps"], GEO["bs"], seed=1)
        print(f"[geometry] {o}: head warm-up done ({GEO['warm_steps']} steps, last loss {l:.3f})")
    res = {"per_alpha": {}, "hybrid": {}, "config": dict(GEO), "ci": "jackknife over clients"}
    lam_grid = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    for a in ALPHAS:
        parts = parts_for(a, SEEDS[0])
        w = np.array([len(p) for p in parts], dtype=np.float64)
        rng = np.random.RandomState(777)
        batches = [[rng.choice(p, GEO["bs"], replace=len(p) < GEO["bs"] * GEO["batches"])
                    for _ in range(GEO["batches"])] for p in parts]
        G = {}

        def work(o):
            m, d, c = models[o]
            with dev_ctx(d):
                G[o] = [per_batch_grads(m, c, d, batches[k], seed=k) for k in range(len(parts))]
        with ThreadPoolExecutor(max_workers=2) as ex:
            list(ex.map(work, objs))
        grams = {o: client_gram(G[o]) for o in objs}
        for o in objs:
            GRAMS[f"{alpha_tag(a)}_{o}_M"], GRAMS[f"{alpha_tag(a)}_{o}_s"] = grams[o]
        GRAMS[f"{alpha_tag(a)}_w"] = w
        r_ = {o: geometry_stats(*grams[o], w) for o in objs}
        r_["paired"] = paired_stats(*grams["mim"], *grams["sup"], w)
        res["per_alpha"][alpha_tag(a)] = r_
        hy = {}
        for lam in lam_grid:
            Mh, sh = client_gram([G["sup"][k] + lam * G["mim"][k] for k in range(len(parts))])
            hy[str(lam)] = kappa_from_gram(Mh, sh, w)["kappa2_het"]
        hy["inf"] = r_["mim"]["kappa2_het"]
        res["hybrid"][alpha_tag(a)] = hy
        p_ = r_["paired"]
        print(f"[geometry] {alpha_label(a):>7}: κ²_het MIM={r_['mim']['kappa2_het']:.4f} "
              f"[{r_['mim']['kappa2_het_lo']:.4f},{r_['mim']['kappa2_het_hi']:.4f}]  "
              f"SUP={r_['sup']['kappa2_het']:.4f} [{r_['sup']['kappa2_het_lo']:.4f},{r_['sup']['kappa2_het_hi']:.4f}]"
              f"  | SUP−MIM={p_['diff']:+.4f} [{p_['diff_lo']:+.4f},{p_['diff_hi']:+.4f}]")
        del G; free_mem()
    for o in objs:
        models[o][0].to("cpu")
    del models; free_mem()
    res["wall_s"] = time.time() - t0
    return res

GRAMS: Dict[str, np.ndarray] = {}
GEOM = json.loads(GEO_PATH.read_text()) if GEO_PATH.exists() else None
if GEOM is None or GEOM.get("ci") != "jackknife over clients":
    GEOM = None
    try:
        GEOM = run_geometry()
        GEO_PATH.write_text(json.dumps(GEOM, indent=1))
        np.savez_compressed(DIRS["metrics"] / "geometry_grams.npz", **GRAMS)
        print(f"[geometry] finished in {GEOM['wall_s']/60:.1f} min")
    except Exception as e:
        print(f"[geometry FAILED] {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        GEOM = None; free_mem()

if GEOM:
    _rows = []
    for a in ALPHAS:
        g = GEOM["per_alpha"].get(alpha_tag(a))
        if not g:
            continue
        for o in ["mim", "sup"]:
            p_ = g.get("paired", {})
            _rows.append({"client split": alpha_label(a), "objective": OBJ_LABEL[o],
                          "κ² raw": g[o]["kappa2_raw"], "κ² het": g[o]["kappa2_het"],
                          "95% CI lo": g[o]["kappa2_het_lo"], "95% CI hi": g[o]["kappa2_het_hi"],
                          "SUP − MIM": p_.get("diff", np.nan) if o == "sup" else np.nan,
                          "diff CI lo": p_.get("diff_lo", np.nan) if o == "sup" else np.nan,
                          "diff CI hi": p_.get("diff_hi", np.nan) if o == "sup" else np.nan,
                          "noise share of raw": g[o]["noise_frac"]})
    T_GEOM = save_table(pd.DataFrame(_rows).round(4), "table3_gradient_geometry",
        "Table 3. Client-gradient dissimilarity at a common parameter point (pretrained backbone, B=0, "
        "shared A, heads warmed centrally). κ² het removes the mini-batch noise contribution and should be "
        "close to 0 under IID. Intervals are leave-one-client-out jackknife 95% CIs; the SUP − MIM "
        "difference is jackknifed jointly over the shared clients.", "geometry", floatfmt="%.4f")
    display(T_GEOM)
# %% [markdown]
# ## 8. Runs
#
# * Each run writes `metrics/<name>.json` (per-round history, final evaluation, test predictions) and `checkpoints/<name>.pth` (the selected aggregated tensors).
# * The reported model is the round with the best validation kNN; the last round is also evaluated when it differs.
# * A run whose loss becomes non-finite is stopped and recorded as diverged.

# %%
def load_ckpt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def strip_preds(d: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in (d or {}).items() if not k.startswith("_")}

def run_experiment(cfg: Config, arm: str) -> Optional[Dict[str, Any]]:
    mpath = DIRS["metrics"] / f"{cfg.name}.json"
    if mpath.exists():
        h = json.loads(mpath.read_text())
        print(f"[cached] {cfg.name}: probe={100*h['final']['probe_top1']:.2f}%")
        return h
    if MODE == "test_only":
        return None
    set_seed(cfg.seed)
    t_start = time.time()
    parts = parts_for(cfg.dirichlet_alpha, cfg.seed)
    tparts = TEST_PARTS[(cfg.dirichlet_alpha, cfg.seed)]
    pool = GPUPool(cfg, DEVICES)
    ref_model = pool.replicas[0][0]
    has_head = ref_model.has_sup
    keys = pool.keys
    global_state = get_state(ref_model, keys)
    n_sel = max(1, int(round(cfg.num_clients * cfg.participation)))
    rng = np.random.RandomState(cfg.seed)
    hist: Dict[str, Any] = {"cfg": asdict(cfg), "arm": arm, "rounds": [], "preset": PRESET, "diverged": False}
    up_all = state_mb(global_state)
    up_lora = state_mb({k: v for k, v in global_state.items() if "lora_" in k})
    best = {"val_knn": -1.0, "round": -1}
    best_state = {k: v.clone() for k, v in global_state.items()}
    local_heads: Dict[int, Dict[str, torch.Tensor]] = {}
    hist["build_s"] = time.time() - t_start
    try:
        for rnd in range(cfg.rounds):
            t0 = time.time()
            lr = round_lr(cfg, rnd)
            sel = sorted(rng.choice(cfg.num_clients, n_sel, replace=False).tolist())
            res = run_round(pool, cfg, global_state, sel, parts, rnd, lr, local_heads)
            states, ws = [r["state"] for r in res], [r["n"] for r in res]
            losses = [r["loss"] for r in res]
            if not all(np.isfinite(losses)):
                hist["diverged"] = True
                print(f"[{cfg.name}] DIVERGED at round {rnd+1} (non-finite client loss)")
                break
            dis = dissimilarity_metrics(states, global_state, ws)
            eps = abo_error(states, ws) if not cfg.fixed_a else 0.0
            global_state = fedavg(states, ws, global_state)
            rec = dict(round=rnd, lr=lr, loss=float(np.average(losses, weights=ws)),
                       client_losses=[float(x) for x in losses], clients=sel,
                       steps=int(sum(r["steps"] for r in res)), abo_err=eps,
                       time_s=time.time() - t0, **dis)
            for ek in ("l_patch", "l_cls", "l_ce", "train_acc"):
                vals = [r[ek] for r in res if ek in r]
                if vals:
                    rec[ek] = float(np.average(vals, weights=ws))
            te = time.time()
            rec.update(val_eval(pool.replicas, global_state, cfg, has_head))
            rec["eval_s"] = time.time() - te
            if rec["val_knn"] > best["val_knn"]:
                best = {"val_knn": rec["val_knn"], "round": rnd}
                best_state = {k: v.clone() for k, v in global_state.items()}
            hist["rounds"].append(rec)
            msg = (f"[{cfg.name}] r{rnd+1:02d}/{cfg.rounds} loss={rec['loss']:.4f} κ={rec['kappa']:.3f} "
                   f"cos={rec['pairwise_cos']:+.3f} |Δ|={rec['update_norm']:.3f} ε_ABO={eps:.1e} "
                   f"val-kNN={100*rec['val_knn']:.2f}%")
            if "val_head" in rec:
                msg += f" val-head={100*rec['val_head']:.2f}%"
            print(msg + f"  ({rec['time_s']:.0f}+{rec['eval_s']:.0f}s)", flush=True)

        if not hist["rounds"]:
            raise RuntimeError("no completed rounds")
        tf = time.time()
        sel_state = best_state if (cfg.select_on == "val_knn" and best["round"] >= 0) else global_state
        final = full_eval(pool.replicas, sel_state, cfg, has_head, tparts, seed=cfg.seed)
        last_round = len(hist["rounds"]) - 1
        if best["round"] == last_round or cfg.select_on != "val_knn":
            final_last = strip_preds(final)
        else:
            final_last = strip_preds(full_eval(pool.replicas, global_state, cfg, has_head, tparts, seed=cfg.seed))
        torch.save({"config": asdict(cfg), "round": best["round"], "state": sel_state,
                    "metrics": strip_preds(final)}, DIRS["checkpoints"] / f"{cfg.name}.pth")
        hist.update(final=final, final_last_round=final_last, best=best, selected_round=int(best["round"]),
                    final_eval_s=time.time() - tf, wall_s=time.time() - t_start,
                    uplink_mb_per_client_round=up_all, uplink_lora_mb=up_lora,
                    total_uplink_mb=up_all * n_sel * len(hist["rounds"]), n_selected_per_round=n_sel)
        mpath.write_text(json.dumps(hist))
        calibrate(cfg, hist)
        msg = (f"[done] {cfg.name}: probe={100*final['probe_top1']:.2f}%  kNN={100*final['knn_top1']:.2f}%"
               f"  few-shot={100*final['fewshot_top1']:.2f}%")
        if "head_top1" in final:
            msg += f"  head={100*final['head_top1']:.2f}%"
        msg += (f"  per-client probe mean/worst10={100*final['client_probe_mean']:.2f}/"
                f"{100*final['client_probe_worst10']:.2f}%  [sel. round {best['round']+1}; last-round probe "
                f"{100*final_last['probe_top1']:.2f}%]  ({hist['wall_s']/60:.1f} min)")
        print(msg, flush=True)
        return hist
    finally:
        pool.close()

def safe_run(cfg: Config, arm: str) -> Optional[Dict[str, Any]]:
    cached = (DIRS["metrics"] / f"{cfg.name}.json").exists()
    if not cached and MODE != "test_only":
        c, l = cost_min(cfg), left_min()
        if c > l:
            print(f"[budget] SKIP {cfg.name}: needs ~{c:.0f} min, {l:.0f} min left")
            return None
        print(f"[budget] {elapsed_min():.0f} min used, {l:.0f} min left; launching {cfg.name} (~{c:.0f} min)")
    try:
        return run_experiment(cfg, arm)
    except Exception as e:
        print(f"[FAILED] {cfg.name}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=4)
        free_mem()
        return None

# %% [markdown]
# ### 8.1 Run order
#
# | tier | runs | purpose |
# |---|---|---|
# | 1 | MIM and SUP at every split, seed 42 | full curve for both objectives |
# | 2 | MIM and SUP at $\alpha=0.05$ and IID, seeds 43, 44 | three seeds on the main contrast |
# | 3 | CE + MIM at $\alpha=0.05$ and IID, seeds 42, 43 | H5 |
# | 4 | SUP + FedProx ($\mu=0.1$) at $\alpha=0.05$, seeds 42, 43 | drift-correction baseline |
# | 5 | MIM and SUP at $\alpha=0.1$, $0.5$, seeds 43, 44 | three seeds on the whole curve |
# | 6 | CE + MIM and FedProx, seed 44 | third seed for tiers 3 and 4 |
#
# If time runs short, runs are dropped from the bottom of this table.

# %%
FEDPROX_MU = 0.1

def make_run(arm: str, a: float, sd: int) -> Config:
    over = dict(dirichlet_alpha=a, seed=sd)
    if arm == "mim":
        over["objective"] = "mim"
    elif arm == "sup":
        over["objective"] = "sup"
    elif arm == "sup_mim":
        over["objective"] = "sup_mim"
    elif arm == "sup_prox":
        over.update(objective="sup", fedprox_mu=FEDPROX_MU)
    elif arm == "mim_fixA":
        over.update(objective="mim", fixed_a=True)
    else:
        raise ValueError(arm)
    return dc_replace(BASE, name=f"{arm}_a{alpha_tag(a)}_s{sd}", **over)

A_LO = min([a for a in ALPHAS if np.isfinite(a)])
A_MID = [a for a in ALPHAS if np.isfinite(a) and a != A_LO]
S0, S_EXTRA = SEEDS[0], SEEDS[1:]
PLAN: List[Tuple[str, float, int]] = []
for a in [A_LO, IID] + A_MID:                                    # tier 1
    PLAN += [("mim", a, S0), ("sup", a, S0)]
for sd in S_EXTRA:                                               # tier 2
    for a in [A_LO, IID]:
        PLAN += [("mim", a, sd), ("sup", a, sd)]
for sd in SEEDS[:2]:                                             # tier 3
    PLAN += [("sup_mim", A_LO, sd), ("sup_mim", IID, sd)]
for sd in SEEDS[:2]:                                             # tier 4
    PLAN += [("sup_prox", A_LO, sd)]
for sd in S_EXTRA:                                               # tier 5
    for a in A_MID:
        PLAN += [("mim", a, sd), ("sup", a, sd)]
for sd in SEEDS[2:]:                                             # tier 6
    PLAN += [("sup_mim", A_LO, sd), ("sup_mim", IID, sd), ("sup_prox", A_LO, sd)]
PLAN = list(dict.fromkeys(PLAN))
_est = sum(cost_min(make_run(*p)) for p in PLAN if not (DIRS["metrics"] / f"{make_run(*p).name}.json").exists())
print(f"{len(PLAN)} runs planned; estimated {_est:.0f} min for the uncached ones, {left_min():.0f} min available")

# %%
HISTORIES: Dict[str, Dict[str, Any]] = {}
for arm, a, sd in PLAN:
    cfg = make_run(arm, a, sd)
    h = safe_run(cfg, arm)
    if h:
        h["arm"] = arm
        HISTORIES[cfg.name] = h
    free_mem()

# pick up anything else cached (e.g. from an attached previous session)
for p in sorted(DIRS["metrics"].glob("*.json")):
    if p.stem not in HISTORIES and not p.stem.startswith("_") and p.stem not in ("zeroshot", "bench", "geometry"):
        try:
            h = json.loads(p.read_text())
            if "rounds" in h and "final" in h:
                HISTORIES[p.stem] = h
        except Exception:
            pass
print(f"\n{len(HISTORIES)} completed runs; {elapsed_min():.0f} min elapsed of {TIME_BUDGET_H*60:.0f}")
# %% [markdown]
# ## 9. Analysis
#
# Heterogeneity penalty: $\Pi(\alpha)=\mathrm{acc}_{IID}-\mathrm{acc}_\alpha$ for the same method and seed (positive means skew hurts), averaged over seeds. Two kinds of uncertainty are reported:
#
# * spread over seeds, and whether the sign holds in every seed;
# * a paired test-set bootstrap (5,000 resamples of the 10,000 test images, shared by all runs).
#
# $\mathrm{DiD}(\alpha)=\Pi_{SUP}(\alpha)-\Pi_{MIM}(\alpha)$ tests H2. H1 is an equivalence test: the 90% CI of $\Pi_{MIM}$ must lie inside $\pm0.5$ pt.

# %%
ARMS_ORDER = ["mim", "sup", "sup_mim", "sup_prox", "mim_fixA"]
ARM_COLOR = {"mim": PALETTE[0], "sup": PALETTE[1], "sup_mim": PALETTE[2], "sup_prox": PALETTE[3],
             "mim_fixA": PALETTE[5]}
ARM_MARK = {"mim": "o", "sup": "s", "sup_mim": "^", "sup_prox": "D", "mim_fixA": "v"}
SPLITS = sorted(ALPHAS, key=lambda a: -a if np.isfinite(a) else -1e9)       # IID first, then decreasing α
YTE_NP = YTE[TEST_IDX]

def cfg_of(h):
    return h["cfg"]

def arm_of(h):
    return h.get("arm", "mim")

def cells() -> Dict[Tuple[str, float], Dict[int, Dict[str, Any]]]:
    out: Dict[Tuple[str, float], Dict[int, Dict[str, Any]]] = {}
    for n, h in HISTORIES.items():
        c = cfg_of(h)
        out.setdefault((arm_of(h), float(c["dirichlet_alpha"])), {})[int(c["seed"])] = h
    return out

CELLS = cells()

def metric_of(h, metric="probe"):
    key = {"probe": "probe_top1", "knn": "knn_top1", "head": "head_top1", "fewshot": "fewshot_top1"}[metric]
    v = h["final"].get(key)
    return 100 * v if v is not None else np.nan

def correct_vec(h, metric="probe") -> Optional[np.ndarray]:
    key = {"probe": "_probe_pred", "knn": "_knn_pred", "head": "_head_pred"}[metric]
    p = h["final"].get(key)
    return None if p is None else (np.asarray(p) == YTE_NP)

_rows = []
for n, h in HISTORIES.items():
    c, f = cfg_of(h), h["final"]
    rs = h["rounds"]
    _rows.append(dict(
        run=n, arm=arm_of(h), alpha=float(c["dirichlet_alpha"]), split=alpha_label(float(c["dirichlet_alpha"])),
        seed=int(c["seed"]), probe=metric_of(h, "probe"), knn=metric_of(h, "knn"),
        fewshot=metric_of(h, "fewshot"), head=metric_of(h, "head"),
        probe_last=100 * h["final_last_round"].get("probe_top1", np.nan),
        client_probe_std=100 * f.get("client_probe_std", np.nan),
        client_probe_worst10=100 * f.get("client_probe_worst10", np.nan),
        client_head_std=100 * f.get("client_head_std", np.nan),
        client_head_worst10=100 * f.get("client_head_worst10", np.nan),
        selected_round=int(h.get("selected_round", -1)) + 1, rounds=len(rs), diverged=bool(h.get("diverged")),
        uplink_mb=h.get("uplink_mb_per_client_round", np.nan), wall_min=h.get("wall_s", np.nan) / 60))
RUNS = pd.DataFrame(_rows)
if len(RUNS):
    RUNS = RUNS.sort_values(["arm", "alpha", "seed"], key=lambda s: s.map(
        {a: i for i, a in enumerate(ARMS_ORDER)}) if s.name == "arm" else s).reset_index(drop=True)
print(f"{len(RUNS)} runs available for analysis")

# %%
# ---- Table 4: the main grid ---------------------------------------------------------------------
_rows = []
for arm in ARMS_ORDER:
    for a in SPLITS:
        cl = CELLS.get((arm, a))
        if not cl:
            continue
        hs = list(cl.values())
        row = {"method": OBJ_LABEL[arm], "split": alpha_label(a), "seeds": len(hs)}
        for m in ["probe", "knn", "fewshot", "head"]:
            v = np.array([metric_of(h, m) for h in hs])
            if np.all(np.isnan(v)):
                row[m] = "—"
            else:
                row[m] = f"{np.nanmean(v):.2f} ± {np.nanstd(v, ddof=1) if len(v) > 1 else 0:.2f}"
        row["Δ probe vs zero-shot"] = np.nanmean([metric_of(h, 'probe') for h in hs]) - 100 * ZERO_SHOT["probe_top1"]
        row["uplink MB/round"] = hs[0].get("uplink_mb_per_client_round", np.nan)
        _rows.append(row)
_rows.insert(0, {"method": "frozen backbone (zero-shot)", "split": "—", "seeds": 1,
                 "probe": f"{100*ZERO_SHOT['probe_top1']:.2f}", "knn": f"{100*ZERO_SHOT['knn_top1']:.2f}",
                 "fewshot": f"{100*ZERO_SHOT['fewshot_top1']:.2f}", "head": "—",
                 "Δ probe vs zero-shot": 0.0, "uplink MB/round": 0.0})
T_MAIN = save_table(pd.DataFrame(_rows), "table4_main",
    f"Table 4. CIFAR-100 top-1 (%) of the validation-selected global model, mean ± s.d. over seeds, "
    f"{BASE.num_clients} clients, {BASE.participation:.0%} participation, {BASE.rounds} rounds, "
    "frozen ViT-B/16-in21k + LoRA r=16. 'head' is the federated classifier of supervised arms. The "
    "supervised arms use labels on every client; FedMIM-LoRA uses none.", "main")
display(T_MAIN)

# %%
# ---- paired test-set bootstrap machinery ---------------------------------------------------------
N_BOOT = 5000 if not TEST_MODE else 200
BOOT_IDX = np.random.RandomState(2024).randint(0, len(YTE_NP), size=(N_BOOT, len(YTE_NP)))

def boot_mean_acc(correct: np.ndarray) -> np.ndarray:
    return 100 * correct[BOOT_IDX].mean(1)

def penalty(arm: str, a: float, metric="probe", ref_arm: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Π(α) = acc(ref_arm, IID) - acc(arm, α), paired by seed. ref_arm defaults to arm."""
    ref_arm = ref_arm or arm
    ci, ca = CELLS.get((ref_arm, IID), {}), CELLS.get((arm, a), {})
    seeds = sorted(set(ci) & set(ca))
    if not seeds:
        return None
    per_seed, boots = [], []
    for sd in seeds:
        ai, aa = metric_of(ci[sd], metric), metric_of(ca[sd], metric)
        per_seed.append(ai - aa)
        cvi, cva = correct_vec(ci[sd], metric), correct_vec(ca[sd], metric)
        if cvi is not None and cva is not None:
            boots.append(boot_mean_acc(cvi) - boot_mean_acc(cva))
    ps = np.array(per_seed)
    out = dict(arm=arm, split=alpha_label(a), metric=metric, n=len(seeds), seeds=seeds, mean=float(ps.mean()),
               sd=float(ps.std(ddof=1)) if len(ps) > 1 else np.nan, per_seed=ps.tolist(),
               sign_pos=int((ps > 0).sum()))
    if boots:
        B = np.mean(boots, axis=0)
        out.update(lo95=float(np.percentile(B, 2.5)), hi95=float(np.percentile(B, 97.5)),
                   lo90=float(np.percentile(B, 5)), hi90=float(np.percentile(B, 95)))
    return out

def did(a: float, metric="probe", arm_a="sup", arm_b="mim") -> Optional[Dict[str, Any]]:
    """DiD = Π_armA(α) - Π_armB(α), paired by seed and by test resample."""
    need = [(arm_a, IID), (arm_a, a), (arm_b, IID), (arm_b, a)]
    if any(k not in CELLS for k in need):
        return None
    seeds = sorted(set.intersection(*[set(CELLS[k]) for k in need]))
    if not seeds:
        return None
    per_seed, boots = [], []
    for sd in seeds:
        hs = [CELLS[k][sd] for k in need]
        v = [metric_of(h, metric) for h in hs]
        per_seed.append((v[0] - v[1]) - (v[2] - v[3]))
        cv = [correct_vec(h, metric) for h in hs]
        if all(x is not None for x in cv):
            b = [boot_mean_acc(x) for x in cv]
            boots.append((b[0] - b[1]) - (b[2] - b[3]))
    ps = np.array(per_seed)
    out = dict(split=alpha_label(a), metric=metric, n=len(seeds), mean=float(ps.mean()),
               sd=float(ps.std(ddof=1)) if len(ps) > 1 else np.nan, per_seed=ps.tolist(),
               sign_pos=int((ps > 0).sum()))
    if boots:
        B = np.mean(boots, axis=0)
        out.update(lo95=float(np.percentile(B, 2.5)), hi95=float(np.percentile(B, 97.5)))
    return out

# ---- Table 5: heterogeneity penalties and the difference-in-differences ---------------------------
PEN: Dict[Tuple[str, str, float], Dict[str, Any]] = {}
_rows = []
for metric in ["probe", "knn", "head"]:
    for arm in ARMS_ORDER:
        for a in [x for x in SPLITS if np.isfinite(x)]:
            ref = "sup" if arm == "sup_prox" else None
            if metric == "head" and arm not in ("sup", "sup_mim", "sup_prox"):
                continue
            p = penalty(arm, a, metric, ref_arm=ref)
            if p is None:
                continue
            PEN[(metric, arm, a)] = p
            _rows.append({"metric": metric, "method": OBJ_LABEL[arm] + (" (vs SUP-IID)" if ref else ""),
                          "split": p["split"], "seeds": p["n"], "penalty (pt)": p["mean"],
                          "seed s.d.": p["sd"], "95% CI lo": p.get("lo95", np.nan),
                          "95% CI hi": p.get("hi95", np.nan), "seeds with penalty>0": f"{p['sign_pos']}/{p['n']}"})
T_PEN = pd.DataFrame(_rows)
if len(T_PEN):
    save_table(T_PEN.round(3), "table5_penalties",
        "Table 5. Heterogeneity penalty Π(α) = acc(IID) − acc(α) in points (positive = label skew hurts), "
        "paired by seed; CIs are a paired test-set bootstrap pooled over seeds. FedProx is referenced to "
        "the plain supervised IID run because FedProx is a non-IID remedy.", "penalties", floatfmt="%.3f")
    display(T_PEN.round(3))

DID: Dict[Tuple[str, float], Dict[str, Any]] = {}
_rows = []
for metric in ["probe", "knn"]:
    for a in [x for x in SPLITS if np.isfinite(x)]:
        d = did(a, metric)
        if d:
            DID[(metric, a)] = d
            _rows.append({"metric": metric, "split": d["split"], "seeds": d["n"],
                          "DiD = Π_SUP − Π_MIM (pt)": d["mean"], "seed s.d.": d["sd"],
                          "95% CI lo": d.get("lo95", np.nan), "95% CI hi": d.get("hi95", np.nan),
                          "seeds with DiD>0": f"{d['sign_pos']}/{d['n']}"})
T_DID = pd.DataFrame(_rows)
if len(T_DID):
    save_table(T_DID.round(3), "table6_did",
        "Table 6. Difference-in-differences of the heterogeneity penalty, supervised FedLoRA minus "
        "FedMIM-LoRA. Positive = MIM loses less accuracy to label skew. Paired by seed and test resample.",
        "did", floatfmt="%.3f")
    display(T_DID.round(3))

# %%
# ---- sensitivity slope: accuracy vs the effective number of classes per client -------------------
def eff_classes_of(a, sd):
    r = PART_STATS_ALL[(PART_STATS_ALL.alpha == alpha_label(a)) & (PART_STATS_ALL.seed == sd)]
    return float(r.eff_classes_mean.iloc[0]) if len(r) else np.nan

def ols(X, y):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    dof = max(len(y) - X.shape[1], 1)
    s2 = float(resid @ resid) / dof
    cov = s2 * np.linalg.pinv(X.T @ X)
    return beta, np.sqrt(np.clip(np.diag(cov), 0, None)), dof

SLOPES = {}
_pts = []
for arm in ["mim", "sup", "sup_mim"]:
    for (ar, a), cl in CELLS.items():
        if ar != arm:
            continue
        for sd, h in cl.items():
            _pts.append((arm, np.log10(eff_classes_of(a, sd)), metric_of(h, "probe")))
_pts = pd.DataFrame(_pts, columns=["arm", "x", "y"]).dropna()
for arm in ["mim", "sup", "sup_mim"]:
    d = _pts[_pts.arm == arm]
    if len(d) >= 3 and d.x.nunique() >= 2:
        b, se, dof = ols(np.c_[np.ones(len(d)), d.x.values], d.y.values)
        SLOPES[arm] = dict(slope=float(b[1]), se=float(se[1]), n=len(d))
d = _pts[_pts.arm.isin(["mim", "sup"])]
if len(d) >= 5 and d[d.arm == "mim"].x.nunique() >= 2 and d[d.arm == "sup"].x.nunique() >= 2:
    s = (d.arm == "sup").astype(float).values
    X = np.c_[np.ones(len(d)), d.x.values, s, d.x.values * s]
    b, se, dof = ols(X, d.y.values)
    SLOPES["interaction_sup_minus_mim"] = dict(slope=float(b[3]), se=float(se[3]), n=len(d), dof=dof)
for k, v in SLOPES.items():
    print(f"slope[{k}]: {v['slope']:+.3f} ± {v['se']:.3f} pt per decade of effective classes (n={v['n']})")
# %%
# ---- Table 7: trajectory stability (H4) ----------------------------------------------------------
def traj_stats(values: List[float]) -> Dict[str, float]:
    v = 100 * np.asarray(values, dtype=np.float64)
    if len(v) < 2:
        return dict(backtrack=np.nan, n_drops=np.nan, max_drawdown=np.nan, final_vs_best=np.nan, roughness=np.nan)
    d = np.diff(v)
    return dict(backtrack=float(np.clip(-d, 0, None).sum()),            # accuracy given back, summed
                n_drops=int((d < -0.05).sum()),
                max_drawdown=float((np.maximum.accumulate(v) - v).max()),
                final_vs_best=float(v[-1] - v.max()),
                roughness=float(np.abs(np.diff(d)).mean()) if len(d) > 1 else 0.0)

def run_dynamics(h) -> Dict[str, float]:
    rs = h["rounds"]
    out = {f"valknn_{k}": v for k, v in traj_stats([r["val_knn"] for r in rs]).items()}
    if "val_head" in rs[0]:
        out.update({f"valhead_{k}": v for k, v in traj_stats([r["val_head"] for r in rs]).items()})
    cv = [np.std(r["client_losses"]) / max(abs(np.mean(r["client_losses"])), 1e-12) for r in rs]
    out.update(client_loss_cv=float(np.mean(cv)), kappa_mean=float(np.mean([r["kappa"] for r in rs])),
               cos_mean=float(np.mean([r["pairwise_cos"] for r in rs])),
               abo_mean=float(np.mean([r["abo_err"] for r in rs])),
               abo_r1=float(rs[0]["abo_err"]),
               update_norm_mean=float(np.mean([r["update_norm"] for r in rs])))
    return out

DYN = {n: run_dynamics(h) for n, h in HISTORIES.items()}

def cell_agg(arm, a, key):
    cl = CELLS.get((arm, a), {})
    v = np.array([DYN[h["cfg"]["name"]].get(key, np.nan) for h in cl.values()], dtype=float)
    v = v[np.isfinite(v)]
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else np.nan, len(v)) if len(v) else (np.nan, np.nan, 0)

def seed_sd(arm, a, metric="probe"):
    cl = CELLS.get((arm, a), {})
    v = np.array([metric_of(h, metric) for h in cl.values()])
    return float(np.std(v, ddof=1)) if len(v) > 1 else np.nan

_rows = []
for arm in ARMS_ORDER:
    for a in SPLITS:
        if (arm, a) not in CELLS:
            continue
        row = {"method": OBJ_LABEL[arm], "split": alpha_label(a), "seeds": len(CELLS[(arm, a)])}
        for key, lab in [("valknn_backtrack", "val-kNN backtracking (pt)"),
                         ("valknn_max_drawdown", "val-kNN max drawdown (pt)"),
                         ("valknn_n_drops", "val-kNN drops"),
                         ("valhead_backtrack", "val-head backtracking (pt)"),
                         ("valhead_max_drawdown", "val-head max drawdown (pt)"),
                         ("client_loss_cv", "client-loss CV")]:
            row[lab] = cell_agg(arm, a, key)[0]
        row["probe seed s.d. (pt)"] = seed_sd(arm, a)
        _rows.append(row)
T_STAB = pd.DataFrame(_rows)
if len(T_STAB):
    save_table(T_STAB.round(3), "table7_stability",
        "Table 7. Training-trajectory stability, mean over seeds. Backtracking = total validation accuracy "
        "given back between consecutive rounds; drawdown = largest fall below the running best; client-loss "
        "CV = dispersion of the local losses of the clients in a round (scale-free, averaged over rounds). "
        "Lower is more stable. 'val-head' is the supervised federated classifier.", "stability",
        floatfmt="%.3f")
    display(T_STAB.round(3))

# ---- Table 8: client-update geometry during training and the aggregation error ----------------------
_rows = []
for arm in ARMS_ORDER:
    for a in SPLITS:
        if (arm, a) not in CELLS:
            continue
        _rows.append({"method": OBJ_LABEL[arm], "split": alpha_label(a),
                      "mean κ̂ (updates)": cell_agg(arm, a, "kappa_mean")[0],
                      "mean pairwise cos": cell_agg(arm, a, "cos_mean")[0],
                      "mean ‖Δ̄‖": cell_agg(arm, a, "update_norm_mean")[0],
                      "ε_ABO round 1": cell_agg(arm, a, "abo_r1")[0],
                      "ε_ABO mean": cell_agg(arm, a, "abo_mean")[0]})
T_UPD = pd.DataFrame(_rows)
if len(T_UPD):
    save_table(T_UPD, "table8_update_geometry",
        "Table 8. Client-update geometry measured every round on the LoRA-B coordinates (mean over rounds "
        "and seeds), and the aggregation–broadcast error ε of FedAvg on (A, B) — a second, independent "
        "measure of how far client adapters diverge. ε ≡ 0 for Fixed-A.", "updates", floatfmt="%.4f")
    display(T_UPD.round(4))

# ---- Table 9: per-client fairness ---------------------------------------------------------------
_rows = []
for arm in ARMS_ORDER:
    for a in SPLITS:
        cl = CELLS.get((arm, a))
        if not cl:
            continue
        f = [h["final"] for h in cl.values()]
        row = {"method": OBJ_LABEL[arm], "split": alpha_label(a),
               "per-client probe mean": 100 * np.mean([x["client_probe_mean"] for x in f]),
               "per-client probe s.d.": 100 * np.mean([x["client_probe_std"] for x in f]),
               "per-client probe worst-10%": 100 * np.mean([x["client_probe_worst10"] for x in f])}
        if "client_head_mean" in f[0]:
            row.update({"per-client head mean": 100 * np.mean([x["client_head_mean"] for x in f]),
                        "per-client head s.d.": 100 * np.mean([x["client_head_std"] for x in f]),
                        "per-client head worst-10%": 100 * np.mean([x["client_head_worst10"] for x in f])})
        _rows.append(row)
T_FAIR = pd.DataFrame(_rows)
if len(T_FAIR):
    save_table(T_FAIR.round(2), "table9_fairness",
        "Table 9. Per-client accuracy on test subsets matching each client's own label marginal "
        "(mean over seeds). The probe columns evaluate the learned representation; the head columns the "
        "classifier supervised FL would deploy.", "fairness")
    display(T_FAIR.round(2))

# ---- Table 10 / 11: hyper-parameters and seed-level raw results ----------------------------------
T_HP = save_table(pd.DataFrame([{"hyper-parameter": k, "value": str(v)} for k, v in asdict(BASE).items()
                                if k != "name"]), "table10_hyperparameters",
                  "Table 10. Hyper-parameters shared by every run (arms override only the objective, "
                  "fixed_a or fedprox_mu).", "hparams")
if len(RUNS):
    save_table(RUNS.round(3), "table11_all_runs", "Table 11. Every run, seed-level.", "allruns")
    display(RUNS[["run", "probe", "knn", "head", "probe_last", "selected_round", "diverged", "wall_min"]].round(2))
# %% [markdown]
# ## 10. Figures
#
# Each figure is saved as a 400-dpi PNG and a PDF in `figures/`. Splits run from IID (left) to the most skewed split (right); the tick labels give the mean effective number of classes per client.

# %%
def split_ticks(ax, splits=None):
    splits = splits or SPLITS
    ax.set_xticks(range(len(splits)))
    ax.set_xticklabels([f"{alpha_label(a)}\n({EFF_CLASSES.get(a, np.nan):.0f} cl.)" for a in splits])

def arm_cell_values(arm, a, metric="probe"):
    cl = CELLS.get((arm, a), {})
    return np.array([metric_of(h, metric) for h in cl.values()], dtype=float)

def err_from_ci(p):
    if p is None or "lo95" not in p:
        return None
    return np.array([[p["mean"] - p["lo95"]], [p["hi95"] - p["mean"]]]).clip(min=0)

def fig_safe(fn):
    try:
        fn()
    except Exception as e:
        print(f"[figure skipped] {fn.__name__}: {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        plt.close("all")

# ---- Figure 2: the gradient-geometry measurement (H3) ----------------------------------------------
def figure2_geometry():
    if not GEOM:
        print("[fig2] no geometry results"); return
    splits = [a for a in SPLITS if alpha_tag(a) in GEOM["per_alpha"]]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
    x = np.arange(len(splits))
    for j, o in enumerate(["mim", "sup"]):
        v = np.array([GEOM["per_alpha"][alpha_tag(a)][o]["kappa2_het"] for a in splits])
        lo = np.array([GEOM["per_alpha"][alpha_tag(a)][o].get("kappa2_het_lo", np.nan) for a in splits])
        hi = np.array([GEOM["per_alpha"][alpha_tag(a)][o].get("kappa2_het_hi", np.nan) for a in splits])
        off = (j - 0.5) * 0.12
        axs[0].errorbar(x + off, v, yerr=[np.clip(v - lo, 0, None), np.clip(hi - v, 0, None)], color=ARM_COLOR[o],
                        marker=ARM_MARK[o], capsize=4, label=OBJ_LABEL[o])
    axs[0].set_ylabel(r"$\kappa^2_{het}$ (noise-debiased)"); axs[0].set_title("(a) client-gradient dissimilarity")
    split_ticks(axs[0], splits); axs[0].legend()
    lams = ["0.0", "0.25", "0.5", "1.0", "2.0", "4.0", "inf"]
    xl = np.arange(len(lams))
    for i, a in enumerate([s for s in splits if np.isfinite(s)]):
        hy = GEOM["hybrid"].get(alpha_tag(a), {})
        axs[1].plot(xl, [hy.get(l, np.nan) for l in lams], marker="o", color=PALETTE[(i + 2) % len(PALETTE)],
                    label=alpha_label(a))
    axs[1].set_xticks(xl); axs[1].set_xticklabels(["0\n(CE)", ".25", ".5", "1", "2", "4", "∞\n(MIM)"])
    axs[1].set_xlabel(r"weight $\lambda$ of MIM in CE + $\lambda\cdot$MIM"); axs[1].set_ylabel(r"$\kappa^2_{het}$")
    axs[1].set_title("(b) hybrid objective"); axs[1].legend(title="split")
    fig.tight_layout()
    savefig(fig, "fig2_gradient_geometry",
            "Figure 2. Client-gradient dissimilarity at a common parameter point (pretrained backbone, B=0, "
            "shared A, heads warmed centrally). (a) noise-debiased κ² with leave-one-client-out jackknife 95% "
            "CIs. (b) κ² of CE + λ·MIM, computed from the same per-batch gradients.")

# ---- Figure 3: the headline — accuracy and heterogeneity penalty per split ---------------------------
def figure3_headline():
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.3))
    x = np.arange(len(SPLITS))
    for arm in ARMS_ORDER:
        m = [np.nanmean(arm_cell_values(arm, a)) if (arm, a) in CELLS else np.nan for a in SPLITS]
        s = [np.nanstd(arm_cell_values(arm, a), ddof=1) if len(arm_cell_values(arm, a)) > 1 else 0 for a in SPLITS]
        if np.all(np.isnan(m)):
            continue
        axs[0].errorbar(x, m, yerr=s, marker=ARM_MARK[arm], color=ARM_COLOR[arm], capsize=3, label=OBJ_LABEL[arm])
    axs[0].axhline(100 * ZERO_SHOT["probe_top1"], ls="--", color="grey", lw=1.2, label="zero-shot floor")
    axs[0].set_ylabel("linear-probe top-1 (%)"); axs[0].set_title("(a) representation quality")
    split_ticks(axs[0]); axs[0].legend(fontsize=8)

    nonid = [a for a in SPLITS if np.isfinite(a)]
    xn = np.arange(len(nonid)); arms = [a for a in ARMS_ORDER if any(("probe", a, s) in PEN for s in nonid)]
    wbar = 0.8 / max(len(arms), 1)
    for i, arm in enumerate(arms):
        for j, a in enumerate(nonid):
            p = PEN.get(("probe", arm, a))
            if p is None:
                continue
            axs[1].bar(j + (i - (len(arms) - 1) / 2) * wbar, p["mean"], wbar * 0.9, color=ARM_COLOR[arm],
                       yerr=err_from_ci(p), capsize=3, label=OBJ_LABEL[arm] if j == 0 else None)
    axs[1].axhline(0, color="k", lw=0.8)
    axs[1].set_xticks(xn); axs[1].set_xticklabels([alpha_label(a) for a in nonid])
    axs[1].set_ylabel(r"penalty $\Pi$ = acc(IID) − acc($\alpha$) (pt)")
    axs[1].set_title("(b) heterogeneity penalty, probe (95 % CI)"); axs[1].legend(fontsize=8)

    for arm in ["sup", "sup_mim", "sup_prox"]:
        m = [np.nanmean(arm_cell_values(arm, a, "head")) if (arm, a) in CELLS else np.nan for a in SPLITS]
        if np.all(np.isnan(m)):
            continue
        s = [np.nanstd(arm_cell_values(arm, a, "head"), ddof=1) if len(arm_cell_values(arm, a, "head")) > 1 else 0
             for a in SPLITS]
        axs[2].errorbar(x, m, yerr=s, marker=ARM_MARK[arm], color=ARM_COLOR[arm], capsize=3, label=OBJ_LABEL[arm])
    axs[2].set_ylabel("federated classifier top-1 (%)"); axs[2].set_title("(c) deployed supervised head")
    split_ticks(axs[2]); axs[2].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig3_headline",
            "Figure 3. (a) Linear-probe accuracy of the validation-selected global model vs client label skew "
            "(mean ± s.d. over seeds; dashed = unadapted backbone). (b) Heterogeneity penalty relative to the "
            "same arm's IID run, paired by seed, 95 % paired test-bootstrap CI. (c) Accuracy of the federated "
            "classifier head that supervised FL would deploy.")

# ---- Figure 4: per-round validation trajectories ----------------------------------------------------
def band(ax, arm, a, key, x_off=0):
    cl = CELLS.get((arm, a), {})
    curves = [[r[key] for r in h["rounds"] if key in r] for h in cl.values()]
    curves = [c for c in curves if len(c)]
    if not curves:
        return False
    L = min(len(c) for c in curves)
    Y = 100 * np.array([c[:L] for c in curves])
    xs = np.arange(1, L + 1)
    ax.plot(xs, Y.mean(0), color=ARM_COLOR[arm], marker=ARM_MARK[arm], ms=3.5, label=OBJ_LABEL[arm])
    if len(Y) > 1:
        ax.fill_between(xs, Y.min(0), Y.max(0), color=ARM_COLOR[arm], alpha=0.18, lw=0)
    return True

def figure4_trajectories():
    fig, axs = plt.subplots(2, len(SPLITS), figsize=(4.0 * len(SPLITS), 7.2), sharey="row", squeeze=False)
    for j, a in enumerate(SPLITS):
        for arm in ["mim", "sup", "sup_mim", "mim_fixA"]:
            band(axs[0, j], arm, a, "val_knn")
        for arm in ["sup", "sup_mim", "sup_prox"]:
            band(axs[1, j], arm, a, "val_head")
        axs[0, j].axhline(100 * ZERO_SHOT["knn_top1"], ls="--", color="grey", lw=1)
        axs[0, j].set_title(alpha_label(a)); axs[1, j].set_xlabel("communication round")
    axs[0, 0].set_ylabel("server val kNN top-1 (%)"); axs[1, 0].set_ylabel("server val head top-1 (%)")
    h0, l0 = axs[0, 0].get_legend_handles_labels(); h1, l1 = axs[1, 0].get_legend_handles_labels()
    if h0:
        axs[0, -1].legend(h0, l0, fontsize=8, loc="lower right")
    if h1:
        axs[1, -1].legend(h1, l1, fontsize=8, loc="lower right")
    fig.tight_layout()
    savefig(fig, "fig4_trajectories",
            "Figure 4. Per-round accuracy of the aggregated model on the server-only validation split (line = "
            "mean over seeds, band = min–max). Top: kNN on the representation (all arms, same readout; dashed = "
            "zero-shot). Bottom: the federated classifier of the supervised arms.")

# ---- Figure 5: stability summary (H4) --------------------------------------------------------------
def figure5_stability():
    keys = [("valknn_backtrack", "val-kNN backtracking (pt)"), ("valknn_max_drawdown", "val-kNN max drawdown (pt)"),
            ("valhead_backtrack", "val-head backtracking (pt)"), ("client_loss_cv", "client-loss CV")]
    fig, axs = plt.subplots(1, len(keys), figsize=(4.2 * len(keys), 4.0))
    x = np.arange(len(SPLITS))
    arms = [a for a in ARMS_ORDER if any((a, s) in CELLS for s in SPLITS)]
    wbar = 0.8 / max(len(arms), 1)
    for ax, (k, lab) in zip(axs, keys):
        for i, arm in enumerate(arms):
            vals = [cell_agg(arm, a, k) for a in SPLITS]
            m = np.array([v[0] for v in vals]); s = np.array([v[1] if np.isfinite(v[1]) else 0 for v in vals])
            if np.all(np.isnan(m)):
                continue
            ax.bar(x + (i - (len(arms) - 1) / 2) * wbar, np.nan_to_num(m), wbar * 0.9, yerr=s, capsize=2,
                   color=ARM_COLOR[arm], label=OBJ_LABEL[arm])
        ax.set_title(lab, fontsize=10.5); split_ticks(ax)
    axs[0].legend(fontsize=7.5)
    fig.tight_layout()
    savefig(fig, "fig5_stability",
            "Figure 5. Trajectory stability (mean ± s.d. over seeds). Backtracking = validation accuracy given "
            "back between consecutive rounds; drawdown = largest fall below the running best; client-loss CV = "
            "dispersion of the participating clients' local losses. Lower is more stable.")

# ---- Figure 6: update geometry during training -------------------------------------------------------
def figure6_updates():
    keys = [("kappa_mean", r"mean $\hat\kappa$ of client updates"), ("cos_mean", "mean pairwise cosine of updates"),
            ("abo_mean", r"mean $\varepsilon_{ABO}$ (FedAvg on A,B)")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
    x = np.arange(len(SPLITS))
    for ax, (k, lab) in zip(axs, keys):
        for arm in ARMS_ORDER:
            m = [cell_agg(arm, a, k)[0] for a in SPLITS]
            if np.all(np.isnan(m)):
                continue
            ax.plot(x, m, marker=ARM_MARK[arm], color=ARM_COLOR[arm], label=OBJ_LABEL[arm])
        ax.set_title(lab, fontsize=10.5); split_ticks(ax)
    if np.nanmax([cell_agg(a, s, "abo_mean")[0] for a in ARMS_ORDER for s in SPLITS] + [0]) > 0:
        axs[2].set_yscale("symlog", linthresh=1e-4)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    savefig(fig, "fig6_update_geometry",
            "Figure 6. Geometry of the multi-step client updates during training (LoRA-B coordinates, mean over "
            "rounds and seeds) and the aggregation–broadcast error of averaging A and B separately.")

# ---- Figure 7: per-client fairness at the most skewed split -----------------------------------------
def figure7_fairness():
    a = A_LO
    data, labels, colors = [], [], []
    zs = []
    for sd in SEEDS:
        zs += ZERO_SHOT.get("client_probe_by_partition", {}).get(f"{alpha_tag(a)}_s{sd}", [])
    if zs:
        data.append(100 * np.array(zs)); labels.append("zero-shot\n(probe)"); colors.append("#999999")
    for arm in ARMS_ORDER:
        cl = CELLS.get((arm, a))
        if not cl:
            continue
        v = np.concatenate([np.array(h["final"].get("client_probe", [])) for h in cl.values()])
        if len(v):
            data.append(100 * v); labels.append(OBJ_LABEL[arm].replace(" (", "\n(") + "\nprobe"); colors.append(ARM_COLOR[arm])
        if arm in ("sup", "sup_mim", "sup_prox"):
            v = np.concatenate([np.array(h["final"].get("client_head", [])) for h in cl.values()])
            if len(v):
                data.append(100 * v); labels.append(OBJ_LABEL[arm].replace(" (", "\n(") + "\nhead")
                colors.append(ARM_COLOR[arm])
    if not data:
        print("[fig7] no per-client data"); return
    fig, ax = plt.subplots(figsize=(max(7, 1.5 * len(data)), 4.4))
    bp = ax.boxplot(data, patch_artist=True, widths=0.6, showfliers=True)
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.45)
    ax.set_xticks(range(1, len(labels) + 1)); ax.set_xticklabels(labels, fontsize=7.5)
    ax.set_ylabel("per-client top-1 (%)"); ax.set_title(f"per-client accuracy at {alpha_label(a)} (all seeds pooled)")
    fig.tight_layout()
    savefig(fig, "fig7_client_fairness",
            f"Figure 7. Distribution over clients of accuracy on test subsets matching each client's label "
            f"marginal at {alpha_label(a)} ({len(SEEDS)} seeds pooled). 'probe' = the learned representation; "
            f"'head' = the federated classifier.")

# ---- Figure 8: measured cost (P1) -----------------------------------------------------------------
def figure8_cost():
    if BENCH is None or not len(BENCH):
        print("[fig8] no benchmark"); return
    fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
    labs = [textwrap.fill(s, 22) for s in BENCH["configuration"]]
    y = np.arange(len(labs))
    axs[0].barh(y, BENCH["peak_mb"] / 1024, color=PALETTE[0]); axs[0].axvline(16, ls="--", color="grey")
    axs[0].axvline(4, ls=":", color="k"); axs[0].set_xlabel("peak GPU memory (GB)")
    axs[1].barh(y, BENCH["img_s"], color=PALETTE[2]); axs[1].set_xlabel("training throughput (img/s, one T4)")
    axs[2].barh(y, BENCH["uplink_mb"], color=PALETTE[1], label="LoRA + aggregated heads")
    axs[2].barh(y, BENCH["uplink_lora_mb"], color=PALETTE[4], label="LoRA only")
    axs[2].set_xscale("log"); axs[2].axvline(VIT_FULL_MB, ls="--", color="grey", label="full fine-tuning")
    axs[2].set_xlabel("uplink per client per round (MB, fp16)"); axs[2].legend(fontsize=8)
    for ax in axs:
        ax.set_yticks(y); ax.set_yticklabels(labs if ax is axs[0] else [], fontsize=8); ax.invert_yaxis()
    fig.tight_layout()
    savefig(fig, "fig8_cost",
            "Figure 8. Measured cost of one local step (batch 64 unless stated) on a single T4: peak memory "
            "(dotted = 4 GB edge budget, dashed = T4 capacity), throughput, and uplink payload vs full fine-tuning.")

# ---- Figure 9: stabilisers at the most skewed split -------------------------------------------------
def figure9_stabilisers():
    arms = [a for a in ["sup", "sup_prox", "sup_mim", "mim", "mim_fixA"] if ("probe", a, A_LO) in PEN]
    if not arms:
        print("[fig9] no penalties"); return
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, metric in zip(axs, ["probe", "head"]):
        xs, done = 0, []
        for arm in arms:
            p = PEN.get((metric, arm, A_LO))
            if p is None:
                continue
            ax.bar(xs, p["mean"], 0.65, color=ARM_COLOR[arm], yerr=err_from_ci(p), capsize=4)
            ax.scatter([xs] * len(p["per_seed"]), p["per_seed"], color="k", s=12, zorder=3)
            done.append(OBJ_LABEL[arm].replace(" (", "\n(")); xs += 1
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(done))); ax.set_xticklabels(done, fontsize=8)
        ax.set_ylabel(f"penalty at {alpha_label(A_LO)} (pt)")
        ax.set_title("representation (linear probe)" if metric == "probe" else "deployed federated classifier")
    fig.tight_layout()
    savefig(fig, "fig9_stabilisers",
            f"Figure 9. Heterogeneity penalty at {alpha_label(A_LO)} for supervised FedLoRA, the FedProx drift "
            f"correction (μ={FEDPROX_MU}, referenced to supervised IID), the CE + MIM hybrid and FedMIM-LoRA. "
            f"Bars = mean over seeds with 95 % paired-bootstrap CI; dots = individual seeds.")

# ---- Figure 10: accuracy vs effective number of classes ---------------------------------------------
def figure10_slope():
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for arm in ["mim", "sup", "sup_mim"]:
        d = _pts[_pts.arm == arm] if len(_pts) else _pts
        if not len(d):
            continue
        ax.scatter(10 ** d.x, d.y, color=ARM_COLOR[arm], marker=ARM_MARK[arm], alpha=0.8,
                   label=OBJ_LABEL[arm] + (f"  (slope {SLOPES[arm]['slope']:+.2f} ± {SLOPES[arm]['se']:.2f})"
                                           if arm in SLOPES else ""))
        if arm in SLOPES and d.x.nunique() >= 2:
            b, _, _ = ols(np.c_[np.ones(len(d)), d.x.values], d.y.values)
            xx = np.linspace(d.x.min(), d.x.max(), 50)
            ax.plot(10 ** xx, b[0] + b[1] * xx, color=ARM_COLOR[arm], lw=1.2)
    ax.set_xscale("log"); ax.set_xlabel("effective number of classes per client (log)")
    ax.set_ylabel("linear-probe top-1 (%)"); ax.legend(fontsize=8)
    ax.axhline(100 * ZERO_SHOT["probe_top1"], ls="--", color="grey", lw=1)
    fig.tight_layout()
    savefig(fig, "fig10_sensitivity_slope",
            "Figure 10. Probe accuracy of every run against the mean effective number of classes per client "
            "(exp of the label entropy) of its partition. Slopes are points per decade (OLS, ± s.e.); a flat line "
            "means insensitivity to heterogeneity.")

for _f in [figure2_geometry, figure3_headline, figure4_trajectories, figure5_stability, figure6_updates,
           figure7_fairness, figure8_cost, figure9_stabilisers, figure10_slope]:
    fig_safe(_f)
# %% [markdown]
# ## 11. Hypotheses
#
# Each hypothesis is checked against the rule in the table at the top. A result is inconclusive only if the runs it needs are missing.

# %%
S_, NS_, INC_ = "SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"
VERDICTS: Dict[str, Dict[str, str]] = {}

def verdict(key, status, evidence, rule):
    VERDICTS[key] = dict(status=status, evidence=evidence, rule=rule)

def ci_str(p, lo="lo95", hi="hi95"):
    return f"[{p[lo]:+.2f}, {p[hi]:+.2f}]" if p and lo in p else "[n/a]"

def paired_diff(arm_a, arm_b, a, metric="probe"):
    """acc(arm_a) − acc(arm_b) at split a, paired by seed and test resample."""
    ca, cb = CELLS.get((arm_a, a), {}), CELLS.get((arm_b, a), {})
    seeds = sorted(set(ca) & set(cb))
    if not seeds:
        return None
    ps, boots = [], []
    for sd in seeds:
        ps.append(metric_of(ca[sd], metric) - metric_of(cb[sd], metric))
        x, y = correct_vec(ca[sd], metric), correct_vec(cb[sd], metric)
        if x is not None and y is not None:
            boots.append(boot_mean_acc(x) - boot_mean_acc(y))
    out = dict(mean=float(np.mean(ps)), per_seed=ps, n=len(seeds))
    if boots:
        B = np.mean(boots, axis=0)
        out.update(lo95=float(np.percentile(B, 2.5)), hi95=float(np.percentile(B, 97.5)))
    return out

LO = alpha_label(A_LO)

# ---- H1: MIM invariance ----------------------------------------------------------------------------
_r = "90% CI of Π_MIM(α_min) inside ±0.5 pt"
p = PEN.get(("probe", "mim", A_LO))
if p is None or "lo90" not in p:
    verdict("H1", INC_, "MIM runs at IID and the most skewed split not both available", _r)
else:
    others = "; ".join(f"{alpha_label(a)}: {PEN[('probe','mim',a)]['mean']:+.2f}" for a in SPLITS
                       if np.isfinite(a) and a != A_LO and ("probe", "mim", a) in PEN)
    verdict("H1", S_ if (p["lo90"] > -0.5 and p["hi90"] < 0.5) else NS_,
            f"Π_MIM({LO}) = {p['mean']:+.2f} pt, 90% CI [{p['lo90']:+.2f}, {p['hi90']:+.2f}], {p['n']} seed(s), "
            f"per seed {np.round(p['per_seed'], 2).tolist()}" + (f"; other splits {others}" if others else ""), _r)

# ---- H2: difference-in-differences -------------------------------------------------------------------
_r = "DiD = Π_SUP − Π_MIM > 0 in every seed and 95% CI excludes 0 (probe)"
d = DID.get(("probe", A_LO))
if d is None:
    verdict("H2", INC_, "the four cells (MIM/SUP × IID/α_min) are not all available", _r)
else:
    ok = d["sign_pos"] == d["n"] and d.get("lo95", -np.inf) > 0
    ps_ = PEN.get(("probe", "sup", A_LO)); ph_ = PEN.get(("head", "sup", A_LO))
    extra = ""
    if ps_ and p:
        extra = f"; Π_SUP(probe) = {ps_['mean']:+.2f} vs Π_MIM = {p['mean']:+.2f}"
    if ph_:
        extra += f"; Π_SUP(deployed head) = {ph_['mean']:+.2f} {ci_str(ph_)}"
    verdict("H2", S_ if ok else NS_,
            f"DiD({LO}) = {d['mean']:+.2f} pt, 95% CI {ci_str(d)}, positive in {d['sign_pos']}/{d['n']} seeds" + extra
            + (" (single seed: 'every seed' is trivially met)" if d["n"] < 2 else ""), _r)

# ---- H3: client-gradient dissimilarity ---------------------------------------------------------------
_r = "κ²(MIM) < κ²(SUP) at every non-IID α; paired 95% CI of κ²(SUP) − κ²(MIM) excludes 0 at α_min"
if not GEOM:
    verdict("H3", INC_, "geometry measurement did not run", _r)
else:
    ga = [(a, GEOM["per_alpha"][alpha_tag(a)]) for a in SPLITS if np.isfinite(a) and alpha_tag(a) in GEOM["per_alpha"]]
    all_lower = all(g["mim"]["kappa2_het"] < g["sup"]["kappa2_het"] for _, g in ga)
    p0 = GEOM["per_alpha"].get(alpha_tag(A_LO), {}).get("paired", {})
    sig = p0.get("diff_lo", -np.inf) > 0
    ev = "; ".join(f"{alpha_label(a)}: MIM {g['mim']['kappa2_het']:.3f} vs SUP {g['sup']['kappa2_het']:.3f}, "
                   f"SUP − MIM {g['paired']['diff']:+.3f} [{g['paired']['diff_lo']:+.3f}, {g['paired']['diff_hi']:+.3f}]"
                   + (f", ratio {g['paired']['ratio']:.2f}x" if "ratio" in g["paired"] else "")
                   for a, g in ga if "paired" in g)
    gi = GEOM["per_alpha"].get("iid")
    if gi:
        ev += f"; IID (should be ≈0): MIM {gi['mim']['kappa2_het']:.3f}, SUP {gi['sup']['kappa2_het']:.3f}"
    verdict("H3", S_ if (ga and all_lower and sig) else NS_, ev, _r)

# ---- H4: smoother trajectories --------------------------------------------------------------------------
_r = "val-kNN backtracking(MIM) < backtracking(SUP) at α_min (same readout for both)"
bm, bs = cell_agg("mim", A_LO, "valknn_backtrack"), cell_agg("sup", A_LO, "valknn_backtrack")
if not (bm[2] and bs[2]):
    verdict("H4", INC_, "MIM or SUP trajectory at α_min missing", _r)
else:
    hb = cell_agg("sup", A_LO, "valhead_backtrack")
    bmi, bsi = cell_agg("mim", IID, "valknn_backtrack"), cell_agg("sup", IID, "valknn_backtrack")
    st = S_ if bm[0] < bs[0] else (INC_ if abs(bm[0] - bs[0]) < 1e-9 else NS_)
    verdict("H4", st,
            f"backtracking at {LO}: MIM {bm[0]:.2f} pt vs SUP {bs[0]:.2f} pt (SUP deployed head {hb[0]:.2f} pt); "
            f"at IID: MIM {bmi[0]:.2f} vs SUP {bsi[0]:.2f}; max drawdown MIM "
            f"{cell_agg('mim', A_LO, 'valknn_max_drawdown')[0]:.2f} vs SUP {cell_agg('sup', A_LO, 'valknn_max_drawdown')[0]:.2f}; "
            f"probe seed s.d. MIM {seed_sd('mim', A_LO):.2f} vs SUP {seed_sd('sup', A_LO):.2f}"
            + (" (tie: both trajectories monotone)" if st == INC_ else ""), _r)

# ---- H5: hybrid ------------------------------------------------------------------------------------------
_r = "Π(CE+MIM) < Π(CE) at α_min on the deployed federated classifier"
phs, phh = PEN.get(("head", "sup", A_LO)), PEN.get(("head", "sup_mim", A_LO))
if phs is None or phh is None:
    verdict("H5", INC_, "hybrid or supervised head results at IID/α_min missing", _r)
else:
    dh = did(A_LO, "head", arm_a="sup", arm_b="sup_mim")
    gain = paired_diff("sup_mim", "sup", A_LO, "head")
    pps, pph = PEN.get(("probe", "sup", A_LO)), PEN.get(("probe", "sup_mim", A_LO))
    verdict("H5", S_ if phh["mean"] < phs["mean"] else NS_,
            f"head penalty: CE {phs['mean']:+.2f} → CE+MIM {phh['mean']:+.2f} pt"
            + (f" (difference {dh['mean']:+.2f}, 95% CI {ci_str(dh)}, {dh['sign_pos']}/{dh['n']} seeds)" if dh else "")
            + (f"; probe penalty CE {pps['mean']:+.2f} → CE+MIM {pph['mean']:+.2f}" if pps and pph else "")
            + (f"; head accuracy at {LO}: hybrid − CE = {gain['mean']:+.2f} pt {ci_str(gain)}" if gain else ""), _r)

# ---- P1: cost ----------------------------------------------------------------------------------------
_r = "peak < 4 GB at batch 64; LoRA uplink ≥100× and total uplink ≥25× below full fine-tuning"
if BENCH is None or not len(BENCH) or not np.isfinite(BENCH.iloc[0]["peak_mb"]):
    verdict("P1", INC_, "benchmark unavailable (no GPU)", _r)
else:
    b0 = BENCH.iloc[0]
    rl, rt = VIT_FULL_MB / b0["uplink_lora_mb"], VIT_FULL_MB / b0["uplink_mb"]
    ok = b0["peak_mb"] < 4096 and rl >= 100 and rt >= 25
    mn = BENCH.loc[BENCH["peak_mb"].idxmin()]
    verdict("P1", S_ if ok else NS_,
            f"{b0['configuration']}: peak {b0['peak_mb']/1024:.2f} GB, {b0['img_s']:.0f} img/s on one T4, uplink "
            f"{b0['uplink_mb']:.2f} MB ({rt:.0f}× below full FT; LoRA alone {b0['uplink_lora_mb']:.2f} MB, {rl:.0f}×); "
            f"cheapest measured configuration '{mn['configuration']}' peaks at {mn['peak_mb']:.0f} MB", _r)

# ---- overall -----------------------------------------------------------------------------------------
st = {k: v["status"] for k, v in VERDICTS.items()}
if st.get("H1") == S_ and st.get("H2") == S_ and st.get("H3") == S_:
    OVERALL = "ESTABLISHED"
elif st.get("H1") == S_ and (st.get("H2") == S_ or st.get("H3") == S_):
    OVERALL = "PARTIALLY ESTABLISHED"
elif st.get("H1") == S_:
    OVERALL = "MIM IS INSENSITIVE TO LABEL SKEW (H1 ONLY)"
else:
    OVERALL = "NOT ESTABLISHED"

T_VERDICT = pd.DataFrame([{"id": k, "verdict": v["status"], "rule": v["rule"], "evidence": v["evidence"]}
                          for k, v in VERDICTS.items()])
save_table(T_VERDICT, "table12_verdicts", "Table 12. Hypotheses, decision rules and results.",
           "verdicts")
print("=" * 100)
print(f"MIM keeps federated LoRA stable across heterogeneous clients:  {OVERALL}")
print("=" * 100)
for k, v in VERDICTS.items():
    print(f"\n[{k}] {v['status']}\n    rule    : {v['rule']}\n    evidence: {textwrap.fill(v['evidence'], 90, subsequent_indent=' ' * 14)}")

# %% [markdown]
# ## 12. Summary file and artifacts

# %%
_done = set(HISTORIES)
_skipped = [make_run(*p).name for p in PLAN if make_run(*p).name not in _done]
_md = ["# FedMIM-LoRA results", "",
       f"Preset `{PRESET}`, {len(HISTORIES)} completed runs, {elapsed_min():.0f} min.", "",
       f"Overall: {OVERALL}", "", "| id | result | rule | evidence |", "|---|---|---|---|"]
_md += [f"| {k} | {v['status']} | {v['rule']} | {v['evidence']} |" for k, v in VERDICTS.items()]
_md += ["", "## Unadapted backbone", "",
        f"kNN {fmt_pct(ZERO_SHOT['knn_top1'])}%, probe {fmt_pct(ZERO_SHOT['probe_top1'])}%, "
        f"few-shot {fmt_pct(ZERO_SHOT['fewshot_top1'])}%", ""]
_md += ["## Runs not completed", ""] + ([f"- {n}" for n in _skipped] or ["- none"])
_md += ["", "## Figures", ""] + [f"- `{k}`: {v}" for k, v in FIG_CAPTIONS.items()]
_md += ["", "## Tables", ""] + [f"- `{k}`: {v}" for k, v in TAB_CAPTIONS.items()]
(OUT / "results_summary.md").write_text("\n".join(_md))
(DIRS["metrics"] / "verdicts.json").write_text(json.dumps({"overall": OVERALL, "verdicts": VERDICTS}, indent=1))
print(f"runs completed: {len(HISTORIES)}/{len(PLAN)}; not completed: {_skipped if _skipped else 'none'}")
print(f"wall-clock: {elapsed_min():.0f} min of {TIME_BUDGET_H*60:.0f} budgeted")

ZIP_PATH = ROOT / "fedmim_lora_outputs.zip"
with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.rglob("*")):
        if f.is_file():
            zf.write(f, f.relative_to(ROOT))
_sizes = {k: sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) / 2 ** 20 for k, d in DIRS.items()}
print("artifacts:", ", ".join(f"{k} {v:.1f} MB" for k, v in _sizes.items()))
print(f"zipped -> {ZIP_PATH} ({ZIP_PATH.stat().st_size/2**20:.1f} MB)")
