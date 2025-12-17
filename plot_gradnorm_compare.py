# plot_gradnorm_compare_multi_pub_singlelegend.py
# One compact legend (two columns) listing warmup + per-run raw and EMA entries.
# Long labels on the plot legend, short labels in the table.
# No separate table-only export.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

# ---------- EDIT THIS BLOCK ----------
RUNS = [
    {"csv": "runs/train/mvaaod/stardnet_mvaaod-gd/grad_norm_rank-inonoutoff.csv",
     "plot_label": "Run A  all-on",        "table_label": "Run A", "color": "#009E73"},
    {"csv": "runs/train/mvaaod/stardnet_mvaaod-gd/grad_norm_rank-inoffouton.csv",
     "plot_label": "Run B  in-off out-on", "table_label": "Run B", "color": "#D55E00"},
    {"csv": "runs/train/mvaaod/stardnet_mvaaod-gd/grad_norm_rank-all.csv",
     "plot_label": "Run C  in-on out-off", "table_label": "Run C", "color": "#0072B2"},
    {"csv": "runs/train/mvaaod/stardnet_mvaaod-gd/grad_norm_rank-inoutoff.csv",
     "plot_label": "Run D  all-off",       "table_label": "Run D", "color": "#CC79A7"},
]
steps_per_epoch = 233
warmup_epochs  = 3
ema_span       = 200          # EMA span in steps
roll_win       = 200          # rolling std window for ±1σ bands
plateau_last_k = 1000         # stats over last K steps
draw_bands     = True
out_base       = "gradnorm_compare_multi_with_table_pub_singlelegend"
dpi_png        = 300
fig_w, fig_h   = 14.0, 9.2    # tall so xlabel never touches the table
# -------------------------------------

# Typography
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "axes.linewidth": 0.8,
    "grid.color": "0.85",
    "grid.linewidth": 0.6,
})

def load_csv(p):
    df = pd.read_csv(p).sort_values("iter").reset_index(drop=True)
    df["iter"] = pd.to_numeric(df["iter"], errors="coerce")
    df["grad_norm"] = pd.to_numeric(df["grad_norm"], errors="coerce")
    return df.dropna(subset=["iter", "grad_norm"])

def ema_np(x, span):
    x = np.asarray(x, float)
    a = 2.0 / (span + 1.0)
    y = np.empty_like(x); y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = a*x[i] + (1-a)*y[i-1]
    return y

def metrics(df):
    steps = df["iter"].to_numpy()
    g     = df["grad_norm"].to_numpy(float)
    gE    = ema_np(g, ema_span)
    warm_end  = warmup_epochs * steps_per_epoch
    warm_mask = steps <= warm_end
    peak      = float(g[warm_mask].max()) if warm_mask.any() else np.nan
    idx_end   = np.where(warm_mask)[0][-1] if warm_mask.any() else -1
    ema_end   = float(gE[idx_end]) if idx_end >= 0 else np.nan

    start_idx = max(0, len(g) - plateau_last_k)
    plate     = g[start_idx:]
    mu        = float(plate.mean()) if len(plate) else np.nan
    sd        = float(plate.std(ddof=1)) if len(plate) > 1 else np.nan
    cv        = float(sd / (mu + 1e-12)) if np.isfinite(mu) else np.nan

    roll = pd.Series(g, index=steps).rolling(roll_win, min_periods=max(5, roll_win//10)).std()

    return dict(steps=steps, g=g, ema=gE, roll_std=roll.to_numpy(),
                peak=peak, ema_end=ema_end, mu=mu, sd=sd, cv=cv)

# Compute metrics
for r in RUNS:
    r["m"] = metrics(load_csv(r["csv"]))

# ---- Layout: extra vertical space so x label cannot touch the table
fig = plt.figure(figsize=(fig_w, fig_h))
gs  = GridSpec(2, 1, height_ratios=[3.4, 1.25], hspace=0.32, figure=fig)
ax  = fig.add_subplot(gs[0])
tax = fig.add_subplot(gs[1]); tax.axis("off")

# Warm-up shading and epoch guides
warm_end = warmup_epochs * steps_per_epoch
ax.axvspan(0, warm_end, color="0.2", alpha=0.06, zorder=0)
max_step = max(r["m"]["steps"].max() for r in RUNS)
for e in range(1, int(np.ceil(max_step/steps_per_epoch)) + 1):
    ax.axvline(e * steps_per_epoch, color="0.8", lw=0.6, ls=(0, (3, 3)), zorder=0)

# Plot each run
for r in RUNS:
    m, c, lab = r["m"], r["color"], r["plot_label"]
    ax.plot(m["steps"], m["g"],   lw=1.0, alpha=0.28, color=c, zorder=2)
    ax.plot(m["steps"], m["ema"], lw=2.2, color=c, zorder=3)
    if draw_bands and len(m["roll_std"]) == len(m["steps"]):
        ax.fill_between(m["steps"], m["ema"]-m["roll_std"], m["ema"]+m["roll_std"],
                        color=c, alpha=0.10, lw=0, zorder=1)

# ax.set_title("Gradient norm per optimizer step")
ax.set_xlabel("Optimizer step", labelpad=22)
ax.set_ylabel("Global L2 grad norm")
ax.grid(True, axis="y", which="both", alpha=0.35)
ax.grid(False, axis="x")

# ---- Build a single, compact legend (two columns)
legend_handles = []
legend_labels  = []

# Warm-up patch
warm_patch = Rectangle((0,0), 1, 1, fc="0.2", ec="0.2", alpha=0.06)
legend_handles.append(warm_patch)
legend_labels.append(f"Warm up 0 to {warmup_epochs} ep")

# For each run, add raw and EMA entries with long labels like the attachment
for r in RUNS:
    color = r["color"]
    # raw
    h_raw = Line2D([0], [0], color=color, lw=1.4, alpha=0.28)
    legend_handles.append(h_raw)
    legend_labels.append(f"{r['plot_label']} raw")
    # EMA
    h_ema = Line2D([0], [0], color=color, lw=2.2)
    legend_handles.append(h_ema)
    legend_labels.append(f"{r['plot_label']} EMA span={ema_span}")

# Place the legend at the top, centered, with two columns like your screenshot
leg = ax.legend(
    handles=legend_handles, labels=legend_labels,
    loc="upper center", bbox_to_anchor=(0.5, 0.985),
    ncol=2, frameon=True, framealpha=0.9, fancybox=True,
    borderpad=0.4, columnspacing=1.2, handlelength=2.0, handletextpad=0.8,
    title=None
)
# Light border tint
leg.get_frame().set_edgecolor("0.75")

# ---- Bottom table (short labels, wider first column)
headers = ["Metric"] + [r["table_label"] for r in RUNS]
rows = [
    ["Peak warm up"]                       + [f"{r['m']['peak']:.1f}"                    for r in RUNS],
    ["EMA at warm up end"]                 + [f"{r['m']['ema_end']:.1f}"                 for r in RUNS],
    [f"Plateau mean ± std  last {plateau_last_k}"]
                                           + [f"{r['m']['mu']:.1f} ± {r['m']['sd']:.1f}" for r in RUNS],
    ["Plateau coefficient of variation"]   + [f"{r['m']['cv']:.3f}"                      for r in RUNS],
]
n_runs = len(RUNS)
col_widths = [0.40] + [ (0.60 / n_runs) ] * n_runs

tbl = tax.table(cellText=rows, colLabels=headers, cellLoc="center",
                colLoc="center", edges="closed",
                bbox=[0.02, 0.18, 0.96, 0.74], colWidths=col_widths)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.06, 1.18)
for (ri, ci), cell in tbl.get_celld().items():
    cell.set_edgecolor("0.82")
    if ri == 0:
        cell.set_text_props(weight="bold")
        cell.set_facecolor((1, 1, 1, 0.98))
        cell.set_edgecolor("0.60")

tax.text(0.02, 0.02,
         "Lower peak and EMA at warm up end with lower plateau coefficient of variation indicate higher stability.",
         fontsize=9, transform=tax.transAxes)

# ---- Save main figure only (PNG + PDF)
for ext in ("png", "pdf"):
    fig.savefig(f"{out_base}.{ext}", dpi=dpi_png if ext=="png" else None,
                bbox_inches="tight")
print("Saved:", Path(f"{out_base}.png").resolve())
print("Saved:", Path(f"{out_base}.pdf").resolve())
