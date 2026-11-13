"""Builds the report's data figures from results/results.json (logged values only)."""
import json, os, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.abspath(__file__))
R = json.load(open(os.path.join(ROOT, "..", "results", "results.json")))
H, S = R["histories"], R["scores"]
OUT = os.path.join(ROOT, "figures")

# Semantic palette (same hexes as the LaTeX preamble)
C_DATA, C_MODEL, C_BASE, C_KEY, C_NEU = "#2E7D6E", "#1F3A5F", "#8A97A8", "#C8553D", "#5B6472"
STYLE = {  # family -> colour, bidirectional -> dashed, marker for greyscale
    "RNN": (C_BASE, "-", "o"), "BiRNN": (C_BASE, "--", "s"),
    "LSTM": (C_MODEL, "-", "^"), "BiLSTM": (C_MODEL, "--", "v"),
    "Transformer": (C_KEY, "-", "D"), "RNN + GloVe": (C_DATA, "-", "P"),
}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": C_NEU, "axes.labelcolor": "#222", "xtick.color": C_NEU, "ytick.color": C_NEU})

# Fig: validation loss curves
fig, ax = plt.subplots(figsize=(8, 4.6))
for m, (c, ls, mk) in STYLE.items():
    v = H[m]["val_loss"]; ep = range(1, len(v) + 1)
    ax.plot(ep, v, color=c, ls=ls, lw=2.2 if m == "Transformer" else 1.6, marker=mk, markevery=7, ms=5, label=m)
ax.set_xlabel("Epoch"); ax.set_ylabel("Validation loss (label-smoothed cross-entropy)")
ax.set_xlim(1, 51); ax.set_ylim(2.65, 4.8)
ax.legend(frameon=False, fontsize=9.5, loc="upper right", ncol=2, handlelength=3)
best = min(H["LSTM"]["val_loss"])
ax.annotate("Transformer still falling\nat epoch 50 (best = last epoch)", xy=(50, H["Transformer"]["val_loss"][-1]),
            xytext=(25, 3.45), color=C_KEY, fontsize=10, arrowprops=dict(arrowstyle="->", color=C_KEY))
ax.annotate("BiLSTM bottoms out at epoch 29", xy=(29, H["BiLSTM"]["val_loss"][28]), xytext=(9, 2.7),
            color=C_MODEL, fontsize=10, arrowprops=dict(arrowstyle="->", color=C_MODEL))
fig.savefig(os.path.join(OUT, "val_loss.pdf"), dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig: test scores, dot plot per metric
metrics = ["BLEU", "ROUGE-L", "METEOR", "chrF"]
order = ["RNN", "BiRNN", "RNN + GloVe", "BiLSTM", "LSTM", "Transformer"]
fig, axes = plt.subplots(1, 4, figsize=(11, 3.6), sharey=True)
for ax, met in zip(axes, metrics):
    for i, m in enumerate(order):
        c, ls, mk = STYLE[m]
        val = S[m][met]
        ax.hlines(i, 0, val, color=c, lw=1.2, alpha=.5)
        ax.plot(val, i, marker=mk, color=c, ms=8, mfc=c if ls == "-" else "white", mew=1.8)
        ax.text(val + 1, i, f"{val:.1f}", va="center", fontsize=9.5,
                color=C_KEY if m == "Transformer" else "#222", fontweight="bold" if m == "Transformer" else "normal")
    lo = min(S[m][met] for m in order)
    ax.set_xlim(lo - 6, max(S[m][met] for m in order) + 7)
    ax.set_title(met + " (0–100)", fontsize=11, color="#222"); ax.set_xlabel("Score")
axes[0].set_yticks(range(len(order))); axes[0].set_yticklabels(order)
for t, m in zip(axes[0].get_yticklabels(), order):
    t.set_color(STYLE[m][0])
fig.tight_layout()
fig.savefig(os.path.join(OUT, "scores.pdf"), dpi=200, bbox_inches="tight"); plt.close(fig)

# Fig: RNN random vs GloVe
thr = R["part4"]["common_val_loss"]; conv = R["part4"]["epochs_to_common_val_loss"]
fig, ax = plt.subplots(figsize=(8, 4.3))
for m, lab in [("RNN", "RNN, random embeddings"), ("RNN + GloVe", "RNN + GloVe embeddings")]:
    c, ls, mk = STYLE[m]
    v, t = H[m]["val_loss"], H[m]["train_loss"]; ep = range(1, len(v) + 1)
    ax.plot(ep, v, color=c, lw=2, marker=mk, markevery=7, ms=5, label=lab + " (validation)")
    ax.plot(ep, t, color=c, lw=1.3, ls=":", label=lab + " (training)")
ax.axhline(thr, color=C_NEU, lw=.9, ls="--")
ax.text(24, 2.55, f"dashed line: random RNN's best\nvalidation loss = {thr:.3f}", color=C_NEU, fontsize=9.5)
for m, dy in [("RNN + GloVe", 0.55), ("RNN", 0.42)]:
    e = conv[m]
    ax.annotate(f"epoch {e}", xy=(e, thr), xytext=(e - (10 if m == "RNN" else 3), thr + dy),
                color=C_KEY if m == "RNN + GloVe" else C_BASE, fontsize=10,
                arrowprops=dict(arrowstyle="->", color=C_KEY if m == "RNN + GloVe" else C_BASE))
ax.text(23, thr + 0.7, "GloVe gets there 2.2× sooner", color=C_KEY, fontsize=10.5, fontweight="bold")
ax.set_xlabel("Epoch"); ax.set_ylabel("Loss (label-smoothed cross-entropy)"); ax.set_ylim(1.7, 4.6)
ax.legend(frameon=False, fontsize=9, loc="upper right")
fig.savefig(os.path.join(OUT, "glove.pdf"), dpi=200, bbox_inches="tight"); plt.close(fig)
print("ok")
