import os
import sys
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42   # embed TrueType so PDF text stays editable
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, Patch
import pandas as pd

# input CSV and output PNG are passed on the command line (a PDF is written next to the PNG):
#   python plot_ari.py ari_rc2_vs_annealing_3pools.csv ari_rc2_vs_annealing_3pools.png
csv, out = sys.argv[1], sys.argv[2]
df = pd.read_csv(csv)

SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
METHODS = ["Annealing, 100 bootstraps", "Annealing, 0 bootstraps", "RC2, 0 bootstraps"]
COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]   # categorical slots 1-3, fixed order
PANELS = [("ARI_called_cells", "Called cells only (unknown excluded), 3 pools completed by RC2")]

pools = list(dict.fromkeys(df.pool))
DPI = 100                    # 1 px = 1/100 in at this base dpi
BAR_W_PX, GAP_PX, RADIUS_PX = 22, 3, 4


def rounded_bar(ax, x, width, height, color):
    """Bar with a rounded data-end and a square baseline; corners are circular on screen."""
    fig = ax.figure
    bbox = ax.get_window_extent()
    px_per_x = bbox.width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    px_per_y = bbox.height / (ax.get_ylim()[1] - ax.get_ylim()[0])
    rx, ry = RADIUS_PX * fig.dpi / DPI / px_per_x, RADIUS_PX * fig.dpi / DPI / px_per_y
    k = 0.5523  # quarter-circle Bezier constant
    x0, x1, top = x - width / 2, x + width / 2, height
    verts = [(x0, 0), (x0, top - ry),
             (x0, top - ry + k * ry), (x0 + rx - k * rx, top), (x0 + rx, top),
             (x1 - rx, top),
             (x1 - rx + k * rx, top), (x1, top - ry + k * ry), (x1, top - ry),
             (x1, 0), (x0, 0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO,
             Path.CURVE4, Path.CURVE4, Path.CURVE4, Path.LINETO, Path.CLOSEPOLY]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor=color, edgecolor="none", zorder=3))


fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=DPI, facecolor=SURFACE)
axes = [ax]
fig.subplots_adjust(left=0.11, right=0.98, top=0.96, bottom=0.2)

for ax, (col, title) in zip(axes, PANELS):
    ax.set_facecolor(SURFACE)
    ax.set_xlim(-0.6, len(pools) - 0.4)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(pools)))
    ax.set_xticklabels([p.split("_")[0] for p in pools], fontsize=9, color=INK2)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.tick_params(axis="y", labelsize=9, colors=MUTED, length=0)
    ax.tick_params(axis="x", length=0, pad=6)
    ax.grid(axis="y", color=GRID, linewidth=1, zorder=0)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(BASELINE)
    ax.spines["bottom"].set_zorder(4)

fig.canvas.draw()  # fix axis geometry so bar widths and corner radii can be set in px
for ax, (col, _) in zip(axes, PANELS):
    px_per_x = ax.get_window_extent().width / (ax.get_xlim()[1] - ax.get_xlim()[0])
    w = BAR_W_PX * fig.dpi / DPI / px_per_x
    step = (BAR_W_PX + GAP_PX) * fig.dpi / DPI / px_per_x
    for i, pool in enumerate(pools):
        for j, (method, color) in enumerate(zip(METHODS, COLORS)):
            value = df.loc[(df.pool == pool) & (df.method == method), col].item()
            rounded_bar(ax, i + (j - 1) * step, w, value, color)

axes[0].set_ylabel("Adjusted Rand Index (ARI)", fontsize=9.5, color=INK2)
fig.legend(handles=[Patch(facecolor=c, edgecolor="none", label=m) for m, c in zip(METHODS, COLORS)],
           loc="lower center", bbox_to_anchor=(0.545, 0.01), ncol=3, frameon=False,
           fontsize=9.5, labelcolor=INK2, handlelength=1.0, handleheight=1.0, columnspacing=1.2)

fig.savefig(out, dpi=350, facecolor=SURFACE)
pdf = os.path.splitext(out)[0] + ".pdf"   # vector copy alongside the PNG
fig.savefig(pdf, facecolor=SURFACE)
print("wrote", out, "and", pdf)
