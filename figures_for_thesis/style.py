"""Figure settings shared by the thesis exports (same sizes as Chapter 4)."""
from pathlib import Path
import shutil, logging
import matplotlib.pyplot as plt

logging.getLogger('fontTools').setLevel(logging.ERROR)

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
THESIS = ROOT.parent / 'PhD_thesis_20251216' / 'figures_proj2'
BLUE, GREEN, YELLOW, ORANGE = '#1478E1', '#28AF3C', '#FAC828', '#FA8C00'
PURPLE, CYAN, GREY = '#8C64E1', '#32B4C8', '#787878'
plt.rcParams.update({'text.usetex': False, 'font.family': 'serif', 'mathtext.fontset': 'cm',
 'axes.labelsize': 16, 'xtick.labelsize': 14, 'ytick.labelsize': 14,
 'legend.fontsize': 14, 'figure.dpi': 196, 'figure.facecolor': 'w',
 'lines.linewidth': 1.5, 'lines.solid_capstyle': 'round', 'lines.dash_capstyle': 'round',
 'lines.solid_joinstyle': 'round', 'lines.dash_joinstyle': 'round', 'pdf.fonttype': 42})


def tab(ax, text):
    ax.set_title(text, loc='left', x=.035, y=.96, pad=0, va='top', fontsize=12,
                 bbox={'boxstyle': 'round', 'facecolor': 'white',
                       'edgecolor': plt.rcParams['legend.edgecolor'],
                       'alpha': plt.rcParams['legend.framealpha']}, zorder=10)


def frame(ax):
    ax.tick_params(direction='in', which='both', top=True, right=True)


def legend(fig, ax, ncol=2):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower left', bbox_to_anchor=(.08, .012),
               ncol=ncol, frameon=True, fancybox=True)


def save(fig, name):
    path = OUT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={'CreationDate': None})
    target = THESIS / name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, target)
    plt.close(fig)
    print(name)
