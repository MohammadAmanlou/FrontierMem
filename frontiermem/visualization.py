from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_main_metrics(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    ax = metrics.set_index("model")[cols].plot(kind="bar", figsize=(11, 5), rot=18)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score")
    ax.set_title("FrontierMem prototype: main decision metrics")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_boundary_metrics(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["inside_accuracy", "outside_accuracy", "near_accuracy"]
    ax = metrics.set_index("model")[cols].plot(kind="bar", figsize=(11, 5), rot=18)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by applicability region")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_error_rates(metrics: pd.DataFrame, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cols = ["overgeneralization_rate", "unnecessary_clarification_rate"]
    ax = metrics.set_index("model")[cols].plot(kind="bar", figsize=(11, 5), rot=18)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Rate (lower is better)")
    ax.set_title("Personalization error rates")
    ax.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


__all__ = ["plot_main_metrics", "plot_boundary_metrics", "plot_error_rates"]
