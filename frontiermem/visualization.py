from __future__ import annotations

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def plot_overall_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    cols = ["accuracy", "balanced_accuracy", "macro_f1"]
    ax = metrics_df.set_index("model")[cols].plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Overall Frontier Decision Performance")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_boundary_metrics(metrics_df: pd.DataFrame, output_path: str) -> None:
    cols = ["inside_accuracy", "outside_accuracy", "near_accuracy"]
    ax = metrics_df.set_index("model")[cols].plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Preference Applicability Region")
    ax.legend(["Inside boundary", "Outside boundary", "Near boundary"], loc="lower right")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_failure_rates(metrics_df: pd.DataFrame, output_path: str) -> None:
    cols = ["overgeneralization_rate", "unnecessary_clarification_rate"]
    ax = metrics_df.set_index("model")[cols].plot(kind="bar", figsize=(9, 5))
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Error rate (lower is better)")
    ax.set_title("Critical Personalization Failure Rates")
    ax.legend(["Overgeneralization", "Unnecessary clarification"], loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=15, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()
