"""Simple reusable plotting functions."""

from pathlib import Path

import matplotlib.pyplot as plt


def save_scatter_plot(
    x_values,
    y_values,
    output_path: Path,
    *,
    x_label: str,
    y_label: str,
    title: str,
    show_identity_line: bool = False,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots()
    axes.scatter(x_values, y_values)
    if show_identity_line:
        axes.axline((0, 0), slope=1, color="gray", linestyle="--")
    axes.set(xlabel=x_label, ylabel=y_label, title=title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
