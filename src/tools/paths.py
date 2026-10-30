"""Paths shared by data generators and practice runners."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def data_file(problem: str, name: str) -> Path:
    path = ROOT / "data" / "generated_data" / problem / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def result_file(problem: str) -> Path:
    return ROOT / "outputs" / problem / "results.txt"
