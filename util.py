from pathlib import Path


def save_mkdir(path):
    Path(path).mkdir(parents=True, exist_ok=True)