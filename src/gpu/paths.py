from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a path relative to the repository root.

    This allows scripts to find data/ and report/ correctly whether they are run
    from the repository root or from src/gpu.
    """
    path = Path(path_value)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path
