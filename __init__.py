"""Top-level package metadata for local imports and tooling."""

try:
    # Package import path (when project is installed or imported as package).
    from .version import __version__  # type: ignore[attr-defined]
except Exception:
    try:
        # Workspace execution path (legacy absolute import).
        from version import __version__  # type: ignore[no-redef]
    except Exception:
        __version__ = "0.0.0"


