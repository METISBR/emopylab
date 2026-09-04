from __future__ import annotations

from pathlib import Path


def ensure_plugin_directories(base_dir: Path) -> dict[str, Path]:
    folders = {
        "algorithms": base_dir / "algorithms",
        "problems": base_dir / "problems",
        "metrics": base_dir / "metrics",
        "operators": base_dir / "operators",
    }
    for path in folders.values():
        path.mkdir(parents=True, exist_ok=True)
        init_file = path / "__init__.py"
        if not init_file.exists():
            init_file.write_text("", encoding="utf-8")

    legacy_metrics_dir = base_dir / "indicators" / "custom_metrics"
    metrics_dir = folders["metrics"]
    if legacy_metrics_dir.exists() and legacy_metrics_dir.is_dir():
        for legacy_file in sorted(legacy_metrics_dir.glob("*.py")):
            if legacy_file.name == "__init__.py":
                continue
            target_file = metrics_dir / legacy_file.name
            if target_file.exists():
                continue
            try:
                legacy_file.replace(target_file)
            except Exception:
                continue

        legacy_init = legacy_metrics_dir / "__init__.py"
        if legacy_init.exists():
            try:
                legacy_init.unlink()
            except Exception:
                pass

        try:
            if not any(legacy_metrics_dir.iterdir()):
                legacy_metrics_dir.rmdir()
        except Exception:
            pass

        legacy_indicators_root = base_dir / "indicators"
        try:
            if legacy_indicators_root.exists() and not any(legacy_indicators_root.iterdir()):
                legacy_indicators_root.rmdir()
        except Exception:
            pass
    return folders
