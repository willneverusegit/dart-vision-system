"""
I/O utilities for atomic file operations.
Prevents data corruption during crashes or power failures.
"""
import os
import yaml
import tempfile
from pathlib import Path
from typing import Any, Dict
import logging

logger = logging.getLogger(__name__)


def atomic_write_yaml(filepath: Path, data: Dict[str, Any]) -> None:
    """
    Write YAML file atomically using temporary file + os.replace().

    Args:
        filepath: Target file path
        data: Dictionary to serialize as YAML

    Raises:
        IOError: If write operation fails
        yaml.YAMLError: If data cannot be serialized
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Create temp file in same directory (ensures same filesystem)
    fd, temp_path = tempfile.mkstemp(
        dir=filepath.parent,
        prefix=f".{filepath.stem}_",
        suffix=".tmp"
    )

    try:
        # Write to temp file
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            yaml.safe_dump(data, f, default_flow_style=False, allow_unicode=True)

        # Atomic replace (POSIX guarantees atomicity)
        os.replace(temp_path, filepath)
        logger.debug(f"Atomically wrote {filepath}")

    except Exception as e:
        # Cleanup temp file on failure
        try:
            os.unlink(temp_path)
        except OSError:
            pass
        logger.error(f"Failed to write {filepath}: {e}")
        raise IOError(f"Atomic write failed: {e}") from e


def load_yaml(filepath: Path) -> Dict[str, Any]:
    """
    Load YAML configuration file.

    Args:
        filepath: Path to YAML file

    Returns:
        Parsed configuration dictionary

    Raises:
        FileNotFoundError: If file does not exist
        yaml.YAMLError: If file is malformed
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        logger.debug(f"Loaded {filepath}")
        return data if data is not None else {}

    except yaml.YAMLError as e:
        logger.error(f"Failed to parse {filepath}: {e}")
        raise


def ensure_config_dir(base_dir: Path = Path("config")) -> Path:
    """
    Ensure configuration directory exists.

    Args:
        base_dir: Base configuration directory (relative to project root)

    Returns:
        Absolute path to config directory
    """
    config_dir = Path(base_dir).resolve()
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def backup_config(filepath: Path, backup_suffix: str = ".bak") -> None:
    """
    Create backup of existing config file.

    Args:
        filepath: Original config file
        backup_suffix: Suffix for backup file (default: .bak)
    """
    filepath = Path(filepath)

    if not filepath.exists():
        return

    backup_path = filepath.with_suffix(filepath.suffix + backup_suffix)

    try:
        import shutil
        shutil.copy2(filepath, backup_path)
        logger.info(f"Created backup: {backup_path}")
    except Exception as e:
        logger.warning(f"Backup failed: {e}")