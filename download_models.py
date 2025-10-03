"""
Cross-platform model downloader using underlying libraries:
- huggingface_hub for HF repo/files

Downloads:
  1) hf download Wan-AI/Wan2.1-T2V-1.3B --local-dir models/Wan2.1-T2V-1.3B --exclude ...
  2) hf download Kijai/WanVideo_comfy umt5-xxl-enc-fp8_e4m3fn.safetensors --local-dir models/WanVideo_comfy
  3) hf download Efficient-Large-Model/LongLive-1.3B --local-dir models/LongLive-1.3B
"""

import logging
import shutil
import sys
from pathlib import Path

from lib.models_config import (
    ensure_models_dir,
    models_are_downloaded,
)

# Set up logger
logger = logging.getLogger(__name__)

# --- third-party libs ---
try:
    from huggingface_hub import hf_hub_download, snapshot_download
except Exception:
    print(
        "Error: huggingface_hub is required. Install with: pip install huggingface_hub",
        file=sys.stderr,
    )
    raise


def download_hf_repo_excluding(
    repo_id: str, local_dir: Path, ignore_patterns: list[str]
) -> None:
    """
    Download an entire HF repo snapshot while excluding specific files.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    # snapshot_download supports exclude via `ignore_patterns`
    # (patterns are glob-like, relative to the repo root)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,  # copy files for portability
        ignore_patterns=ignore_patterns,
        # token is picked up automatically from HUGGINGFACE_TOKEN if set
        # revision=None,  # optionally pin a commit/tag if you like
    )
    print(f"[OK] Downloaded repo '{repo_id}' to: {local_dir}")


def download_hf_single_file(repo_id: str, filename: str, local_dir: Path) -> None:
    """
    Download a single file from an HF repo into a target folder.
    """
    local_dir.mkdir(parents=True, exist_ok=True)
    out_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
    )
    print(f"[OK] Downloaded file '{filename}' from '{repo_id}' to: {out_path}")


def move_file(src: Path, dst: Path) -> None:
    """
    Move a file to a destination path, creating directories as needed.
    """
    if not src.exists():
        raise FileNotFoundError(f"Source file not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))
    print(f"[OK] Moved '{src}' -> '{dst}'")


def check_models_downloaded() -> bool:
    """Check if required model files are already downloaded."""
    return models_are_downloaded()


def download_required_models():
    """Download required models if they are not already present."""
    if check_models_downloaded():
        logger.info("Models already downloaded, skipping download")
        return

    logger.info("Downloading required models...")
    try:
        download_models()
        logger.info("Model download completed successfully")
    except Exception as e:
        logger.error(f"Error downloading models: {e}")
        raise


def download_models() -> None:
    # HuggingFace repos
    wan_video_repo = "Wan-AI/Wan2.1-T2V-1.3B"
    wan_video_comfy_repo = "Kijai/WanVideo_comfy"
    wan_video_comfy_file = "umt5-xxl-enc-fp8_e4m3fn.safetensors"
    longlive_repo = "Efficient-Large-Model/LongLive-1.3B"

    # Ensure models directory exists and get paths
    models_root = ensure_models_dir()
    wan_video_dst = models_root / "Wan2.1-T2V-1.3B"
    wan_video_comfy_dst = models_root / "WanVideo_comfy"
    longlive_dst = models_root / "LongLive-1.3B"

    # 1) HF repo download excluding a large file
    wan_video_exclude = ["models_t5_umt5-xxl-enc-bf16.pth"]
    download_hf_repo_excluding(
        wan_video_repo, wan_video_dst, ignore_patterns=wan_video_exclude
    )

    # 2) HF single file download into a folder
    download_hf_single_file(
        wan_video_comfy_repo, wan_video_comfy_file, wan_video_comfy_dst
    )

    # 3) HF repo download for LongLive-1.3B
    download_hf_repo_excluding(longlive_repo, longlive_dst, ignore_patterns=[])

    print("\nAll downloads complete.")


if __name__ == "__main__":
    try:
        download_models()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(130)
    except Exception as e:
        print(f"\nERROR: {e}", file=sys.stderr)
        sys.exit(1)
