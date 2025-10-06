from pathlib import Path
from beartype.typing import List
from beartype import beartype
import numpy as np


@beartype
def read_camera_path(workspace_path: Path) -> List[np.ndarray]:
    """Read camera path from saved numpy file."""
    camera_path_file = workspace_path / "camera_path.npy"
    if camera_path_file.exists():
        kp = list(np.load(camera_path_file))
        print(f"Loaded {len(kp)} keypoints from {camera_path_file}")
        return kp
    else:
        print(f"No camera path file found at {camera_path_file}")
        return []


@beartype
def write_camera_path(workspace_path: Path, keypoints: List[np.ndarray]) -> None:
    """Write camera path to numpy file."""
    camera_path_file = workspace_path / "camera_path.npy"
    np.save(camera_path_file, np.array(keypoints))
    print(f"Saved {len(keypoints)} keypoints to {camera_path_file}")
