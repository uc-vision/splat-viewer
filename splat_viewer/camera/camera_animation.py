import math
from beartype.typing import List, Tuple
from beartype import beartype
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp
from splat_viewer.camera.fov import split_rt

import scipy.interpolate

def generalized_sigmoid(x, smooth, eps=1e-6):
    """Smooth interpolation function used for camera animation."""
    if x < eps:
        return 0.0
    if (1 - x) < eps:
        return 1.0
    else:
        return 1/(1 + (x / (1 - x)) ** -smooth)



@dataclass
class AnimationConfig:
    """Configuration for camera animation."""
    animate_speed: float = 1.0
    animate_pausing: float = 0.4
    
    def update_from_settings(self, settings):
        """Update animation config from viewer settings."""
        self.animate_speed = settings.animate_speed
        self.animate_pausing = settings.animate_pausing


class CameraPathAnimator:
    """Handles camera path animation without GUI dependencies."""

    @beartype
    def __init__(self, motion_path: List[np.ndarray], loop: bool = True,
                 config: AnimationConfig = AnimationConfig()):
        self.loop = loop
        self.config = config
        self.t = 0.0

        if loop:
            motion_path = [*motion_path, motion_path[0]]

        self.total = len(motion_path) - 1
        times = np.arange(len(motion_path))
        r, t = zip(*[split_rt(m) for m in motion_path])

        self.rots, self.pos = np.array(r), np.array(t)
        self.slerp = Slerp(times, R.from_matrix(self.rots))
        self.interp = scipy.interpolate.CubicSpline(times, self.pos, axis=0, 
          bc_type='periodic' if loop else 'not-a-knot')

    def update_config(self, animate_speed: float = None, animate_pausing: float = None) -> None:
        """Update animation configuration parameters."""
        if animate_speed is not None:
            self.config.animate_speed = animate_speed
        if animate_pausing is not None:
            self.config.animate_pausing = animate_pausing

    def get_camera_pose(self, dt: float) -> Tuple[np.ndarray, np.ndarray, bool]:
        """Get the camera pose at the current time step.
        
        Returns:
            Tuple of (rotation_matrix, position_vector, finished)
            finished is True when a non-looping animation reaches the end
        """
        inc = dt * self.config.animate_speed
        finished = self.t + inc >= self.total

        if not self.loop:
            self.t = min(self.total, self.t + inc)
        else:
            self.t = (self.t + inc) % self.total

        frac = math.fmod(self.t, 1)
        t = math.floor(self.t) + generalized_sigmoid(frac, self.config.animate_pausing + 1)

        r = self.slerp(np.array([t])).as_matrix()[0]
        pos = self.interp(t)

        return r, pos, finished

    def reset(self) -> None:
        """Reset animation to start."""
        self.t = 0.0

    def is_finished(self) -> bool:
        """Check if animation has finished (for non-looping animations)."""
        return not self.loop and self.t >= self.total
