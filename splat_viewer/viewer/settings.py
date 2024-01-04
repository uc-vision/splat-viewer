from dataclasses import dataclass
from enum import Enum
from typing import Tuple


class ViewMode(Enum):
  Normal = 0
  Depth = 1
  Points = 2
  Hidden = 3

@dataclass 
class Show:
  initial_points: bool = False
  cameras: bool = False
  cropped : bool = False




@dataclass(frozen=True)
class Settings:
  update_rate : int = 20
  move_speed : float = 1.0

  transition_time : float = 0.5
  rotate_speed : float = 1.0

  animate_speed : float = 0.5
  animate_pausing: float = 0.4

  zoom_discrete : float = 1.2
  zoom_continuous : float = 0.1

  drag_speed : float = 1.0
  point_size : float = 2.0

  snapshot_size: Tuple[int, int] = (8192, 6144) 
  snapshot_tile: int = 1024

  tile_size : int = 16
  device : str = 'cuda:0'
  
  bg_color : Tuple[float, float, float] = (1, 1, 1)
  alpha_depth: float = 1.0

  show : Show = Show()
  view_mode : ViewMode = ViewMode.Normal
  

