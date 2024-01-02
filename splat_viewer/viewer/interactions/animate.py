
import math
from typing import Callable, List, Optional
from PySide6 import QtGui
from PySide6.QtCore import Qt
from beartype import beartype

import numpy as np
import scipy

from splat_viewer.camera.fov import split_rt

from ..interaction import Interaction
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp



def generalized_sigmoid(x, smooth, eps=1e-6):
    if x < eps:
      return 0.0
    if (1 - x) < eps:
      return 1.0
    else:
      return 1/(1 + (x / (1 - x)) ** -smooth)
      

def animate_to(state, current_pose, dest_pose, on_finish=None):
  to = AnimateCamera([current_pose, dest_pose], loop=False, on_finish=on_finish)
  state.transition(to)

def animate_to_loop(state, current_pose, loop_poses, on_finish=None):
  loop = AnimateCamera(loop_poses, loop=True, on_finish=on_finish)
  to = AnimateCamera([current_pose, loop_poses[0]], loop=False, 
                     on_finish=lambda: state.transition(loop))

  state.transition(to)

class AnimateCamera(Interaction):
  @beartype
  def __init__(self, motion_path:List[np.ndarray], loop=True, 
               on_finish:Optional[Callable]=None):
    super(AnimateCamera, self).__init__()

    bc_type = 'not-a-knot' 

    if loop:
      motion_path = [*motion_path, motion_path[0]]
      bc_type = 'periodic' 

    self.loop = loop
    self.on_finish = on_finish

    self.total = len(motion_path) - 1
    times = np.arange(len(motion_path))    
      
    r, t = zip(*[split_rt(m) for m in motion_path])

    self.rots, self.pos = np.array(r), np.array(t)

    self.slerp = Slerp(times, R.from_matrix(self.rots))
    self.interp = scipy.interpolate.CubicSpline(times, self.pos, axis=0, bc_type=bc_type)
    
    self.t = 0.0  

    self.speed_controls = { 
      Qt.Key_Plus : 1,
      Qt.Key_Minus : -1,
    }


  def keyPressEvent(self, event: QtGui.QKeyEvent):
      if event.key() in self.speed_controls:
        modifier = self.speed_controls[event.key()]

        if event.modifiers() & Qt.ShiftModifier:
          self.update_setting(animate_pausing = np.clip(self.settings.animate_pausing + (0.2 * modifier), 0, 2))
        else:
          self.update_setting(animate_speed = self.settings.animate_speed * (2 ** modifier))

        return True



  def update(self, dt:float):
    scene = self.scene_widget

    inc = dt * self.settings.animate_speed

    if self.t + inc >= self.total:
      if self.on_finish is not None:
        self.on_finish()

    if not self.loop:
      self.t = min(self.total, self.t + inc)
    else:
      self.t = (self.t + inc) % self.total

    frac = math.fmod(self.t, 1)    
    t = math.floor(self.t) + generalized_sigmoid(frac, self.settings.animate_pausing + 1)   

    r = self.slerp(np.array([t])).as_matrix()[0]

    pos = self.interp(t)

    scene.set_camera_pose(r, pos)

    

    