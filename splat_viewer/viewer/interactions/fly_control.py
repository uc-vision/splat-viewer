
from PySide6 import QtGui
from PySide6.QtCore import Qt, QEvent

import numpy as np

from ..interaction import Interaction


class FlyControl(Interaction):
  def __init__(self):
    super(FlyControl, self).__init__()
    self.drag_mouse_pos = None

    self.directions = { 
      Qt.Key_Q : np.array([0.,  -1.,  0.]),
      Qt.Key_E : np.array([0.,  1.,  0.]),
      
      Qt.Key_W : np.array([0.,  0.,  1.]),
      Qt.Key_S : np.array([0.,  0.,  -1.]),

      Qt.Key_A : np.array([-1., 0.,  0.]),
      Qt.Key_D : np.array([1.,  0.,  0.])
    }

    self.rotations = { 
      Qt.Key_Z : np.array([0.,  0.,  1.]),
      Qt.Key_C : np.array([0.,  0.,  -1.]),

      Qt.Key_Up : np.array([0.,  1.,  0.]),
      Qt.Key_Down : np.array([0.,  -1.,  0.]),
      
      Qt.Key_Left : np.array([-1.,  0.,  0.]),
      Qt.Key_Right : np.array([1.,  0.,  0.]),
    }


    self.speed_controls = { 
      Qt.Key_Plus : 2.0,
      Qt.Key_Minus : 0.5,
    }


    self.held_keys = set(self.directions.keys()) | set(self.rotations.keys())


  def keyPressEvent(self, event: QtGui.QKeyEvent):
    if event.key() in self.held_keys and not event.isAutoRepeat():
      self.transition(None)

    if event.key() in self.speed_controls and event.modifiers() & Qt.KeypadModifier:
      self.update_setting(move_speed = self.settings.move_speed * self.speed_controls[event.key()])
      return True
      

  def update(self, dt:float):
    scene = self.scene_widget
    mod = 0.1 if Qt.Key_Shift in self.keys_down else 1.0

    for key in self.keys_down:
      if key in self.rotations:
        scene.rotate_camera(mod * self.rotations[key] * dt * self.settings.rotate_speed)

      elif key in self.directions:
        scene.move_camera(mod * self.directions[key] * dt * self.settings.move_speed)
    

  def mousePressEvent(self, event: QtGui.QMouseEvent):
    if event.buttons() & Qt.RightButton:
      self.drag_mouse_pos = event.localPos()
      return True
    

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    if event.button() & Qt.RightButton:
      self.drag_mouse_pos = None
      return True
    
  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    if event.buttons() & Qt.RightButton and self.drag_mouse_pos is not None:
      delta = event.localPos() - self.drag_mouse_pos

      sz = self.scene_widget.size()

      speed = self.settings.drag_speed
      self.scene_widget.rotate_camera([delta.x() / sz.width() * speed, 
                                -delta.y() / sz.height() * speed,
                                0])
      
      self.drag_mouse_pos = event.localPos()
    