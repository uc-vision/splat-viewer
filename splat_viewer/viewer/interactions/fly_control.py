
from PySide6 import QtGui
from PySide6.QtCore import Qt, QEvent, QPoint
from PySide6.QtGui import QCursor
import numpy as np

from splat_viewer.viewer.scene_camera import rotation_around

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

    self.drag_2d = np.array([0, 0], dtype=np.float32)
    self.anchor_2d = None

    self.anchor_3d = None
    self.ref_pose = None


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
      p = event.localPos()
      self.anchor_3d = self.lookup_point_3d(np.array([p.x(), p.y()]))

      if self.anchor_3d is not None:
        self.drag_2d = np.array([0, 0], dtype=np.float32)
        self.anchor_2d = p    
        self.ref_pose = self.scene_widget.view_matrix
        self.scene_widget.setCursor(Qt.BlankCursor)
        return True
    

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    if event.button() & Qt.RightButton:

      self.anchor_3d = None
      self.ref_pose = None
      self.scene_widget.setCursor(Qt.ArrowCursor)

      return True
    
  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    if event.buttons() & Qt.RightButton and self.anchor_3d is not None:
      p = event.localPos()
      delta = p - self.anchor_2d

      self.drag_2d += np.array([delta.x(), delta.y()], dtype=np.float32)
      dx, dy = self.drag_2d

      # Get screen position of the anchor point
      screen_pos = self.scene_widget.mapToGlobal(self.anchor_2d.toPoint())
      QCursor.setPos(screen_pos)

      sz = self.scene_widget.size()
      speed = self.settings.drag_speed
      ref_pos = self.ref_pose[3, :3]
      
      r = rotation_around(ref_pos - self.anchor_3d, 'yxz', [dx / sz.width() * speed, -dy / sz.height() * speed, 0])
      
      self.scene_widget.set_view_matrix(self.ref_pose @ r) 
      return True
    