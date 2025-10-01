import math

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from beartype import beartype
import numpy as np
import torch

from splat_annotate.viewer.interaction import Interaction




def in_sphere(positions:torch.Tensor, center:torch.Tensor, radius:float):
  idx = in_box(positions, center - radius, center + radius)
  return idx[torch.linalg.norm(positions[idx] - center, dim=-1) <= radius]

def in_box(positions:torch.Tensor, lower:torch.Tensor, upper:torch.Tensor):
  mask = ((positions >= lower) & (positions <= upper)).all(dim=-1)
  return torch.nonzero(mask, as_tuple=True)[0]


class ScribbleGeometric(Interaction):
  def __init__(self):
    super(ScribbleGeometric, self).__init__()

    self.drawing = False

    self.current_label = 0
    self.current_points = None

    self.color = np.array([1, 0, 0], dtype=np.float32)

  @property
  def ready(self):
    return bool(self.modifiers & Qt.KeyboardModifier.ControlModifier)
  

  def mousePressEvent(self, event: QtGui.QMouseEvent) -> bool:
    if event.button() == Qt.MouseButton.LeftButton and event.modifiers() & Qt.KeyboardModifier.ControlModifier:
      self.drawing = True
      self.draw(np.array([event.x(), event.y()]))
      return True
    
    return False

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> bool:
    if event.button() == Qt.MouseButton.LeftButton and self.drawing:
      self.drawing = False
      return True
    
    return False
  
  @beartype
  def draw(self, cursor_pos:np.ndarray):
    depth = self.lookup_depth(cursor_pos)

    p, r = self.unproject_radius(cursor_pos, depth, self.settings.brush_size)
    idx = in_sphere(self.gaussians.position, self.from_numpy(p), r)

    if self.current_points is None:
      self.current_points = idx
    else:
      self.current_points = torch.cat([self.current_points, idx]).unique()

      self.update_gaussians(self.gaussians.set_colors(self.color, self.current_points))

    self.set_dirty()


      

    


  def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> bool:
    if self.drawing:
        self.draw(np.array([event.x(), event.y()]))
        return True
    
    return False

  def wheelEvent(self, event: QtGui.QWheelEvent) -> bool:
    if self.ready:
      dy = event.pixelDelta().y()
      factor = math.pow(1.0015, dy)

      self.update_setting(brush_size = np.clip(self.settings.brush_size * factor, 1, 100))
      return True
    
    return False

  def keyPressEvent(self, event: QtGui.QKeyEvent) -> bool:

    
    return super().keyPressEvent(event)

  def paintEvent(self, event: QtGui.QPaintEvent, view_changed:bool) -> bool:
    if self.ready:
      painter = QtGui.QPainter(self.scene_widget)
      painter.setRenderHint(QtGui.QPainter.RenderHint.Antialiasing)
      painter.setPen(QtGui.QPen(Qt.GlobalColor.red, 1, Qt.PenStyle.DashLine))

      point = QtCore.QPointF(*self.cursor_pos)
      painter.drawEllipse(point, 
                          self.settings.brush_size, self.settings.brush_size)
      painter.end()

      return True
    
    return False


