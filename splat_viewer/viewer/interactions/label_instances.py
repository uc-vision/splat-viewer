import math
from typing import Callable, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
import numpy as np
import torch

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.interaction import Interaction



class LabelInstances(Interaction):
  def __init__(self):
    super(LabelInstances, self).__init__()

    self.instances = []

    

    
  def draw(self, cursor_pos:Tuple[int, int]):
    depth = self.lookup_depth(cursor_pos)

    p, r = self.unproject_radius(cursor_pos, depth, self.settings.brush_size)
    idx = in_sphere(self.gaussians.position, self.from_numpy(p), r)

    if self.current_points is None:
      self.current_points = idx
    else:
      self.current_points = torch.cat([self.current_points, idx]).unique()

      self.update_gaussians(self.gaussians.set_colors(self.color, self.current_points))

    self.set_dirty()


  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    if self.drawing:
        self.draw((event.x(), event.y()))

  def wheelEvent(self, event: QtGui.QWheelEvent):
    if self.ready:
      dy = event.pixelDelta().y()
      factor = math.pow(1.0015, dy)

      self.update_setting(brush_size = np.clip(self.settings.brush_size * factor, 1, 100))
      return True

  def keyPressEvent(self, event: QtGui.QKeyEvent):

    
    return super().keyPressEvent(event)

  def paintEvent(self, event: QtGui.QPaintEvent, dirty:bool):
    if self.ready:
      painter = QtGui.QPainter(self.scene_widget)
      painter.setRenderHint(QtGui.QPainter.Antialiasing)
      painter.setPen(QtGui.QPen(Qt.red, 1, Qt.DashLine))

      point = QtCore.QPointF(*self.cursor_pos)
      painter.drawEllipse(point, 
                          self.settings.brush_size, self.settings.brush_size)
      painter.end()
      


