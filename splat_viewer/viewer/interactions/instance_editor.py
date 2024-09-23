from abc import ABCMeta
from dataclasses import dataclass
from enum import Enum
import math
from typing import Callable, Optional, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
import numpy as np
import torch

from splat_viewer.editor.editor import AddInstance, Editor
from splat_viewer.editor.gaussian_scene import GaussianScene, Instance, random_color
from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.interaction import Interaction



def in_sphere(positions:torch.Tensor, center:torch.Tensor, radius:float):
  idx = in_box(positions, center - radius, center + radius)
  return idx[torch.linalg.norm(positions[idx] - center, dim=-1) <= radius]

def in_box(positions:torch.Tensor, lower:torch.Tensor, upper:np.array):
  mask = ((positions >= lower) & (positions <= upper)).all(dim=-1)
  return torch.nonzero(mask, as_tuple=True)[0]






class DrawMode(Enum):
  Draw = 0
  Erase = 1
  Select = 2


class InstanceEditor(Interaction):
  def __init__(self):
    super(InstanceEditor, self).__init__()

    self.mode: Optional[DrawMode] = None
    self.current_label = 0

    self.current_instance:Optional[int] = None
    self.current_mask = torch.zeros(self.gaussians.position.shape[0], dtype=torch.bool, device=self.gaussians.device)

    self.color = (1, 0, 0)


  @property
  def ready_mode(self) -> Optional[DrawMode]:
    draw = bool(self.modifiers & Qt.ControlModifier)
    erase = bool(self.modifiers & Qt.ShiftModifier) and self.current_instance is not None

    return DrawMode.Draw if draw else DrawMode.Erase if erase else None
  
  
  
  def mousePressEvent(self, event: QtGui.QMouseEvent):
    if event.button() == Qt.LeftButton and self.ready_mode is not None:
      self.mode = self.ready_mode




      self.draw((event.x(), event.y()))
      return True
    
  def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    if event.button() == Qt.LeftButton and self.mode is not None:
      self.mode = None
      return True
    
  def select(self, cursor_pos:Tuple[int, int]):
    depth = self.lookup_depth(cursor_pos)

    p, r = self.unproject_radius(cursor_pos, depth, self.settings.brush_size)
    return in_sphere(self.gaussians.position, self.from_numpy(p), r)



  def draw(self, cursor_pos:Tuple[int, int]):
    idx = self.select((cursor_pos))


    if self.mode == DrawMode.Draw:
      self.current_mask[idx] = True
    elif self.mode == DrawMode.Erase:
      self.current_mask[idx] = False


    if self.current_instance is None:
      self.current_instance = self.scene.next_instance()
      name = self.scene.class_labels[self.current_label]
      instance = Instance(self.scene.next_instance_id(), self.current_label, name, random_color())

      self.editor.apply(AddInstance(instance))


  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    if self.mode is not None:
      self.draw((event.x(), event.y()))


    

  def wheelEvent(self, event: QtGui.QWheelEvent):
    if self.ready_mode is not None:
      dy = event.pixelDelta().y()
      factor = math.pow(1.0015, dy)

      self.update_setting(brush_size = np.clip(self.settings.brush_size * factor, 1, 200))
      return True


  def paintEvent(self, event: QtGui.QPaintEvent, dirty:bool):
    painter = QtGui.QPainter(self.scene_widget)
    painter.setRenderHint(QtGui.QPainter.Antialiasing)

    color = Qt.red if self.ready == DrawMode.Draw else Qt.green if self.ready == DrawMode.Erase else Qt.black
    painter.setPen(QtGui.QPen(color, 1, Qt.DashLine))

    point = QtCore.QPointF(*self.cursor_pos)
    painter.drawEllipse(point, 
                        self.settings.brush_size, self.settings.brush_size)
    painter.end()
      


