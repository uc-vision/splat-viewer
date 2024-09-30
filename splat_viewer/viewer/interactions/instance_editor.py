from enum import Enum
import math
from typing import Optional, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
from beartype import beartype
import numpy as np
import torch

from splat_viewer.editor.edit import ModifyInstances, AddInstance
from splat_viewer.editor.gaussian_scene import GaussianScene, Instance, random_color
from splat_viewer.viewer.interaction import Interaction



def in_sphere(positions:torch.Tensor, center:torch.Tensor, radius:float):
  idx = in_box(positions, center - radius, center + radius)
  return idx[torch.linalg.norm(positions[idx] - center, dim=-1) <= radius]

def in_box(positions:torch.Tensor, lower:torch.Tensor, upper:np.array):
  mask = ((positions >= lower) & (positions <= upper)).all(dim=-1)
  return torch.nonzero(mask, as_tuple=True)[0]


def instance_counts(scene:GaussianScene, indexes:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
  instance_ids = scene.gaussians.instance_label[indexes]
  instance_ids = instance_ids[instance_ids >= 0]

  unique, counts = torch.unique(instance_ids, return_counts=True)
  return unique, counts


def paint_instance(scene:GaussianScene, instance_id:int, indexes:torch.Tensor):
  assert instance_id in scene.instances

  instance_ids = torch.full_like(indexes, instance_id, device=scene.device)
  class_ids = scene.gaussians.label[indexes]

  return ModifyInstances(indexes, instance_ids, class_ids)

def erase_instance(scene:GaussianScene, instance_id:int, indexes:torch.Tensor):
  assert instance_id in scene.instances

  instance_ids = torch.full_like(indexes, -1, device=scene.device)
  class_ids = torch.zeros_like(indexes, device=scene.device)

  return ModifyInstances(indexes, instance_ids, class_ids)


class DrawMode(Enum):
  Draw = 0
  Erase = 1
  Select = 2


class InstanceEditor(Interaction):
  def __init__(self):
    super(InstanceEditor, self).__init__()

    self.mode: Optional[DrawMode] = None
    self.current_label = 0

    self.current_mask = torch.zeros(self.gaussians.position.shape[0], dtype=torch.bool, device=self.gaussians.device)
    self.color = (1, 0, 0)


  def ready_mode(self, modifiers:Qt.KeyboardModifiers) -> Optional[DrawMode]:

    if bool(modifiers & Qt.ControlModifier):
        return DrawMode.Draw
    
    if self.current_instance is not None and bool(modifiers & Qt.ShiftModifier):
      return DrawMode.Erase

    return None
  
  def cursor_mode(self, modifiers:Qt.KeyboardModifiers) -> Qt.CursorShape:
    mode = self.ready_mode(modifiers)
    if mode == DrawMode.Select:
      return Qt.CursorShape.CrossCursor
    elif mode in [DrawMode.Draw, DrawMode.Erase]:
      return Qt.CursorShape.BlankCursor
    else:
      return Qt.CursorShape.ArrowCursor



  def keyPressEvent(self, event: QtGui.QKeyEvent):
    self.scene_widget.setCursor(self.cursor_mode(event.modifiers()))
    return False
  
  def keyReleaseEvent(self, event: QtGui.QKeyEvent):
    self.scene_widget.setCursor(self.cursor_mode(event.modifiers()))
    return False

  @property
  def current_instance(self) -> Optional[int]:
    return self.scene.selected_instance

  def mousePressEvent(self, event: QtGui.QMouseEvent) -> bool:
    mode = self.ready_mode(event.modifiers())
    cursor_pos = (event.x(), event.y())

    if mode is None:       
      instance = self.get_select_instance(cursor_pos)
      if instance is not None:
        self.select_instance(instance)
        return True

    return False
    
  def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> bool:
    if event.button() == Qt.LeftButton and self.mode is not None:
      self.mode = None
      return True
    
  @beartype
  def get_select_sphere(self, cursor_pos:Tuple[int, int], radius:float) -> torch.Tensor:
    depth = self.lookup_depth(cursor_pos)

    p, r = self.unproject_radius(cursor_pos, depth, radius)
    return in_sphere(self.gaussians.position, self.from_numpy(p), r)

  @beartype
  def get_select_instance(self, cursor_pos:Tuple[int, int]) -> Optional[int]:
    idx = self.get_select_sphere(cursor_pos, self.settings.select_radius)
    unique, counts = instance_counts(self.scene, idx)
    
    if counts.shape[0] == 0:
      return None
    
    idx = unique[counts.argmax()]
    return idx.item()


  def draw(self, cursor_pos:Tuple[int, int]):
    idx = self.get_select_sphere(cursor_pos, self.settings.brush_size)


    if self.current_instance is None:
      self.current_instance = self.scene.next_instance()
      name = self.scene.class_labels[self.current_label]
      instance = Instance(self.scene.next_instance_id(), self.current_label, name, random_color())
      self.editor.apply(AddInstance(instance))

    if self.mode == DrawMode.Draw:
      self.current_mask[idx] = True
    elif self.mode == DrawMode.Erase:
      self.current_mask[idx] = False

    edit = paint_instance(self.scene, self.current_instance, self.current_mask)
    self.editor.update_edit(edit)


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

      mode = self.ready_mode(self.modifiers)

      painter = QtGui.QPainter(self.scene_widget)
      painter.setRenderHint(QtGui.QPainter.Antialiasing)

      if mode in [DrawMode.Draw, DrawMode.Erase]:
        color = Qt.red if mode == DrawMode.Erase else Qt.green
        painter.setPen(QtGui.QPen(color, 1, Qt.DashLine))

        point = QtCore.QPointF(*self.cursor_pos)
        painter.drawEllipse(point, 
                            self.settings.brush_size, self.settings.brush_size)
        

      painter.end()
        


