from enum import Enum
import math
from typing import Optional, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
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


  @property
  def ready_mode(self) -> Optional[DrawMode]:

    if self.current_instance is not None:
      if bool(self.modifiers & Qt.ControlModifier):
        return DrawMode.Erase
      else:
        return DrawMode.Draw

    if bool(self.modifiers & Qt.ShiftModifier):
      return DrawMode.Select

    return None
  

  def keyPressEvent(self, event: QtGui.QKeyEvent):
    # mode = self.ready_mode
    # if mode == DrawMode.Select:
    #   # set cursor to crosshair
    #   self.scene_widget.setCursor(Qt.CursorShape.CrossCursor)
    #   return True
    # elif mode in [DrawMode.Draw, DrawMode.Erase]:

    #   # Hide the cursor when in Draw mode
    #   self.scene_widget.setCursor(Qt.CursorShape.BlankCursor)
    #   return True
    # else:
    #   self.scene_widget.setCursor(Qt.CursorShape.ArrowCursor)

    return False

  @property
  def current_instance(self) -> Optional[int]:
    return self.scene.selected_instance

  def mousePressEvent(self, event: QtGui.QMouseEvent):
    pass
    # if event.button() == Qt.LeftButton and self.current_instance is not None:
    #   if self.ready_mode is not None:
    #     self.mode = self.ready_mode
    #     self.draw((event.x(), event.y()))
    #     return True
      
    #   else:

    #     self.editor.unselect_instance()

  
    
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

      mode = self.ready_mode

      painter = QtGui.QPainter(self.scene_widget)
      painter.setRenderHint(QtGui.QPainter.Antialiasing)

      if mode in [DrawMode.Draw, DrawMode.Erase]:
        color = Qt.red if mode == DrawMode.Erase else Qt.green
        painter.setPen(QtGui.QPen(color, 1, Qt.DashLine))

        point = QtCore.QPointF(*self.cursor_pos)
        painter.drawEllipse(point, 
                            self.settings.brush_size, self.settings.brush_size)
        

      painter.end()
        


