
from typing import Optional, Set, Tuple
from PySide6 import QtGui
from PySide6.QtCore import QEvent

from beartype import beartype
import numpy as np
import torch
from splat_viewer.gaussians.data_types import Rendering

from splat_viewer.viewer.settings import Settings


class Interaction():
  def __init__(self):
    super(Interaction, self).__init__()
    self._child = None 
    self.active = False

  def transition(self, interaction:Optional['Interaction']):
    self.pop()
    if interaction is not None:
      self.push(interaction)

  def push(self, interaction:'Interaction'):
    self._child = interaction
    if self.active:
      self._child._activate()


  def pop(self):    
    if self._child is not None:
      child = self._child
      self._child = None
      child._deactivate()

  def transition(self, interaction:Optional['Interaction']):
    self.pop()
    if interaction is not None:
      self.push(interaction)

  def _activate(self):
    self.on_activate()

    if self._child is not None:
      self._child._activate()

    self.active = True


  def _deactivate(self):
    if self._child is not None:
      child = self._child
      self._child = None
      child._deactivate()

    self.on_deactivate()


  def trigger_event(self, event: QEvent) -> bool:
    if self._child is not None:
      if self._child.trigger_event(event):
        return True
      
    return self.event(event) 
  
  def _update(self, dt:float) -> bool:
    if self._child is not None:
      if self._child._update(dt):
        return True
    
    return self.update(dt)

  @beartype
  def event(self, event: QEvent) -> bool:
    event_callbacks = {
      QEvent.KeyPress: self.keyPressEvent,
      QEvent.KeyRelease: self.keyReleaseEvent,
      QEvent.MouseButtonPress: self.mousePressEvent,
      QEvent.MouseButtonRelease: self.mouseReleaseEvent,
      QEvent.MouseMove: self.mouseMoveEvent,
      QEvent.Wheel: self.wheelEvent,
      QEvent.FocusIn: self.focusInEvent,
      QEvent.FocusOut: self.focusOutEvent,
    }

    if event.type() in event_callbacks:
      return event_callbacks[event.type()](event) or False
    
    return False
    
  def trigger_paint(self, event: QtGui.QPaintEvent, view_changed:bool) -> bool:
    if self._child is not None:
      if self._child.trigger_paint(event, view_changed):
        return True
      
    return self.paintEvent(event, view_changed) 

  def keyPressEvent(self, event: QtGui.QKeyEvent):
    return False

  def keyReleaseEvent(self, event: QtGui.QKeyEvent):
    return False

  def mousePressEvent(self, event: QtGui.QMouseEvent):
    return False

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
    return False

  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    return False

  def wheelEvent(self, event: QtGui.QWheelEvent):
    return False

  def focusInEvent(self, event: QtGui.QFocusEvent):
    return False
  
  def focusOutEvent(self, event: QtGui.QFocusEvent):
    return False
  
  def paintEvent(self, event: QtGui.QPaintEvent, view_changed:bool):
    return False

  @beartype
  def update(self, dt) -> bool:
    return False

  def on_activate(self):
    pass

  def on_deactivate(self):
    pass

  @property
  def scene_widget(self):
    from .scene_widget import SceneWidget
    return SceneWidget.instance
  
  
  
  @property
  def settings(self) -> Settings: 
    return self.scene_widget.settings
  
  @property
  def modifiers(self) -> QtGui.Qt.KeyboardModifier:
    return self.scene_widget.modifiers
  
  @property
  def keys_down(self) -> Set[QtGui.Qt.Key]:
    return self.scene_widget.keys_down
  
  @property
  def cursor_pos(self) -> Tuple[int, int]:
    return self.scene_widget.cursor_pos
  
  @property
  def current_point(self) -> np.ndarray:
    return self.scene_widget.current_point_3d

  def lookup_point_3d(self, p:Tuple[int, int]) -> np.ndarray:
    return self.scene_widget.lookup_point_3d(p)

  def lookup_depth(self, p:Tuple[int, int]) -> np.ndarray:
    return self.scene_widget.lookup_depth(p)
  

  def lookup_depths(self, p:np.ndarray) -> np.ndarray:
    return self.scene_widget.lookup_depths(p)
  
  def test_depths(self, p:np.ndarray, depth:np.ndarray) -> np.ndarray:
    return self.scene_widget.test_depths(p, depth)
  
  def unproject_point(self, p:Tuple[int, int], depth:float) -> np.ndarray:
    return self.scene_widget.unproject_point(p, depth)
  
  def unproject_radius(self, p:Tuple[int, int], depth:float, radius:float
                        ) -> Tuple[np.ndarray, float]:
      return self.scene_widget.unproject_radius(p, depth, radius)
  
  @property
  def depth_map(self):
    return self.scene_widget.depth_map

  def from_numpy(self, a:np.ndarray):
    return torch.from_numpy(a).to(device=self.settings.device)

  @property
  def rendering(self) -> Rendering:
    return self.scene_widget.renderer.rendering
  

  @property
  def renderer(self):
    return self.scene_widget.renderer
  
  @property
  def gaussians(self):
    return self.scene_widget.renderer.gaussians

  def update_setting(self, **kwargs):
    self.scene_widget.update_setting(**kwargs)


