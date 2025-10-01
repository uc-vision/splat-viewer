
from typing import Self
from PySide6 import QtGui
from PySide6.QtCore import QEvent

from beartype import beartype
import numpy as np
import torch
from splat_annotate.gaussians.data_types import Gaussians, Rendering

from splat_annotate.viewer.settings import Settings


class Interaction():
  def __init__(self):
    super(Interaction, self).__init__()
    self._child = None 
    self.active = False

  def transition(self, interaction:Self | None):
    self.pop()
    if interaction is not None:
      self.push(interaction)

  def push(self, interaction:Self):
    self._child = interaction
    if self.active:
      self._child._activate()


  def pop(self):    
    if self._child is not None:
      child = self._child
      self._child = None
      child._deactivate()


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
      QEvent.Type.KeyPress: self.keyPressEvent,
      QEvent.Type.KeyRelease: self.keyReleaseEvent,
      QEvent.Type.MouseButtonPress: self.mousePressEvent,
      QEvent.Type.MouseButtonRelease: self.mouseReleaseEvent,
      QEvent.Type.MouseMove: self.mouseMoveEvent,
      QEvent.Type.Wheel: self.wheelEvent,
      QEvent.Type.FocusIn: self.focusInEvent,
      QEvent.Type.FocusOut: self.focusOutEvent,
    }

    if event.type() in event_callbacks:
      return event_callbacks[event.type()](event) or False
    
    return False
    
  def trigger_paint(self, event: QtGui.QPaintEvent, view_changed:bool) -> bool:
    if self._child is not None:
      if self._child.trigger_paint(event, view_changed):
        return True
      
    return self.paintEvent(event, view_changed) 

  def keyPressEvent(self, event: QtGui.QKeyEvent) -> bool:
    return False

  def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> bool:
    return False

  def mousePressEvent(self, event: QtGui.QMouseEvent) -> bool:
    return False

  def mouseReleaseEvent(self, event: QtGui.QMouseEvent) -> bool:
    return False

  def mouseMoveEvent(self, event: QtGui.QMouseEvent) -> bool:
    return False

  def wheelEvent(self, event: QtGui.QWheelEvent) -> bool:
    return False

  def focusInEvent(self, event: QtGui.QFocusEvent) -> bool:
    return False
  
  def focusOutEvent(self, event: QtGui.QFocusEvent) -> bool:
    return False
  
  def paintEvent(self, event: QtGui.QPaintEvent, view_changed:bool) -> bool:
    return False

  @beartype
  def update(self, dt:float) -> bool:
    return False

  def on_activate(self) -> None:
    pass

  def on_deactivate(self) -> None:
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
  def keys_down(self) -> set[QtGui.Qt.Key]:
    return self.scene_widget.keys_down
  
  @property
  def cursor_pos(self) -> np.ndarray:
    return self.scene_widget.cursor_pos
  
  @property
  def current_point(self) -> np.ndarray:
    return self.scene_widget.current_point_3d

  def lookup_point_3d(self, p:np.ndarray) -> np.ndarray:
    return self.scene_widget.lookup_point_3d(p)

  def lookup_depth(self, p:np.ndarray) -> float:
    return self.scene_widget.lookup_depth(p)
  

  def lookup_depths(self, p:np.ndarray) -> np.ndarray:
    return self.scene_widget.lookup_depths(p)
  
  def test_depths(self, p:np.ndarray, depth:np.ndarray) -> np.ndarray:
    return self.scene_widget.test_depths(p, depth)
  
  def unproject_point(self, p:np.ndarray, depth:float) -> np.ndarray:
    return self.scene_widget.unproject_point(p, depth)
  
  def unproject_radius(self, p:np.ndarray, depth:float, radius:float
                        ) -> tuple[np.ndarray, float]:
      return self.scene_widget.unproject_radius(p, depth, radius)
  
  def set_dirty(self):
    self.scene_widget.set_dirty()
  
  @property
  def depth_map(self):
    return self.scene_widget.depth_map

  def from_numpy(self, a:np.ndarray):
    return torch.from_numpy(a).to(device=self.settings.device)


  @property
  def rendering(self) -> Rendering:
    return self.renderer.rendering
  

  @property
  def renderer(self):
    assert self.scene_widget.renderer is not None
    return self.scene_widget.renderer
  
  @property
  def gaussians(self) -> Gaussians:
    return self.scene_widget.gaussians
  
  def update_gaussians(self, gaussians:Gaussians):
    return self.scene_widget.update_gaussians(gaussians)

  def update_setting(self, **kwargs):
    self.scene_widget.update_setting(**kwargs)


