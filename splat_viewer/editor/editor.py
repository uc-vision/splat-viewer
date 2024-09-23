

from abc import ABCMeta, abstractmethod
from typing import List, Tuple

from splat_viewer.editor.gaussian_scene import GaussianScene

from PySide6.QtCore import QObject, Signal



class Edit( metaclass=ABCMeta):
  @abstractmethod
  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    raise NotImplementedError



class Editor(QObject):
  scene_changed = Signal()

  def __init__(self, scene:GaussianScene, parent:QObject = None):
    super().__init__(parent)

    self.scene = scene 

    self.undos : List[Edit] = []
    self.redos : List[Edit] = []

  def update_edit(self, edit:Edit):
    """
    Replace the last edit in the undo stack.
    """
    assert len(self.undos) > 0

    last_undo = self.undos.pop()
    scene, _ = last_undo.apply(self.scene)
    self.scene, undo = edit.apply(scene)
    self.undos.append(undo)

    self.scene_changed.emit()

  def set_scene(self, scene:GaussianScene):
    self.scene = scene
    self.undos.clear()
    self.redos.clear()

    self.scene_changed.emit()


  def apply(self, edit:Edit):
    if self.scene is None:
      raise ValueError("No active scene")
    
    self.scene, undo_edit = edit.apply(self.scene)
    self.undos.append(undo_edit)
    self.redos.clear()
    self.scene_changed.emit()
  
  def undo(self) -> bool:
    if not self.undos:
      return False
    
    edit = self.undos.pop()
    self.scene, redo_edit = edit.undo(self.scene)
    self.redos.append(redo_edit)
    self.scene_changed.emit()
    return True
  
  def redo(self) -> bool:
    if not self.redos:
      return False
    
    edit = self.redos.pop()
    self.scene, undo_edit = edit.apply(self.scene)
    self.undos.append(undo_edit)
    self.scene_changed.emit()

    return True
  
  @property
  def can_undo(self) -> bool:
    return len(self.undos) > 0

  @property
  def can_redo(self) -> bool:
    return len(self.redos) > 0




