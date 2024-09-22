

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generic, List, Optional, Tuple, TypeVar

import torch

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.editor.gaussian_scene import GaussianScene, Instance





class Edit( metaclass=ABCMeta):
  @abstractmethod
  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    raise NotImplementedError


@dataclass
class Editor:
  scene: GaussianScene

  undos : List[Edit] = field(default_factory=list)
  redos : List[Edit] = field(default_factory=list)

  def apply(self, edit:Edit):
    self.scene, undo_edit = edit.apply(self.scene)
    self.undos.append(undo_edit)
    self.redos.clear()

  def undo(self) -> bool:
    if not self.undos:
      return False
    
    edit = self.undos.pop()
    self.scene, redo_edit = edit.undo(self.scene)
    self.redos.append(redo_edit)

  def redo(self):
    if not self.redos:
      return False
    
    edit = self.redos.pop()
    self.scene, undo_edit = edit.apply(self.scene)
    self.undos.append(undo_edit)

    return True
    
  @property
  def can_undo(self) -> bool:
    return len(self.undos) > 0

  @property
  def can_redo(self) -> bool:
    return len(self.redos) > 0


class AddInstance(Edit):
  def __init__(self, instance:Instance):
    self.instance = instance

  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    new_scene = scene.add_instance(self.instance)
    return new_scene, RemoveInstance(self.instance.id)
    




class RemoveInstance(Edit):
  def __init__(self, instance_id:int):
    self.instance_id = instance_id

  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:  
    instance = scene.instances[self.instance_id]
    return scene.remove_instance(self.instance_id), AddInstance(instance)




class ModifyInstances(Edit):
  def __init__(self, indexes:torch.Tensor, instance_ids:torch.Tensor, class_ids:torch.Tensor):
    self.indexes = indexes
    self.instance_ids = instance_ids
    self.class_ids = class_ids

  def apply(self, scene:GaussianScene) -> GaussianScene:
    restore_ids = scene.gaussians.instance_labels[self.indexes]
    restore_class_ids = scene.gaussians.label[self.indexes]
    undo_edit = ModifyInstances(self.indexes, restore_ids, restore_class_ids)


    scene = replace(scene, gaussians=replace(scene.gaussians, 
      instance_labels=torch.index_put(scene.gaussians.instance_labels, (self.indexes,), self.instance_ids), 
      label=torch.index_put(scene.gaussians.label, (self.indexes,), self.class_ids)))

    return scene, undo_edit

  

  @staticmethod 
  def paint_instance(scene:GaussianScene, instance_id:int, indexes:torch.Tensor):
    instance_ids = torch.full_like(indexes, instance_id)
    class_ids = scene.gaussians.label[indexes]

    return ModifyInstances(indexes, instance_ids, class_ids)



