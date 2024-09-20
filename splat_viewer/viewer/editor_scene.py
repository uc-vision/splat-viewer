

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Generic, List, Tuple, TypeVar

import torch

from splat_viewer.gaussians.data_types import Gaussians


@dataclass(frozen=True)
class Instance:
  id: int
  class_id: int

  name: str
  

def instances_from_gaussians(gaussians:Gaussians, class_labels:list[str]) -> dict[int, Instance]:
  if gaussians.instance_labels is None:
    return {}
  
  def get_instance(i:int):
    idx = (gaussians.instance_labels == i).nonzero()[0]
    class_id = gaussians.label[idx][0].item()

    return Instance(i, class_id, class_labels[class_id])

  instance_ids = torch.unique(gaussians.instance_labels).tolist()  
  return {i:get_instance(i) for i in instance_ids}


@dataclass(frozen=True)
class GaussianScene:
  gaussians: Gaussians
  class_labels: list[str]
  instances: dict[int, Instance]


  @staticmethod
  def from_gaussians(gaussians:Gaussians, class_labels:list[str]):
    instances = instances_from_gaussians(gaussians, class_labels)
    return GaussianScene(gaussians, class_labels, instances)


T = TypeVar("T")

class Edit(Generic[T], metaclass=ABCMeta):
  @abstractmethod
  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, T]:
    raise NotImplementedError

  @abstractmethod
  def undo(self, scene:GaussianScene) -> 'Edit[T]':
    raise NotImplementedError


@dataclass
class Editor:
  scene: GaussianScene

  undos : List[Edit] = field(default_factory=list)
  redos : List[Edit] = field(default_factory=list)

  def apply(self, edit:Edit):
    self.scene = edit.apply(self.scene)
    self.undos.append(edit)
    self.redos.clear()

  def undo(self):
    if not self.undos:
      return
    
    edit = self.undos.pop()
    self.scene = edit.undo(self.scene)
    self.redos.append(edit)

  def redo(self):
    if not self.redos:
      return
    
    edit = self.redos.pop()
    self.scene = edit.apply(self.scene)
    self.undos.append(edit)


    
class AddInstance(Edit):
  def __init__(self, instance:Instance):
    self.instance = instance

  def apply(self, scene:GaussianScene) -> GaussianScene:
    raise NotImplementedError
    
  def undo(self, scene:GaussianScene) -> 'AddInstance':
    return RemoveInstance(self.instance)


def RemoveInstance(Edit):
  def __init__(self, instance:Instance):
    self.instance = instance

  def apply(self, scene:GaussianScene) -> GaussianScene:  
    scene.gaussians 
    
    



class ModifyInstances(Edit):
  def __init__(self, indexes:torch.Tensor, instance_ids:torch.Tensor, class_ids:torch.Tensor):
    self.indexes = indexes
    self.instance_ids = instance_ids
    self.class_ids = class_ids

  def apply(self, scene:GaussianScene) -> GaussianScene:
    return replace(scene, gaussians=replace(scene.gaussians, 
      instance_labels=torch.index_put(scene.gaussians.instance_labels, (self.indexes,), self.instance_ids), 
      label=torch.index_put(scene.gaussians.label, (self.indexes,), self.class_ids)))

  
  def undo(self, scene:GaussianScene) -> 'ModifyInstances':
    restore_ids = scene.gaussians.instance_labels[self.indexes]
    restore_class_ids = scene.gaussians.label[self.indexes]
    return ModifyInstances(self.indexes, restore_ids, restore_class_ids)

  @staticmethod 
  def paint_instance(scene:GaussianScene, instance_id:int, indexes:torch.Tensor):
    instance_ids = torch.full_like(indexes, instance_id)
    class_ids = scene.gaussians.label[indexes]

    return ModifyInstances(indexes, instance_ids, class_ids)



