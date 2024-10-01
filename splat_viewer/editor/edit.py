

from abc import ABCMeta, abstractmethod
from dataclasses import replace
from typing import  Set, Tuple

from immutables import Map
import torch
from splat_viewer.editor.gaussian_scene import GaussianScene, Instance, random_color




class Edit( metaclass=ABCMeta):
  @abstractmethod
  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    raise NotImplementedError



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


class DeleteInstances(Edit):
  def __init__(self, instance_ids:Set[int]):
    self.instance_ids = instance_ids

  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    instances = {i:scene.instances[i] for i in self.instance_ids}
    return scene.remove_instances(self.instance_ids), AddInstances(instances)


class AddInstances(Edit):
  def __init__(self, instances:Map[int, Instance]):
    self.instances = instances

  def apply(self, scene:GaussianScene) -> Tuple[GaussianScene, 'Edit']:
    return scene.add_instances(self.instances), DeleteInstances(set(self.instances.keys()))

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








