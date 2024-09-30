

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Iterable, List, Optional, Tuple, Set
from beartype import beartype
from immutables import Map
import torch
from splat_viewer.gaussians.data_types import Gaussians




@dataclass(frozen=True)
class Instance:
  id: int
  class_id: int
  name: str
  color: torch.Tensor


def random_color():
  return torch.sigmoid(torch.randn(3))
  

def instances_from_gaussians(gaussians:Gaussians, class_labels:list[str]) -> Map[int, Instance]:
  if gaussians.instance_label is None:
    return Map()
  
  def get_instance(i:int):
    point_indices = (gaussians.instance_label == i).nonzero()[0]
    class_id = gaussians.label[point_indices][0].item()

    return Instance(i, class_id, f"{class_labels[class_id - 1]}_{i}", random_color())

  instance_ids = sorted(torch.unique(gaussians.instance_label).tolist())    
  return Map({i:get_instance(i) for i in instance_ids if i >= 0})


@beartype
@dataclass(frozen=True)
class GaussianScene:
  gaussians: Gaussians
  class_labels: list[str]

  instances: Map[int, Instance] = Map()
  selected_instances: Set[int] = field(default_factory=set)


  def with_unselected(self) -> 'GaussianScene':
    return replace(self, selected_instances=set())

  @beartype
  def with_selected(self, instance_ids:Set[int]) -> 'GaussianScene':
    return replace(self, selected_instances=instance_ids)


  @property
  def device(self) -> torch.device:
    return self.gaussians.device

  @staticmethod
  def empty(device:str | torch.device = torch.device('cpu')):
    return GaussianScene(Gaussians.empty(device=device), ["object"])


  def next_instance_id(self) -> int:
    return max(self.instances.keys(), default=0) 
  


  def add_instance(self, instance:Instance, select:bool = True) -> 'GaussianScene':
    assert instance.id not in self.instances

    instances = self.instances | {instance.id: instance}
    return replace(self, 
          instances=instances, 
          selected_instances={instance.id} if select else self.selected_instances)


  def remove_instance(self, instance_id:int) -> 'GaussianScene':
    assert instance_id in self.instances
    return replace(self, 
          instances=self.instances.delete(instance_id), 
          selected_instances=self.selected_instances - {instance_id})



  

  @cached_property
  def instance_boxes(self) -> Map[int, torch.Tensor]:
    def get_box(instance:Instance):
      point_indices = (self.gaussians.instance_label == instance.id).nonzero()[0]

      points = self.gaussians.position[point_indices]
      return torch.cat([points.min(dim=0).values, points.max(dim=0).values], dim=0)

    return self.instances.map(get_box)

    
  @staticmethod
  def from_gaussians(gaussians:Gaussians, class_labels:list[str]):
    instances = instances_from_gaussians(gaussians, class_labels)
    return GaussianScene(gaussians, class_labels, instances, set())

