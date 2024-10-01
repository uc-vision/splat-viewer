

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


@dataclass
class PointLabels:
  indexes: torch.Tensor
  instance_id: torch.Tensor
  class_id: torch.Tensor


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
  
  def modify_points(self, indexes:torch.Tensor, instance_ids:torch.Tensor, class_ids:torch.Tensor) -> 'GaussianScene':
    return replace(self, 
          gaussians=replace(self.gaussians, 
          instance_label=torch.index_put(self.gaussians.instance_label, (indexes,), instance_ids), 
          label=torch.index_put(self.gaussians.label, (indexes,), class_ids)))

  def add_instances(self, instances:Map[int, Instance], select:bool = True) -> 'GaussianScene':
    assert all(instance.id not in self.instances for instance in instances.values())

    instances = self.instances | instances
    return replace(self, 
          instances=instances, 
          selected_instances=set(instances.keys()) if select else self.selected_instances)


  def remove_instances(self, instance_ids:Set[int]) -> 'GaussianScene':
    assert all(instance_id in self.instances for instance_id in instance_ids)


    return replace(self, 
          instances=self.instances.delete(instance_ids), 
          selected_instances=self.selected_instances - instance_ids)



  

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

