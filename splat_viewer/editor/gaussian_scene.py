

from dataclasses import dataclass
from functools import cached_property
from typing import Optional, Tuple
import numpy as np
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
  

def instances_from_gaussians(gaussians:Gaussians, class_labels:list[str]) -> dict[int, Instance]:
  if gaussians.instance_label is None:
    return {}
  
  def get_instance(i:int):
    point_indices = (gaussians.instance_label == i).nonzero()[0]
    class_id = gaussians.label[point_indices][0].item()

    return Instance(i, class_id, class_labels[class_id], random_color())

  instance_ids = torch.unique(gaussians.instance_label).tolist()  
  return {i:get_instance(i) for i in instance_ids}


@dataclass(frozen=True)
class GaussianScene:
  gaussians: Gaussians
  class_labels: list[str]

  instances: dict[int, Instance]
  selected_instance: Optional[int] = None


  @staticmethod
  def empty(device:str = 'cpu'):
    return GaussianScene(Gaussians.empty(device=device), [], {})


  def next_instance_id(self) -> int:
    return max(self.instances.keys(), default=0) 
  


  def add_instance(self, instance:Instance):
    assert instance.id not in self.instances

    self.instances[instance.id] = instance
    self.selected_instance = instance.id



  def remove_instance(self, instance_id:int):
    assert instance_id in self.instances

    del self.instances[instance_id]
    if self.selected_instance == instance_id:
      self.selected_instance = None



  @cached_property
  def instance_colors(self) -> torch.Tensor:
    colors = torch.tensor([i.color for i in self.instances.values()])
    return colors.to(self.gaussians.device)


  @cached_property
  def instance_color_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
    colors = self.instance_colors
    
    mask = self.gaussians.instance_labels > 0
    colors = self.instance_colors[self.gaussians.instance_labels]

    return colors, mask
  

  @cached_property
  def instance_boxes(self) -> dict[int, torch.Tensor]:
    def get_box(instance:Instance):
      point_indices = (self.gaussians.instance_label == instance.id).nonzero()[0]

      points = self.gaussians.position[point_indices]
      return torch.cat([points.min(dim=0).values, points.max(dim=0).values], dim=0)

    return {instance.id: get_box(instance) for instance in self.instances.values()}

    
  @staticmethod
  def from_gaussians(gaussians:Gaussians, class_labels:list[str]):
    instances = instances_from_gaussians(gaussians, class_labels)
    return GaussianScene(gaussians, class_labels, instances)

