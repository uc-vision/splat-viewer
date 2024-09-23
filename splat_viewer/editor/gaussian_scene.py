

from dataclasses import dataclass
from functools import cached_property
from typing import Tuple
import numpy as np
import torch

from splat_viewer.gaussians.data_types import Gaussians


@dataclass(frozen=True)
class Instance:
  id: int
  class_id: int
  name: str

  color: torch.Tensor
  point_indices: torch.Tensor
  

def instances_from_gaussians(gaussians:Gaussians, class_labels:list[str]) -> dict[int, Instance]:
  if gaussians.instance_label is None:
    return {}
  
  def get_instance(i:int):
    point_indices = (gaussians.instance_label == i).nonzero()[0]
    class_id = gaussians.label[point_indices][0].item()

    color = torch.sigmoid(torch.randn(3))

    return Instance(i, class_id, class_labels[class_id], color, point_indices)

  instance_ids = torch.unique(gaussians.instance_label).tolist()  
  return {i:get_instance(i) for i in instance_ids}


@dataclass(frozen=True)
class GaussianScene:
  gaussians: Gaussians
  class_labels: list[str]

  instances: dict[int, Instance]



  @staticmethod
  def empty(device:str = 'cpu'):
    return GaussianScene(Gaussians.empty(device='cpu'), [], {})


  @cached_property
  def instance_colors(self) -> torch.Tensor:
    colors = torch.tensor([i.color for i in self.instances.values()])
    return colors.to(self.gaussians.device)


  @cached_property
  def instance_color_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
    colors = self.instance_colors
    
    mask = torch.zeros(self.gaussians.n(), dtype=torch.bool, device=self.gaussians.device)
    colors = torch.zeros(self.gaussians.n(), 3, device=self.gaussians.device)
    
    for instance in self.instances.values():
      mask[instance.point_indices] = True
      colors[instance.point_indices] = instance.color

    return colors, mask
  

  @cached_property
  def instance_boxes(self) -> dict[int, torch.Tensor]:
    def get_box(instance:Instance):
      points = self.gaussians.position[instance.point_indices]
      return torch.cat([points.min(dim=0).values, points.max(dim=0).values], dim=0)

    return {instance.id: get_box(instance) for instance in self.instances.values()}

    
  @staticmethod
  def from_gaussians(gaussians:Gaussians, class_labels:list[str]):
    instances = instances_from_gaussians(gaussians, class_labels)
    return GaussianScene(gaussians, class_labels, instances)

