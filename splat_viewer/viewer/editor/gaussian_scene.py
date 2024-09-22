

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

