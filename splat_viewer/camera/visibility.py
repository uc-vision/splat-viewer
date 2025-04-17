
from beartype.typing import List
from beartype import beartype
import numpy as np

from tqdm import tqdm

import torch
from .fov import FOVCamera




def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return torch.concatenate([points, torch.ones(shape, dtype=torch.float32, device=points.device)], axis=-1)

def _transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  homog = make_homog(points).reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ homog
  return transformed[..., 0].reshape(-1, 4)

def project_points(transform, xyz):
  homog = _transform_points(transform, xyz)
  depth = homog[..., 2:3]
  xy = homog[..., 0:2] 
  return (xy / depth), depth



@beartype
def visibility(cameras:List[FOVCamera], points:torch.Tensor, near=0.1, far=torch.inf):
  vis_counts = torch.zeros(len(points), dtype=torch.int32, device=points.device)

  projections = np.array([camera.projection for camera in cameras])
  torch_projections = torch.from_numpy(projections).to(dtype=torch.float32, device=points.device)

  min_distance = torch.full((len(points), ), fill_value=far, dtype=torch.float32, device=points.device)

  for camera, proj in tqdm(zip(cameras, torch_projections), total=len(cameras), desc="Evaluating visibility"):
  
    proj, depth = project_points(proj, points)
    width, height = camera.image_size

    is_valid = ((proj[:, 0] >= 0) & (proj[:, 0] < width) & 
             (proj[:, 1] >= 0) & (proj[:, 1] < height)
             & (depth[:, 0] > near)
             )

    min_distance[is_valid] = torch.minimum(depth[is_valid, 0], min_distance[is_valid])
    vis_counts[is_valid] += 1


  return vis_counts, min_distance


