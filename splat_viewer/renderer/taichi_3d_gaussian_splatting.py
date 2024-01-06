
from dataclasses import replace
import math
from beartype import beartype
import torch


from taichi_3d_gaussian_splatting.GaussianPointCloudRasterisation \
  import GaussianPointCloudRasterisation, CameraInfo 

from taichi_3d_gaussian_splatting.utils import SE3_to_quaternion_and_translation_torch

import torch.nn.functional as F
import numpy as np

from splat_viewer.camera.fov import FOVCamera
from splat_viewer.gaussians.data_types import Gaussians, Rendering
from splat_viewer.viewer.scene_camera import fov_to_focal


def pad_to_tile(camera:FOVCamera, m:int):
  def next_mult(x):
      return int(math.ceil(x / m) * m)
      
  w, h = camera.image_size

  round_size = (next_mult(w), next_mult(h))
  return camera.pad_to(np.array(round_size))


default_camera = FOVCamera(position=np.array([0, 0, 0]),
                          rotation=np.eye(3),
                          focal_length=fov_to_focal(60, (640, 480)),
                          image_size = np.array((640, 480)),
                          image_name="viewport")

class TaichiRenderer:
  def __init__(self):
    config = GaussianPointCloudRasterisation.GaussianPointCloudRasterisationConfig(
      near_plane=0.1,
      far_plane=1000.0,
      depth_to_sort_key_scale=10000.0,
    )

    self.renderer = GaussianPointCloudRasterisation(config)

  def camera_info(self, camera:FOVCamera, device=torch.device("cuda:0")):
    return CameraInfo(
      torch.from_numpy(camera.intrinsic).to(dtype=torch.float32, device=device),
      camera_width=camera.image_size[0],
      camera_height=camera.image_size[1],
      camera_id = 0
    )
  
  def camera_qt(self, camera:FOVCamera, device):
    return SE3_to_quaternion_and_translation_torch(
          torch.from_numpy(camera.world_t_camera
          ).unsqueeze(0).to(dtype=torch.float32, device=device))

  @beartype
  def with_camera(self, camera:FOVCamera, inputs:GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput):
    q, t = self.camera_qt(camera, inputs.point_cloud.device)
    return replace(inputs,
      camera_info = self.camera_info(camera, device=inputs.point_cloud.device),
      q_pointcloud_camera=q.contiguous(),
      t_pointcloud_camera=t.contiguous()
    )
      
  @beartype 
  def as_features(self, gaussians:Gaussians):
    n_points, n_feats, _ = gaussians.sh_feature.shape
    sh_features = F.pad(gaussians.sh_feature.permute(0, 2, 1), (0, 16 - n_feats))

    return torch.concatenate([
        gaussians.rotation,
        gaussians.log_scaling,
        gaussians.alpha_logit,
        sh_features.reshape(n_points, -1)
        ], dim=1)

  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):

    return GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput(
      point_cloud = gaussians.position.requires_grad_(requires_grad),
      point_cloud_features = self.as_features(gaussians).requires_grad_(requires_grad),

      camera_info = self.camera_info(default_camera, device=gaussians.device),
      q_pointcloud_camera=torch.eye(4, dtype=torch.float32, device=gaussians.device),
      t_pointcloud_camera=torch.zeros(3, dtype=torch.float32, device=gaussians.device),

      point_object_id = torch.zeros(gaussians.position.shape[0], dtype=torch.int32),
      point_invalid_mask = torch.zeros(gaussians.position.shape[0], dtype=torch.int8), 
      color_max_sh_band = gaussians.sh_degree(),
    )
  

    

  def render(self, inputs:GaussianPointCloudRasterisation.GaussianPointCloudRasterisationInput, 
             camera:FOVCamera, render_depth:bool=False):
    self.renderer.config = replace(self.renderer.config, 
        near_plane=camera.near, far_plane=camera.far,
        rgb_only=not render_depth
        )
    
    w, h = camera.image_size
    padded = pad_to_tile(camera, 16)
    inputs = self.with_camera(padded, inputs)

    image, depth, _ = self.renderer.forward(inputs)
    return Rendering(image=image[:h, :w], 
                     depth=depth[:h, :w] if depth is not None else None, 
                     camera=camera)
