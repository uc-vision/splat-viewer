
from dataclasses import dataclass, replace
from beartype import beartype
import torch
from splat_viewer.camera.fov import FOVCamera
from .data_types import Gaussians

from taichi_splatting import renderer
from taichi_splatting.data_types import CameraParams


def to_camera_params(camera:FOVCamera, device=torch.device("cuda:0")):

  params = CameraParams(
    T_camera_world=torch.from_numpy(camera.camera_t_world),
    T_image_camera=torch.from_numpy(camera.intrinsic),
    
    image_size=tuple(camera.image_size),
    near_plane=camera.near,
    far_plane=camera.far
  )

  return params.to(device=device, dtype=torch.float32)


@dataclass
class DepthRendering:
  image : torch.Tensor
  depth : torch.Tensor
  camera : FOVCamera

  @property
  def image_size(self):
    y, x = self.image.shape[1:]
    return x, y
    

@dataclass
class PackedGaussians:
  gaussians : torch.Tensor
  features : torch.Tensor

  def requires_grad_(self, requires_grad):
    self.gaussians.requires_grad_(requires_grad)
    self.features.requires_grad_(requires_grad)

class GaussianRenderer:
  @dataclass 
  class Config:
    tile_size : int = 16
    alpha_multiplier : float = 1.0

  def __init__(self, **kwargs):
    self.config = GaussianRenderer.Config(**kwargs)

  @beartype
  def pack_inputs(self, gaussians:Gaussians):
      packed = torch.cat(
        [gaussians.position, 
         gaussians.log_scaling, 
         gaussians.rotation, 
         gaussians.alpha_logit], 
         dim=-1)
      
      return PackedGaussians(
        gaussians=packed, 
        features=gaussians.sh_feature)

  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, inputs:PackedGaussians, camera:FOVCamera):
    device = inputs.gaussians.device

    config = renderer.RasterConfig(
      tile_size=self.config.tile_size,
    )
    image, depth = renderer.render_sh_gaussians(
      inputs.gaussians, inputs.features,
      to_camera_params(camera, device),
      config)
    
    depth[depth == 0] = torch.inf

    return DepthRendering(image.contiguous(), depth, camera) 

