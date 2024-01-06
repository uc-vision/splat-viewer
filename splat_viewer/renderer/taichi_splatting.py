
from dataclasses import dataclass, replace
from beartype import beartype
import torch
from splat_viewer.camera.fov import FOVCamera

from taichi_splatting import renderer
from taichi_splatting.data_types import CameraParams

from splat_viewer.gaussians.data_types import Gaussians, Rendering


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
class PackedGaussians:
  gaussians : torch.Tensor
  features : torch.Tensor

  def requires_grad_(self, requires_grad):
    self.gaussians.requires_grad_(requires_grad)
    self.features.requires_grad_(requires_grad)
    return self

class GaussianRenderer:
  @dataclass 
  class Config:
    tile_size : int = 16
    alpha_multiplier : float = 1.0

  def __init__(self, **kwargs):
    self.config = GaussianRenderer.Config(**kwargs)

  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):
      packed = torch.cat(
        [gaussians.position, 
         gaussians.log_scaling, 
         gaussians.rotation, 
         gaussians.alpha_logit], 
         dim=-1)
      
      return PackedGaussians(
        gaussians=packed, 
        features=gaussians.sh_feature).requires_grad_(requires_grad)

  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, inputs:PackedGaussians, camera:FOVCamera, render_depth:bool = True):
    device = inputs.gaussians.device

    config = renderer.RasterConfig(
      tile_size=self.config.tile_size,
    )
    rendering = renderer.render_gaussians(
      inputs.gaussians, inputs.features,
      to_camera_params(camera, device),
      config, 
      use_sh=True, render_depth=render_depth)
    
    return Rendering(rendering.image, 
                            rendering.depth, rendering.depth_var, camera)
    
