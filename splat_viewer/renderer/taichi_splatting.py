
from dataclasses import dataclass, replace
from typing import Tuple
from beartype import beartype
import torch
from splat_viewer.camera.fov import FOVCamera

from taichi_splatting import Gaussians3D, renderer
from taichi_splatting.perspective import CameraParams

from splat_viewer.gaussians.data_types import Gaussians, Rendering


def to_camera_params(camera:FOVCamera, device=torch.device("cuda:0")):

  params = CameraParams(
    T_camera_world=torch.from_numpy(camera.camera_t_world),
    projection=torch.tensor([*camera.focal_length, *camera.principal_point]),
    
    image_size=tuple(camera.image_size),
    near_plane=camera.near,
    far_plane=camera.far
  )

  return params.to(device=device, dtype=torch.float32)
    

class GaussianRenderer:
  @dataclass 
  class Config:
    tile_size : int = 16
    use_depth16 : bool = False
    pixel_stride : Tuple[int, int] = (2, 2)
    antialias : bool = True
    blur_cov: float = 0.3

  def __init__(self, **kwargs):
    self.config = GaussianRenderer.Config(**kwargs)

  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):
      return gaussians.to_gaussians3d().requires_grad_(requires_grad)

  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, inputs:Gaussians3D, camera:FOVCamera, render_depth:bool = True):
    device = inputs.position.device
    
    config = renderer.RasterConfig(
      tile_size=self.config.tile_size,
      pixel_stride=self.config.pixel_stride,
      antialias=self.config.antialias,
      blur_cov=self.config.blur_cov,
      saturate_threshold=0.5

    )
      
    rendering = renderer.render_gaussians(
      gaussians=inputs, 
      camera_params=to_camera_params(camera, device),
      config=config, 
      use_sh=True, 
      use_depth16=self.config.use_depth16,
      render_depth=render_depth)
    
    weight = 1 / (rendering.image_weight + 1e-6)

    return Rendering(image=rendering.image * weight.unsqueeze(-1), 
                            depth=rendering.depth, 
                            depth_var=rendering.depth_var, 
                             camera=camera)
    
