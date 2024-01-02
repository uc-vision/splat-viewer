
from dataclasses import dataclass, replace
from beartype import beartype
import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.gaussians.gaussians import Gaussians

from taichi_splatting import renderer
from taichi_splatting.data_types import CameraParams, Gaussians3D

@dataclass
class RenderOutputs:
  image : torch.Tensor
  depth : torch.Tensor
  camera : FOVCamera

  @property
  def image_size(self):
    y, x = self.image.shape[1:]
    return x, y
  
@dataclass 
class Config:
  tile_size : int = 16
  alpha_multiplier : float = 1.0
  
def to_camera_params(camera:FOVCamera, device=torch.device("cuda:0")):
  def to_torch(x):
    return torch.tensor(x, dtype=torch.float32, device=device)


  return CameraParams(
    T_camera_world=to_torch(camera.camera_t_world),
    T_image_camera=to_torch(camera.intrinsic),
    
    image_size=tuple(camera.image_size),
    near_plane=camera.near,
    far_plane=camera.far
  )

class GaussianRenderer:
  def __init__(self, config:Config = Config()):
    self.config = config



  @beartype
  def make_inputs(self, gaussians:Gaussians):
      packed = torch.cat(
        [gaussians.position, 
         gaussians.log_scaling, 
         gaussians.rotation, 
         gaussians.alpha_logit], 
         dim=-1)
      
      return packed, gaussians.sh_feature

  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render_gaussians(self, gaussians:Gaussians, camera:FOVCamera):
    inputs = self.make_inputs(gaussians)
    return self.render_inputs(inputs, camera)
    

  def render_inputs(self, inputs, camera:FOVCamera):
    device = inputs[0].device

    image, depth = renderer.render_sh_gaussians(inputs, 
              to_camera_params(camera, device), 
              tile_size=self.config.tile_size,)

    config = renderer.RasterConfig(
      tile_size=self.config.tile_size,
    )
    image, depth = renderer.render_gaussians(*inputs, 
              to_camera_params(camera, device),
              config)
    
    return RenderOutputs(image.contiguous(), depth, camera) 

