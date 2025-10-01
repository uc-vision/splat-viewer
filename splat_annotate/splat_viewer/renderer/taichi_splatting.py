
from dataclasses import dataclass, replace
from beartype import beartype
import torch
from splat_annotate.camera.fov import FOVCamera

from taichi_splatting import Gaussians3D, renderer
from taichi_splatting.perspective import CameraParams

from splat_annotate.gaussians.data_types import Gaussians, Rendering


def to_camera_params(camera:FOVCamera, device=torch.device("cuda:0")):
  w, h = camera.image_size
  params = CameraParams(
    T_camera_world=torch.from_numpy(camera.camera_t_world),
    projection=torch.tensor([*camera.focal_length, *camera.principal_point]),
    
    image_size=(int(w), int(h)),
    near_plane=camera.near,
    far_plane=camera.far
  )

  return params.to(device=device, dtype=torch.float32)
    

class GaussianRenderer:
  @dataclass 
  class Config:
    antialias : bool = False
    blur_cov: float = 0.3

  def __init__(self, **kwargs):

    
    self.config = GaussianRenderer.Config(**kwargs)

  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):
      return gaussians.to_gaussians3d().requires_grad_(requires_grad)

  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, inputs:Gaussians3D, camera:FOVCamera):
    device = inputs.position.device
    
    config = renderer.RasterConfig(
      antialias=self.config.antialias,
      blur_cov=self.config.blur_cov
    )
      
    rendering = renderer.render_gaussians(
      gaussians=inputs, 
      camera_params=to_camera_params(camera, device),
      config=config, 
      use_sh=True, 
      render_median_depth=True)
    
    depth = rendering.median_depth_image
    assert depth is not None
   
    return Rendering(image=rendering.image, depth=depth, camera=camera)
    
