
from dataclasses import replace
from beartype import beartype
import torch
from splat_viewer.camera.fov import FOVCamera

import ray_splatting
from splat_viewer.gaussians.data_types import Gaussians, Rendering


def to_camera_params(camera:FOVCamera, device=torch.device("cuda:0")):

  params = ray_splatting.Camera(
    world_T_camera=torch.from_numpy(camera.world_t_camera),
    projection=torch.tensor([*camera.focal_length, *camera.principal_point]),
    
    image_size=tuple(int(x) for x in camera.image_size),
    near_plane=camera.near,
    far_plane=camera.far
  )

  return params.to(device=device, dtype=torch.float32)
    

class GaussianRenderer:
  def __init__(self):
    config = ray_splatting.RenderConfig(
      max_hits = 16,
      kernel_function = ray_splatting.KernelFunction.HalfCosine,
      alpha_threshold=1/255,
      num_features=3
    )

    self.renderer = ray_splatting.Renderer(config, device=torch.device("cuda:0"))
    self.last_points = None


# class Point3D(TensorClass):
#   position     : torch.Tensor # 3  - xyz
#   log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
#   rotation      : torch.Tensor # 4  - quaternion wxyz

#   alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
#   feature      : torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)


  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):
      
      return ray_splatting.Point3D(
          position=gaussians.position,
          feature=gaussians.get_colors(),
          log_scaling=gaussians.log_scaling,
          rotation=gaussians.rotation,
          alpha_logit=gaussians.alpha_logit,
          batch_size=gaussians.batch_size
      )
  
  def update_settings(self, **kwargs):
    self.config = replace(self.config, **kwargs)

  @beartype
  def render(self, points:ray_splatting.Point3D, camera:FOVCamera):
    
    rendering:ray_splatting.Rendering = self.renderer.render(
      camera=to_camera_params(camera),
      points=points, 
      scene_changed=id(points) != self.last_points)
      
    depth = torch.zeros_like(rendering.feature_image[..., 0])
    self.last_points = id(points)
    
    return Rendering(image=rendering.feature_image, 
                            depth=depth, 
                            camera=camera)
    
