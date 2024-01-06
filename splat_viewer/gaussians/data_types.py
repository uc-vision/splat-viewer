from dataclasses import replace
import math
from typing import Optional
from tensordict import tensorclass
import torch

from .sh_utils import check_sh_degree, num_sh_features, rgb_to_sh, sh_to_rgb

from dataclasses import dataclass
from splat_viewer.camera.fov import FOVCamera



@dataclass
class Rendering:
  image : torch.Tensor
  camera : FOVCamera

  depth : Optional[torch.Tensor] = None
  depth_var: Optional[torch.Tensor] = None

  @property
  def image_size(self):
    y, x = self.image.shape[1:]
    return x, y

@tensorclass
class Gaussians():
  position     :  torch.Tensor # 3  - xyz
  log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 4  - quaternion wxyz
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  sh_feature      :  torch.Tensor # (any rgb (3), spherical harmonics (3x16) etc)
  
  foreground : Optional[torch.Tensor] # 1 (bool) 
  label      : Optional[torch.Tensor] # 1 (int)

  def __post_init__(self):
    assert self.position.shape[1] == 3, f"Expected shape (N, 3), got {self.position.shape}"
    assert self.log_scaling.shape[1] == 3, f"Expected shape (N, 3), got {self.log_scaling.shape}"
    assert self.rotation.shape[1] == 4, f"Expected shape (N, 4), got {self.rotation.shape}"
    assert self.alpha_logit.shape[1] == 1, f"Expected shape (N, 1), got {self.alpha_logit.shape}"

    check_sh_degree(self.sh_feature)
    assert self.foreground is None or self.foreground.shape[1] == 1, f"Expected shape (N, 1), got {self.foreground.shape}"
    assert self.label is None or self.label.shape[1] == 1, f"Expected shape (N, 1), got {self.label.shape}"

  def __repr__(self):
    return f"Gaussians with {self.batch_shape[0]} points, sh_degree={self.sh_degree}"

  def __str__(self):
    return repr(self)
    
  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)

  def scale(self):
    return torch.exp(self.log_scaling)

  def alpha(self):
    return torch.sigmoid(self.alpha_logit)
  
  def sh_degree(self):
    return check_sh_degree(self.sh_feature)
  
  def with_fixed_scale(self, scale:float):
    return replace(self, 
            log_scaling=torch.full_like(self.log_scaling, math.log(scale)),
            batch_size=self.batch_size)
  
  def get_colors(self):
    return sh_to_rgb(self.sh_feature[:, :, 0])
  

  def with_colors(self, colors):
    sh_feature = self.sh_feature.clone()
    sh_feature[:, :, 0] = rgb_to_sh(colors)

    return replace(self, 
            sh_feature=sh_feature,
            batch_size=self.batch_size)
  
  def with_sh_degree(self, sh_degree:int):
    assert sh_degree >= 0

    if sh_degree <= self.sh_degree():
      return replace(self, 
          sh_feature = self.sh_feature[:, :, :num_sh_features(sh_degree)],
          batch_size=self.batch_size)
    else:
      num_extra = num_sh_features(sh_degree) - num_sh_features(self.sh_degree)
      extra_features = torch.zeros((self.batch_shape[0], 
              3, num_extra), device=self.device)
      
      return replace(self, sh_feature = torch.cat(
        [self.sh_feature, extra_features], dim=2), 
        batch_size=self.batch_size)
    
  def sorted(self):
    max_axis = torch.max(self.log_scaling, dim=1).values
    indices = torch.argsort(max_axis, descending=False)

    return self[indices]
    

