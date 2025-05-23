from dataclasses import replace
import math
from beartype.typing import Optional
from tensordict import TensorClass
import torch

from .sh_utils import check_sh_degree, num_sh_features, rgb_to_sh, sh_to_rgb

from dataclasses import dataclass
from splat_viewer.camera.fov import FOVCamera

from taichi_splatting import Gaussians3D
import roma



@dataclass
class Rendering:
  image : torch.Tensor
  camera : FOVCamera

  depth : torch.Tensor

  @property
  def image_size(self):
    y, x = self.image.shape[1:]
    return x, y

class Gaussians(TensorClass):
  position     :  torch.Tensor # 3  - xyz
  log_scaling   : torch.Tensor # 3  - scale = exp(log_scalining) 
  rotation      : torch.Tensor # 4  - quaternion wxyz
  alpha_logit   : torch.Tensor # 1  - alpha = sigmoid(alpha_logit)
  
  sh_feature      :  torch.Tensor # (spherical harmonics (3, deg + 1)**2))
  
  foreground : Optional[torch.Tensor] = None # 1 (bool) 
  label      : Optional[torch.Tensor] = None # 1 (int)
  instance_label : Optional[torch.Tensor] = None # 1 (int)
 
  def __post_init__(self):
    assert self.position.shape[1] == 3, f"Expected shape (N, 3), got {self.position.shape}"
    assert self.log_scaling.shape[1] == 3, f"Expected shape (N, 3), got {self.log_scaling.shape}"
    assert self.rotation.shape[1] == 4, f"Expected shape (N, 4), got {self.rotation.shape}"
    assert self.alpha_logit.shape[1] == 1, f"Expected shape (N, 1), got {self.alpha_logit.shape}"

    check_sh_degree(self.sh_feature)
    assert self.foreground is None or self.foreground.shape[1] == 1, f"Expected shape (N, 1), got {self.foreground.shape}"
    assert self.label is None or self.label.shape[1] == 1, f"Expected shape (N, 1), got {self.label.shape}"
    assert self.instance_label is None or self.instance_label.shape[1] == 1, f"Expected shape (N, 1), got {self.instance_label.shape}"

  def n(self):
    return self.batch_size[0]

  def __repr__(self):
    return f"Gaussians with {self.batch_shape[0]} points, sh_degree={self.sh_degree}"

  def __str__(self):
    return repr(self)
    
  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)
  
  @property
  def device(self):
    return self.position.device 
  
  def to_gaussians3d(self):
    return Gaussians3D(
      position=self.position,
      log_scaling=self.log_scaling,
      rotation=self.rotation,
      alpha_logit=self.alpha_logit,
      feature=self.sh_feature,
      batch_size=self.batch_size
    )

  @staticmethod
  def from_gaussians3d(g:Gaussians3D):
    return Gaussians(
      position=g.position,
      log_scaling=g.log_scaling,
      rotation=g.rotation,
      alpha_logit=g.alpha_logit,
      sh_feature=g.feature,
      batch_size=g.batch_size
    )

  def crop_foreground(self):
    if self.foreground is None:
      return self
    else:
      return self[self.foreground[:, 0]]
    


  def scale(self):
    return torch.exp(self.log_scaling)

  def alpha(self):
    return torch.sigmoid(self.alpha_logit)
  
  def mul_alpha(self, factor) -> 'Gaussians':
    return self.replace(alpha_logit=inverse_sigmoid(self.alpha() * factor))
  
  def split_sh(self):
    return self.sh_feature[:, :, 0], self.sh_feature[:, :, 1:]
  
  def sh_degree(self):
    return check_sh_degree(self.sh_feature)
  
  def with_fixed_scale(self, scale:float):
    return replace(self, 
            log_scaling=torch.full_like(self.log_scaling, math.log(scale)),
            batch_size=self.batch_size)
  
  def get_colors(self):
    return sh_to_rgb(self.sh_feature[:, :, 0])
  
  def get_rotation_matrix(self):
    return roma.unitquat_to_rotmat(self.rotation)
  
  def set_colors(self, color: tuple[float, float, float], indexes: Optional[torch.Tensor]):
    colors = torch.tensor(color, device=self.device).expand(indexes.shape[0], -1)  
    return self.with_colors(colors, indexes)


  def with_colors(self, colors, index=None):
    sh_feature = self.sh_feature.clone()
    if index is None:
      sh_feature[:, :, 0] = rgb_to_sh(colors)
    else:
      sh_feature[index, :, 0] = rgb_to_sh(colors)
    return self.replace(sh_feature=sh_feature)
  
  def with_labels(self, labels, index=None):
    if index is None:
      return self.replace(label=labels)
    else:
      return self.replace(label=self.label[index])
  
  def with_sh_degree(self, sh_degree:int):
    assert sh_degree >= 0

    if sh_degree <= self.sh_degree():
      return self.replace(sh_feature = self.sh_feature[:, :, :num_sh_features(sh_degree)])
    else:
      num_extra = num_sh_features(sh_degree) - num_sh_features(self.sh_degree)
      extra_features = torch.zeros((self.batch_shape[0], 
              3, num_extra), device=self.device)
      
      return self.replace(sh_feature = torch.cat(
        [self.sh_feature, extra_features], dim=2))
    
    
  def replace(self, **kwargs):
    return replace(self, **kwargs, batch_size=self.batch_size)


  def sorted(self):
    max_axis = torch.max(self.log_scaling, dim=1).values
    indices = torch.argsort(max_axis, descending=False)

    return self[indices]
    


def inverse_sigmoid(x, eps=1e-6):
  return -torch.log((1 / (x + eps)) - 1)
