import math
from typing import Optional
from tensorclass import tensorclass
import torch

def check_sh_degree(sh_features):
  assert len(sh_features.shape) == 3, f"SH features must have 3 dimensions, got {sh_features.shape}"

  n_sh = sh_features.shape[2]
  n = int(math.sqrt(n_sh))

  assert n * n == n_sh, f"SH feature count must be square, got {n_sh} ({sh_features.shape})"
  return (n - 1)



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

    
  def packed(self):
    return torch.cat([self.position, self.log_scaling, self.rotation, self.alpha_logit], dim=-1)

  @property
  def scale(self):
    return torch.exp(self.log_scaling)

  @property
  def alpha(self):
    return torch.sigmoid(self.alpha_logit)
  
  @property
  def sh_degree(self):
    return check_sh_degree(self.sh_feature)