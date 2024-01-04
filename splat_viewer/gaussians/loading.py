
from pathlib import Path
import tempfile
import open3d as o3d
import open3d.core as o3c

import numpy as np
import torch
import torch.nn.functional as F

from .data_types import Gaussians


def torch_to_o3d(tensor:torch.Tensor) -> o3d.core.Tensor:
  return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

def o3d_to_torch(tensor:o3c.Tensor) -> torch.Tensor:
  return torch.from_dlpack(o3d.core.Tensor.to_dlpack(tensor))



def to_pcd(gaussians:Gaussians) -> o3d.t.geometry.PointCloud:
  pos = torch_to_o3d(gaussians.positions)

  pcd = o3d.t.geometry.PointCloud(pos.device)
  pcd.point['positions'] = pos
  pcd.point['opacity'] = torch_to_o3d(gaussians.alpha_logit)

  sh_dc, sh_rest = gaussians.split_sh()

  sh_dc = sh_dc.view(-1, 3)
  sh_rest = sh_rest.permute(0, 2, 1).reshape(sh_rest.shape[0], sh_rest.shape[1] * sh_rest.shape[2])

  for i in range(3):
    pcd.point[f'f_dc_{i}'] = torch_to_o3d(sh_dc[:, i:i+1])

  for i in range(sh_rest.shape[-1]):
    pcd.point[f'f_rest_{i}'] = torch_to_o3d(sh_rest[:, :, i:i+1])
  

  for i in range(3):
    pcd.point[f'scale_{i}'] = torch_to_o3d(gaussians.log_scaling[:, i:i+1])


  # convert back to wxyz quaternion
  rotation = torch.roll(gaussians.rotation, 1, dims=(1,))

  for i in range(4):
    pcd.point[f'rot_{i}'] = torch_to_o3d(rotation[:, i:i+1])

  if gaussians.foreground is not None:
    pcd.point['foreground'] = torch_to_o3d(gaussians.foreground.to(torch.int32))

  if gaussians.label is not None:
    pcd.point['label'] = torch_to_o3d(gaussians.label)
  

  return pcd



def from_pcd(pcd:o3d.t.geometry.PointCloud) -> Gaussians:
  def get_keys(ks):
    values = [o3d_to_torch(pcd.point[k]) for k in ks]
    return torch.concat(values, dim=-1)

  positions = o3d_to_torch(pcd.point['positions'])

  attrs = sorted(dir(pcd.point))
  sh_attrs = [k for k in attrs if k.startswith('f_rest_') or k.startswith('f_dc_')]
  
  n_sh = len(sh_attrs) // 3
  deg = int(np.sqrt(n_sh))

  assert deg * deg == n_sh, f"SH feature count must be (3x) square, got {len(sh_attrs)}"
  log_scaling = get_keys([f'scale_{k}' for k in range(3)])
  log_scaling = log_scaling.clamp_min(-8)


  sh_dc = get_keys([f'f_dc_{k}' for k in range(3)]).view(positions.shape[0], 3, 1)
  sh_rest = get_keys([f'f_rest_{k}' for k in range(3 * (deg * deg - 1))])
  sh_rest = sh_rest.view(positions.shape[0], 3, n_sh - 1)

  sh_features = torch.cat([sh_dc, sh_rest], dim=2)  

  rotation = get_keys([f'rot_{k}' for k in range(4)])
  # convert from wxyz to xyzw quaternion and normalize
  rotation = torch.roll(F.normalize(rotation), -1, dims=(1,))
  
  alpha_logit = get_keys(['opacity'])

  
  foreground = (get_keys(['foreground']).to(torch.bool) 
    if 'foreground' in pcd.point else None)
  
  label = get_keys(['label']) if 'label' in pcd.point else None
  

  return Gaussians(
    position = positions, 
    rotation = rotation,
    alpha_logit = alpha_logit,
    log_scaling = log_scaling,
    sh_feature = sh_features,

    foreground = foreground,
    label = label,

    batch_size = (positions.shape[0],)
  )


def write_gaussians(filename:Path | str, gaussians:Gaussians):
  filename = Path(filename)

  pcd = to_pcd(gaussians)
  o3d.t.io.write_point_cloud(str(filename), pcd)

def read_gaussians(filename:Path | str) -> Gaussians:
  filename = Path(filename) 

  pcd:o3d.t.geometry.PointCloud = o3d.t.io.read_point_cloud(str(filename))
  if 'positions' not in pcd.point:
    raise ValueError(f"Could not load point cloud from {filename}")

  return from_pcd(pcd)
  

def random_gaussians(n:int, sh_degree:int):
  points = torch.randn(n, 3)

  return Gaussians( 
    position = points,
    rotation = F.normalize(torch.randn(n, 4), dim=1),
    alpha_logit = torch.randn(n, 1),
    log_scaling = torch.randn(n, 3) * 4,

    sh_feature = torch.randn(n, (sh_degree + 1)**2, 3),
  )

def test_read_write():
  temp_path = Path(tempfile.mkdtemp())

  print("Testing write/read")
  for i in range(10):
    g = random_gaussians((i + 1) * 1000, 3)
    write_gaussians(temp_path / f'gaussians_{i}.ply', g)
    g2 = read_gaussians(temp_path / f'gaussians_{i}.ply')

    assert torch.allclose(g.position, g2.position)
    assert torch.allclose(g.rotation, g2.rotation)
    assert torch.allclose(g.alpha, g2.alpha)
    assert torch.allclose(g.log_scaling, g2.log_scaling)
    assert torch.allclose(g.sh_feature, g2.sh_feature)





if __name__ == '__main__':
  test_read_write()


