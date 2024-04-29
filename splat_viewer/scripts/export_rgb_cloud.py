import numpy as np
import torch
from splat_viewer.gaussians  import read_gaussians
import argparse
from pathlib import Path
import open3d as o3d

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.gaussians.workspace import load_workspace
import open3d.core as o3c



def torch_to_o3d(tensor:torch.Tensor) -> o3d.core.Tensor:
  return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

def o3d_to_torch(tensor:o3c.Tensor) -> torch.Tensor:
  return torch.from_dlpack(o3d.core.Tensor.to_dlpack(tensor))



def sample_points(gaussians:Gaussians, n:int):
  m = gaussians.batch_size[0]
  basis = gaussians.get_rotation_matrix()   # (M, 3, 3)

  samples = (torch.randn((m, n, 3), device=gaussians.device) 
             * gaussians.scale()[:, None, :]) # (M, N, 3)
  
  samples = torch.einsum('mij,mnj->mni', basis, samples) # (M, N, 3)
  return samples + gaussians.position[:, None, :] # (M, N, 3)




def to_rgb(gaussians:Gaussians, densify=1) -> o3d.t.geometry.PointCloud:
  colors = gaussians.get_colors()

  if densify > 1:
    positions = sample_points(gaussians, densify).reshape(-1, 3)
    colors = colors.repeat_interleave(densify, dim=0)

  else:
    positions = gaussians.position


  positions, colors = [torch_to_o3d(t) for t in (positions, colors)]

  cloud = o3d.t.geometry.PointCloud(positions.device)
  
  cloud.point['positions'] = positions
  cloud.point['colors'] = colors

  return cloud


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=Path)
  parser.add_argument('--write', type=Path)
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--densify', default=1, type=int)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--sample', default=None, type=float)
  args = parser.parse_args()

  if args.write is None and not args.show:
    raise ValueError("Must specify --output or --show")

  input:Path = args.input

  if input.is_dir():
    workspace = load_workspace(input)
    gaussians:Gaussians = workspace.load_model()
  else:
    gaussians = read_gaussians(args.input)

  gaussians = gaussians.to(device=args.device)

  
  print("Loaded:", gaussians)

  if gaussians.foreground is not None:
    gaussians = gaussians.crop_foreground()

  pcd:o3d.t.geometry.PointCloud = to_rgb(gaussians, densify=args.densify)
   
  print(pcd)

  if args.sample is not None:
    pcd = pcd.voxel_down_sample(args.sample)
    print(f"After sampling to {args.sample}", pcd)



  if args.show:
    o3d.visualization.draw([pcd])
  
  if args.write:
    o3d.t.io.write_point_cloud(str(args.write), pcd)

    print(f"Wrote {pcd} to {args.write}")