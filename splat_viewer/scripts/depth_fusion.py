import numpy as np
import torch
from splat_viewer.gaussians  import read_gaussians
import argparse
from pathlib import Path
import open3d as o3d

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.gaussians.workspace import load_workspace
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
import open3d.core as o3c


def torch_to_o3d(tensor:torch.Tensor) -> o3d.core.Tensor:
  return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

def o3d_to_torch(tensor:o3c.Tensor) -> torch.Tensor:
  return torch.from_dlpack(o3d.core.Tensor.to_dlpack(tensor))


def to_o3d_rgbd(rendering:Rendering) -> o3d.geometry.RGBDImage:
  return o3d.t.geometry.RGBDImage.create_from_color_and_depth(
    torch_to_o3d(rendering.image),
    torch_to_o3d(rendering.depth)
  )



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

  renderer = GaussianRenderer()

  for camera in workspace.cameras:
    rendering = renderer.render(gaussians, camera)