import argparse
from pathlib import Path
import numpy as np
import math

from splat_viewer.camera.visibility import visibility
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.gaussians.workspace import load_workspace, Gaussians
from splat_viewer.camera import FOVCamera

import torch


def render_tiled(camera:FOVCamera, gaussians:Gaussians, renderer:GaussianRenderer, tile_size:int=1024):
  nw, nh = [int(math.ceil(x / tile_size)) 
        for x in camera.image_size]
  
  full_image = np.zeros((nh * tile_size, nw * tile_size, 3), dtype=np.uint8)
  
  for x in range(0, nw):
    for y in range(0, nh):
      tile_camera = camera.crop(np.array([x * tile_size, y * tile_size]), 
                            np.array([tile_size, tile_size]))
      
      image = renderer.render(tile_camera, gaussians)
      tile = full_image[y * tile_size:(y + 1) * tile_size, 
                        x * tile_size:(x + 1) * tile_size, :] 
      
      print(tile.shape, image.shape, x, y)
      tile[:] = image
      
  return full_image[:camera.image_size[1], :camera.image_size[0]]

def fit_plane(camera_centers):
  
  centroid = np.mean(camera_centers, axis=0)
    # fit plane through centers
  _, _, Vt = np.linalg.svd(camera_centers - centroid)
  normal = Vt[-1]
  normal /= np.linalg.norm(normal)

  return normal, centroid

def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("workspace", type=Path, help="workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")
  parser.add_argument("--output", type=Path, default="output.png", help="output image file")

  parser.add_argument("--far", type=float, default=1.0, help="far clipping plane for visibility")
  return parser.parse_args()

def main():
  np.set_printoptions(precision=3, suppress=True)
  torch.set_printoptions(precision=3, sci_mode=False)

  args = parse_args()

  workspace = load_workspace(args.workspace)
  renderer =  GaussianRenderer()
  
  gaussians = workspace.load_model()

  if gaussians.foreground is None:
    fg_count, _ = visibility(workspace.cameras, gaussians.position, far=args.far)
    gaussians = gaussians.replace(foreground=fg_count > len(workspace.cameras) / 10)

  # compute camera plane
  normal, centroid = fit_plane([camera.position for camera in workspace.cameras])
  print(normal, centroid)

  gaussians = gaussians.crop_foreground()

  # compute basis on the plane along with camera "up" vector
  up = np.mean([camera.up for camera in workspace.cameras], axis=0)
  up = up / np.linalg.norm(up)

  basis = torch.from_numpy(np.array([np.cross(normal, up), up, normal])).to(torch.float32)
  on_basis = basis @ gaussians.position.T 

  lower = on_basis.min(dim=1).values
  upper = on_basis.max(dim=1).values

  centroid2 = (upper + lower) / 2
  print(centroid, basis.T @ centroid2)

  # project foregorund points onto the plane and compute the bounds
  
