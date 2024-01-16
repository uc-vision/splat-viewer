
import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import math

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.gaussians.workspace import load_workspace

from taichi_splatting import RasterConfig, map_to_tiles, pad_to_tile, perspective, tile_mapper

import taichi as ti
from splat_viewer.renderer.taichi_splatting import to_camera_params


def overlap_stats(overlap_counts, tile_counts,  max_hist = 8):
    large_thresh = (overlap_counts.std() * 3).item()
    very_large = (overlap_counts > large_thresh).sum()

    hist = torch.histc(overlap_counts.clamp(0, max_hist + 1), bins=max_hist + 1, min=0, max=max_hist + 1)

    n = overlap_counts.shape[0]
    print(f"Overlap mean: {overlap_counts.mean():.2f} very large (> {large_thresh:.2f}) %: {100.0 * very_large / n:.2f}")
    print(f"Overlap histogram %: {100 * hist[:max_hist] / n} > 10: {100 * hist[max_hist] / n}")


    print(f"Mean tile count: {tile_counts.mean():.2f}, max tile count: {tile_counts.max()}")


def main():

  torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)

  parser = argparse.ArgumentParser()

  parser.add_argument("model_path", type=Path, help="workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")
  parser.add_argument("--device", type=str, default="cuda:0", help="torch device to use")
  parser.add_argument("--model", type=str, default=None, help="model to load")

  parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterizer")
  parser.add_argument("--image_size", type=int, default=None, help="resize longest edge of camera image sizes")
  parser.add_argument("--no_tight_culling", action="store_true", help="disable tight culling")
  parser.add_argument("--debug", action="store_true", help="enable taichi debug mode")

  args = parser.parse_args()

  ti.init(arch=ti.cuda, offline_cache=True, log_level=ti.INFO,
          debug=args.debug, device_memory_GB=0.1)


  workspace = load_workspace(args.model_path)
  if args.model is None:
      args.model = workspace.latest_iteration()
    
  gaussians:Gaussians = workspace.load_model(args.model).to(args.device)
  gaussians = gaussians.sorted()


  print(f"Using {args.model_path} with {gaussians.batch_size[0]} points")

  image_sizes = set([tuple(camera.image_size) for camera in workspace.cameras])
  print(f"Cameras: {len(workspace.cameras)}, Image sizes: {image_sizes}")
    
  config = RasterConfig(
    tile_size=args.tile_size, 
    tight_culling=not args.no_tight_culling)

  packed = gaussians.packed()

  overlaps = []
  tile_counts = []

  for camera in tqdm(workspace.cameras):
    camera_params = to_camera_params(camera, device=args.device)

    mask = perspective.frustum_culling(packed, camera_params, margin_pixels=50)
    gaussians2d, depth = perspective.project_to_image(packed[mask], camera_params)
      
    overlap_to_point, tile_ranges = map_to_tiles(
      gaussians2d, depth, camera_params.image_size, config)
    
    image_size = pad_to_tile(camera.image_size, config.tile_size)

    mapper = tile_mapper.tile_mapper(config)
    overlap_offsets, total_overlap = mapper.generate_tile_overlaps(
      gaussians2d, image_size)
  
    cum_overlap_counts = torch.cat([overlap_offsets.cpu(), torch.tensor([total_overlap])])

    
    overlaps.append((cum_overlap_counts[1:] - cum_overlap_counts[:-1]).float())
    tile_counts.append((tile_ranges[:, 1] - tile_ranges[:, 0]).float())


  overlap_stats(torch.cat(overlaps), torch.cat(tile_counts))
  max_counts = [torch.max(overlap) for overlap in overlaps]
  print(f"Max overlap (mean): {sum(max_counts) / len(max_counts):.2f}")
     

if __name__ == "__main__":
  main()  

