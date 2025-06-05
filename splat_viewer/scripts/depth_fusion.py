import numpy as np
import torch
from splat_viewer.gaussians  import read_gaussians
import argparse
from pathlib import Path
import open3d as o3d
from tqdm import tqdm
from dataclasses import dataclass

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.gaussians.workspace import load_workspace
from splat_viewer.camera import FOVCamera

from splat_viewer.renderer.taichi_splatting import GaussianRenderer, Rendering
import open3d.core as o3c

from taichi_splatting import TaichiQueue
import taichi as ti


@dataclass
class TSDFConfig:
    voxel_size: float = 0.0005
    block_resolution: int = 8
    depth_max: float = 1.5
    device: str = "CUDA:0"


def torch_to_o3d(tensor:torch.Tensor) -> o3d.core.Tensor:
  return o3d.core.Tensor.from_dlpack(torch.utils.dlpack.to_dlpack(tensor))

def o3d_to_torch(tensor:o3c.Tensor) -> torch.Tensor:
  return torch.from_dlpack(o3d.core.Tensor.to_dlpack(tensor))


def to_o3d_rgbd(rendering:Rendering) -> o3d.t.geometry.RGBDImage:
  return o3d.t.geometry.RGBDImage.create_from_color_and_depth(
    torch_to_o3d(rendering.image),
    torch_to_o3d(rendering.depth)
  )


def create_voxel_block_grid(config: TSDFConfig) -> o3d.t.geometry.VoxelBlockGrid:
    """Create and initialize a voxel block grid for TSDF integration"""
    device = o3d.core.Device(config.device)
    
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=('tsdf', 'weight'),
        attr_dtypes=(o3c.float32, o3c.float32),
        attr_channels=((1), (1)),
        voxel_size=config.voxel_size,
        block_resolution=config.block_resolution,
        block_count=10000,
        device=device
    )
    
    return vbg


def integrate_tsdf(vbg: o3d.t.geometry.VoxelBlockGrid, 
                   rendering: Rendering, 
                   camera: FOVCamera,
                   config: TSDFConfig) -> None:
    """Integrate a single depth/color image into the TSDF volume"""
    
    # Convert rendering to Open3D format
    depth_tensor = torch_to_o3d(rendering.depth.squeeze())
    depth = o3d.t.geometry.Image(depth_tensor)

    K = camera.intrinsic
    intrinsic = o3d.core.Tensor(K, dtype=o3d.core.float64)
    
    # Create extrinsic matrix (world-to-camera transform)
    extrinsic = o3d.core.Tensor(camera.camera_t_world, dtype=o3d.core.float64)
    
    # Compute unique block coordinates in current viewing frustum
    frustum_block_coords = vbg.compute_unique_block_coordinates(
        depth, intrinsic, extrinsic, 1.0, config.depth_max
    )
    
    # Integrate depth only (no color to save memory)
    vbg.integrate(frustum_block_coords, depth, intrinsic, extrinsic, 1.0, config.depth_max)


def main():

  parser = argparse.ArgumentParser()
  parser.add_argument('input', type=Path)
  parser.add_argument('--write', type=Path)
  parser.add_argument('--show', action='store_true')

  parser.add_argument('--densify', default=1, type=int)
  parser.add_argument('--device', default='cuda:0')
  parser.add_argument('--sample', default=None, type=float)
  
  # TSDF parameters
  parser.add_argument('--voxel_size', default=0.001, type=float)
  parser.add_argument('--depth_max', default=1.0, type=float)
  parser.add_argument('--image_scale', default=1.0, type=float, help='Scale factor for images to reduce memory usage')
  
  args = parser.parse_args()

  if args.write is None and not args.show:
    raise ValueError("Must specify --output and/or --show")

  input:Path = args.input

  TaichiQueue.init(ti.gpu, offline_cache=True, debug=False, device_memory_GB=0.1)

  torch.set_grad_enabled(False)

  assert input.is_dir()
  workspace = load_workspace(input)
  gaussians:Gaussians = workspace.load_model()

  gaussians = gaussians.to(device=args.device)

  renderer = GaussianRenderer()
  inputs = renderer.pack_inputs(gaussians)

  # Create TSDF configuration
  config = TSDFConfig(
      voxel_size=args.voxel_size,
      depth_max=args.depth_max,
      device="CUDA:0" if args.device.startswith('cuda') else "CPU:0"
  )
  
  # Initialize voxel block grid
  vbg = create_voxel_block_grid(config)
  
  print(f"Starting TSDF integration with {len(workspace.cameras)} views...")
  
  for camera in tqdm(workspace.cameras, desc="Integrating views"):
    # Scale down camera to reduce memory usage
    scaled_camera = camera.scale_size(args.image_scale) if args.image_scale != 1.0 else camera
    rendering = renderer.render(inputs, scaled_camera)
    integrate_tsdf(vbg, rendering, scaled_camera, config)

    torch.cuda.empty_cache()
  
  print("TSDF integration complete, extracting geometry...")
  
  # Extract point cloud
  pcd = vbg.extract_point_cloud()
  
  if args.write:
      o3d.io.write_point_cloud(str(args.write), pcd)
      print(f"Saved point cloud to {args.write}")
  
  if args.show:
      o3d.visualization.draw([pcd])


if __name__ == "__main__":
  main()