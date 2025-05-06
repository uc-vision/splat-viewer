import argparse
from pathlib import Path
import taichi as ti
import torch

from splat_viewer.gaussians.workspace import load_workspace
from taichi_splatting.taichi_queue import TaichiQueue

import matplotlib.pyplot as plt
import open3d as o3d
import cv2
import numpy as np
import os
import glob
from time import perf_counter

from splat_viewer.scripts.noise_adder import NoiseAdder
from splat_viewer.scripts.model_icp import ModelICP

def clear_test_images():
    """
    Deletes all files in the specified images folder.
    Safely handles cases where folder doesn't exist.
    """
    BASE_DIR = Path(__file__).parent.parent

    # Define paths for outputs
    SAVE_DIR = BASE_DIR / "test_images" 
    SAVE_DIR.mkdir(parents=True, exist_ok=True)  # Creates dir if it doesn't exist

    folder_path = SAVE_DIR
    
    # Get all image files (common extensions)
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(folder_path, ext)))
    
    # Delete each file
    for file_path in image_files:
        try:
            os.remove(file_path)
            print(f"Deleted: {file_path}")
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def compare_pointcloud_scales(source, target):
    """
    Compare scales of two point clouds by analyzing:
    1. Bounding box dimensions
    2. Average distance from centroid
    3. Standard deviation of distances
    """
    # Convert to numpy arrays if not already
    source_pts = np.asarray(source.points)
    target_pts = np.asarray(target.points)
    
    # 1. Compare bounding box dimensions
    source_box = source.get_axis_aligned_bounding_box()
    target_box = target.get_axis_aligned_bounding_box()
    
    print("\nBounding Box Dimensions:")
    print(f"Source (X,Y,Z): {source_box.get_extent()}")
    print(f"Target (X,Y,Z): {target_box.get_extent()}")
    print(f"Ratio (Source/Target): {source_box.get_extent() / target_box.get_extent()}")
    
    # 2. Compare average distances from centroid
    def get_scale_metrics(points):
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        return {
            'avg_distance': np.mean(distances),
            'std_distance': np.std(distances),
            'max_distance': np.max(distances)
        }
    
    source_metrics = get_scale_metrics(source_pts)
    target_metrics = get_scale_metrics(target_pts)
    
    print("\nDistance from Centroid Metrics:")
    print(f"Source - Avg: {source_metrics['avg_distance']:.4f}, "
          f"Std: {source_metrics['std_distance']:.4f}, "
          f"Max: {source_metrics['max_distance']:.4f}")
    print(f"Target - Avg: {target_metrics['avg_distance']:.4f}, "
          f"Std: {target_metrics['std_distance']:.4f}, "
          f"Max: {target_metrics['max_distance']:.4f}")
    print(f"Scale Ratio (Source/Target): {source_metrics['avg_distance'] / target_metrics['avg_distance']:.4f}")
    
    # 3. Visual comparison
    source.paint_uniform_color([1, 0, 0])  # Red
    target.paint_uniform_color([0, 1, 0])  # Green
    o3d.visualization.draw_geometries([source, target], 
                                     window_name="Scale Comparison (Red=Source, Green=Target)")
    
def show_alignment_from_view(source, target, w2c=None, point_size=1.0, show_image=True):
    """
    Visualize alignment of source (red) and target (green) point clouds,
    optionally from a specified camera pose, with adjustable point size.
    
    Args:
        source (o3d.geometry.PointCloud): Source point cloud (will be shown in red)
        target (o3d.geometry.PointCloud): Target point cloud (will be shown in green)
        w2c (np.ndarray): Optional 4x4 world-to-camera transformation matrix
        point_size (float): Point size for visualization (default: 1.0, finest)
    """
    # Colorize point clouds
    source.paint_uniform_color([1, 0, 0])  # Red = source
    target.paint_uniform_color([0, 1, 0])  # Green = target

    # Create visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    
    # Add geometries
    vis.add_geometry(source)
    vis.add_geometry(target)

    # Set point size (finest by default)
    render_opt = vis.get_render_option()
    render_opt.point_size = point_size

    # Set camera view if w2c is provided
    if w2c is not None:
        view_ctl = vis.get_view_control()
        params = view_ctl.convert_to_pinhole_camera_parameters()
        params.extrinsic = w2c
        view_ctl.convert_from_pinhole_camera_parameters(params)

    # Capture image
    vis.poll_events()
    vis.update_renderer()
    image = vis.capture_screen_float_buffer(do_render=True)
    image = np.asarray(image) * 255  # Convert to 0-255 range
    image = image.astype(np.uint8)

    if show_image:
       vis.run()

    vis.destroy_window()

    return image

def write_rmse_on_image(image, rmse):
  image = np.asarray(image) * 255
  image = image.astype(np.uint8)
  image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  
  # Add text annotation if specified
  if rmse is not None:
      font = cv2.FONT_HERSHEY_SIMPLEX
      position = (30, 50)  # (x,y) top-left corner
      font_scale = 1.5
      color = (255, 255, 255)  # White text
      thickness = 3
      
      cv2.putText(image, str(rmse), position, font, 
                font_scale, color, thickness, cv2.LINE_AA)
  
  # Convert back to RGB for saving/display
  image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

  return image

def parse_arguments():
  parser = argparse.ArgumentParser(description="Add a 'foreground' annotation to a .ply gaussian splatting file")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  parser.add_argument("--far", default=1.0, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.01, type=float, help="Min depth to determine the visible ROI")
  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')
  parser.add_argument("--show_result", action="store_true", help='Display resulting ICP alignment')
  parser.add_argument("--num_samples", default=5, type=int, help='Number of samples (different camera FOV) to test within scan')

  args = parser.parse_args()

  return args

def save_image(aligned_image, cam_num):
  BASE_DIR = Path(__file__).parent.parent 

  SAVE_DIR = BASE_DIR / "test_images"
  SAVE_DIR.mkdir(parents=True, exist_ok=True) 

  file_path = SAVE_DIR / f"alignment_{cam_num}.png"

  plt.imsave(file_path, aligned_image)

def main():
  torch.cuda.empty_cache()

  TaichiQueue.init(ti.gpu, offline_cache=True, device_memory_GB=0.1)

  clear_test_images()

  args = parse_arguments()

  workspace = load_workspace(args.model_path)

  model_icp = ModelICP(workspace=workspace, args=args)

  model_pcd = model_icp.model_pcd

  noise_adder = NoiseAdder()

  with torch.inference_mode():

    num_cameras = model_icp.num_cameras

    # Calculate step size to for #num_samples steps
    step_size = max(1, num_cameras // args.num_samples) if num_cameras > args.num_samples else 1

    for i in range(0, num_cameras, step_size):

      print(f"=== CAMERA {i} ===")
      
      camera = model_icp.get_camera(i)

      w2c = model_icp.get_w2c(camera)
      c2w = model_icp.get_c2w(camera)

      w2c_noisy = noise_adder.add_camera_pose_noise(w2c, pos_noise_scale=0.1, rot_noise_deg=5.0)

      # create a rendering
      rendering = model_icp.render(camera)

      colour_image = rendering.image.detach().cpu().numpy()
      depth_image = rendering.depth.detach().cpu().numpy()

      # Depth is already float32, Colour needs float32 -> uint8 conversion
      colour_image = (colour_image * 255).astype(np.uint8) 

      query_pcd = model_icp.create_query_pcd(colour_image=colour_image, depth_image= depth_image)

      query_pcd = noise_adder.add_outliers(query_pcd, noise_std=0.5, outlier_ratio=0.01)
      query_pcd = noise_adder.add_gaussian_noise(query_pcd, std=0.0001)
      query_pcd = noise_adder.add_density_variation(query_pcd, keep_ratio=0.8)
      query_pcd = noise_adder.add_quantization_noise(query_pcd, step_size=0.001)

      # start = perf_counter()
      reg_gicp = model_icp.apply_GICP(query_pcd, w2c_noisy)
      # mid = perf_counter()

      small_GICP_T = model_icp.apply_smallGICP(query_pcd, w2c=w2c_noisy)

      # end =perf_counter()

      # print(f"open3d: {mid - start:.6f} seconds")
      # print(f"smallGICP: {end - mid:.6f} seconds")

      model_icp.verify_result(small_GICP_T, reg_gicp.transformation)

      # query_pcd.transform(reg_gicp.transformation)
      query_pcd.transform(small_GICP_T)

      aligned_image = show_alignment_from_view(query_pcd, model_pcd, c2w, show_image=args.show_result)

      save_image(aligned_image=aligned_image, cam_num=i)

if __name__ == "__main__":
  main()  
