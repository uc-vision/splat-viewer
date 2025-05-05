import argparse
from pathlib import Path
from beartype.typing import List

import taichi as ti
import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.gaussians.workspace import load_workspace

from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from taichi_splatting.taichi_queue import TaichiQueue

import matplotlib.pyplot as plt
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R

import numpy as np

import os
import glob

from splat_viewer.scripts.pcd_noise import PointCloudNoise

def clear_test_images(folder_path="splat-viewer/splat_viewer/test_images/"):
    """
    Deletes all files in the specified images folder.
    Safely handles cases where folder doesn't exist.
    """
    # Create the folder if it doesn't exist
    os.makedirs(folder_path, exist_ok=True)
    
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

def crop_model(model, cameras:List[FOVCamera], args):
  num_visible, min_distance = visibility(cameras, model.position, near = args.near)

  min_views = max(1, len(cameras) * args.min_percent / 100)
  is_visible = (num_visible > min_views)

  is_near = (min_distance < args.far)
  n_near = is_near.sum(dtype=torch.int32)

  print(f"Cropped model from {model.batch_size[0]} to {is_visible.sum().item()} visible points, {n_near} near (at least {min_views} views)")
  model = model.replace(foreground=is_near.reshape(-1, 1))

  model = model[is_visible]

  model = model.crop_foreground()

  return model

def load_model(workspace):
  model_name = workspace.latest_iteration()
  model_file = workspace.model_filename(model_name)
  model = read_gaussians(model_file)
  return model

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

def create_model_pcd(model):
  # Create open3d pointcloud of the model
  xyz_np = model.position.cpu().numpy().astype(np.float64)  # Shape [N, 3]

  model_pcd = o3d.geometry.PointCloud()
  model_pcd.points = o3d.utility.Vector3dVector(xyz_np)  

  colors_np = model.get_colors().cpu().numpy().astype(np.float64)
  model_pcd.colors = o3d.utility.Vector3dVector(colors_np)
  
  return model_pcd

def define_intrinsics(image_colour, camera):
  return o3d.camera.PinholeCameraIntrinsic(
      width=image_colour.shape[1],
      height=image_colour.shape[0],
      fx=camera.focal_length[0],  
      fy=camera.focal_length[1],  
      cx=camera.principal_point[0],  
      cy=camera.principal_point[1]   
  )

def scale_pcd(pcd, scale_factor):
  center = pcd.get_center().reshape(3, 1)
  return pcd.scale(scale_factor, [0,0,0])
  

def apply_GICP(source, target, initial_guess=np.eye(4)):

  # Step 1: Auto-tune parameters
  bbox = target.get_axis_aligned_bounding_box().get_extent()
  threshold = 0.15 * np.max(bbox)  # 15% of largest cloud dimension

  distances = source.compute_nearest_neighbor_distance()
  radius = 2.5 * np.mean(distances)

  # Step 2: Estimate normals
  source.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
      radius=radius, max_nn=40))
  target.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
      radius=radius, max_nn=40))

  # Step 3: Run GICP
  criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
      max_iteration=150,
      relative_fitness=1e-6,
      relative_rmse=1e-6
  )

  reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
      source, target, threshold, initial_guess,
      o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
      criteria
  )

  # Results
  # print("Generalized ICP Transformation Matrix:\n", reg_gicp.transformation)
  # print("Fitness Score (inlier ratio):", reg_gicp.fitness)  # Range [0, 1], higher is better
  print("RMSE:", reg_gicp.inlier_rmse)  # Root Mean Square Error of inliers

  return reg_gicp

def create_w2c_matrix(camera):
  # 4x4 world-to-camera matrix
  w2c = np.eye(4)
  w2c[:3, :3] = camera.rotation
  w2c[:3, 3] = camera.position

  return w2c

def create_rgbd_from_depth_and_rgb(rgb, depth):
  return o3d.geometry.RGBDImage.create_from_color_and_depth(
      o3d.geometry.Image(rgb),
      o3d.geometry.Image(depth),
      convert_rgb_to_intensity=False
  )

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

def generate_pose_noise(pos_noise_scale=0.15, rot_noise_deg=7.5):
    """
    Generates random pose noise components.
    
    Args:
        pos_noise_scale (float): Max translation noise (meters)
        rot_noise_deg (float): Max rotation noise (degrees)
        
    Returns:
        tuple: (position_noise, rotation_matrix_noise)
               position_noise: np.ndarray (3,) translation noise vector
               rotation_matrix_noise: np.ndarray (3,3) rotation noise matrix
    """
    # Generate translation noise
    pos_noise = np.random.uniform(-pos_noise_scale, pos_noise_scale, 3)
    
    # Generate rotation noise
    rot_noise_rad = np.radians(rot_noise_deg)
    angle_noise = np.random.uniform(-rot_noise_rad, rot_noise_rad, 3)
    R_noise = o3d.geometry.get_rotation_matrix_from_xyz(angle_noise)
    
    return pos_noise, R_noise

def apply_pose_noise(w2c, pos_noise, R_noise):
    """
    Applies noise components to a world-to-camera (w2c) matrix.
    
    Args:
        w2c (np.ndarray): 4x4 world-to-camera matrix
        pos_noise (np.ndarray): (3,) translation noise vector
        R_noise (np.ndarray): (3,3) rotation noise matrix
        
    Returns:
        np.ndarray: Noisy 4x4 w2c matrix
    """
    noisy_w2c = w2c.copy()
    noisy_w2c[:3, 3] += pos_noise  # Apply translation noise
    noisy_w2c[:3, :3] = noisy_w2c[:3, :3] @ R_noise  # Apply rotation noise
    return noisy_w2c

def print_pose_noise(pos_noise, rot_matrix):
    """
    Prints pose noise in human-friendly units:
    - Translation in centimeters
    - Rotation as axis-angle in degrees
    """
    # Convert translation to cm (m to cm)
    pos_cm = pos_noise * 100
    
    # Convert rotation matrix to axis-angle (degrees)
    rot = R.from_matrix(rot_matrix)
    axis_angle = rot.as_rotvec()
    angle_deg = np.degrees(np.linalg.norm(axis_angle))
    if angle_deg > 1e-6:  # Avoid division by zero
        axis = axis_angle / np.linalg.norm(axis_angle)
    else:
        axis = np.zeros(3)
    
    # Format output
    print("\n=== Pose Noise ===")
    print(f"Translation (cm):")
    print(f"  X: {pos_cm[0]:+.2f} cm")
    print(f"  Y: {pos_cm[1]:+.2f} cm")
    print(f"  Z: {pos_cm[2]:+.2f} cm")
    
    print("\nRotation:")
    print(f"  Angle: {angle_deg:.2f}°")
    print(f"  Axis: [{axis[0]:+.2f}, {axis[1]:+.2f}, {axis[2]:+.2f}]")

def add_noise_to_w2c(w2c, pos_noise_scale=0.1, rot_noise_deg=5.0):
    """Original function now using the two new functions"""
    pos_noise, R_noise = generate_pose_noise(pos_noise_scale, rot_noise_deg)

    print_pose_noise(pos_noise, R_noise)
    return apply_pose_noise(w2c, pos_noise, R_noise)

def main():
  torch.cuda.empty_cache()

  TaichiQueue.init(ti.gpu, offline_cache=True, device_memory_GB=0.1)

  clear_test_images()

  args = parse_arguments()

  workspace = load_workspace(args.model_path)

  with torch.inference_mode():
    workspace = load_workspace(args.model_path)
    model = load_model(workspace)
    model = model.to(args.device)

    model = crop_model(model, workspace.cameras, args)

    model_pcd = create_model_pcd(model)
    
    renderer = GaussianRenderer()
    noise_adder = PointCloudNoise()

    num_cameras = len(workspace.cameras)

    # Calculate step size to for #num_samples steps
    step_size = max(1, num_cameras // args.num_samples) if num_cameras > args.num_samples else 1

    for i in range(0, num_cameras, step_size):

      print(f"=== CAMERA {i} ===")
      
      camera = workspace.cameras[i]

      w2c = camera.world_t_camera
      c2w = camera.camera_t_world

      w2c_noisy = add_noise_to_w2c(w2c, pos_noise_scale=0.10, rot_noise_deg=5.0)

      # create a rendering
      rendering = renderer.render(renderer.pack_inputs(model), camera)

      image_colour = rendering.image.detach().cpu().numpy()
      image_depth = rendering.depth.detach().cpu().numpy()

      # Depth is already float32, Colour needs float32 -> uint8 conversion
      image_colour = (image_colour * 255).astype(np.uint8) 

      rgbd = create_rgbd_from_depth_and_rgb(image_colour, image_depth)

      # Create open3d pointcloud from rendered rgbd image 
      query_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
          rgbd,
          define_intrinsics(image_colour, camera)
      )

      # scale to match model
      query_pcd = scale_pcd(query_pcd, 1000)

      query_pcd = noise_adder.add_outliers(query_pcd, noise_std=0.1)
      query_pcd = noise_adder.add_gaussian_noise(query_pcd, std=0.001)
      query_pcd = noise_adder.add_density_variation(query_pcd, keep_ratio=0.75)
      query_pcd = noise_adder.add_quantization_noise(query_pcd, step_size=0.003)

      reg_gicp = apply_GICP(query_pcd, model_pcd, w2c_noisy)

      query_pcd.transform(reg_gicp.transformation)

      aligned_image = show_alignment_from_view(query_pcd, model_pcd, c2w, show_image=args.show_result)

      plt.imsave(f"splat-viewer/splat_viewer/test_images/alignment_{i}.png", aligned_image)

if __name__ == "__main__":
  main()  






