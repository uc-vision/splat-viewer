import argparse
from pathlib import Path
import taichi as ti
import torch
import copy

from beartype import beartype

from splat_viewer.gaussians.workspace import load_workspace
from taichi_splatting.taichi_queue import TaichiQueue
from splat_viewer.camera.fov import FOVCamera

import matplotlib.pyplot as plt
import open3d as o3d

import cv2
import numpy as np
import os
import glob
from time import perf_counter
from argparse import Namespace

from splat_viewer.scripts.noise_adder import NoiseAdder
from splat_viewer.scripts.differentiable_renderer import DifferentiableRenderer, extract_rt

import numpy as np
from typing import Dict, Union, Literal, Optional, Tuple
from numpy.typing import NDArray

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

def parse_arguments() -> Namespace:
  parser = argparse.ArgumentParser(description="Add a 'foreground' annotation to a .ply gaussian splatting file")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  parser.add_argument("--far", default=1.0, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.01, type=float, help="Min depth to determine the visible ROI")
  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')
  parser.add_argument("--show_result", action="store_true", help='Display resulting ICP alignment')
  parser.add_argument("--num_samples", default=5, type=int, help='Number of samples (different camera FOV) to test within scan')
  parser.add_argument("--icp", default='open3d', help='icp algorithm. Choices: "smallgicp", "open3d"')

  args = parser.parse_args()

  return args

def save_image(
  aligned_image:np.ndarray,
  cam_num:int
):

  BASE_DIR = Path(__file__).parent.parent 

  SAVE_DIR = BASE_DIR / "test_images"
  SAVE_DIR.mkdir(parents=True, exist_ok=True) 

  file_path = SAVE_DIR / f"alignment_{cam_num}.png"

  plt.imsave(file_path, aligned_image)

def compare_arrays(
    arr1: NDArray[np.float64],
    arr2: NDArray[np.float64],
    method: Literal["absolute", "relative", "mse", "rmse", "mae", "all"] = "all",
    epsilon: float = 1e-10  # Small value to avoid division by zero in relative diff
) -> Union[float, Dict[str, float]]:
    """
    Compare the difference between two NumPy arrays of float64 type.
    
    Parameters:
    -----------
    arr1 : NDArray[np.float64]
        First array for comparison
    arr2 : NDArray[np.float64]
        Second array for comparison
    method : str, optional
        Method to use for comparison:
        - "absolute": Maximum absolute difference
        - "relative": Maximum relative difference
        - "mse": Mean squared error
        - "rmse": Root mean squared error
        - "mae": Mean absolute error
        - "all": Return all metrics (default)
    epsilon : float, optional
        Small value to avoid division by zero in relative difference calculations
        
    Returns:
    --------
    Union[float, Dict[str, float]]
        If method is specific, returns the calculated metric as a float.
        If method is "all", returns a dictionary with all metrics.
        
    Raises:
    -------
    ValueError
        If arrays have different shapes or if an invalid method is specified
    """
    # Check if arrays have the same shape
    if arr1.shape != arr2.shape:
        raise ValueError(f"Arrays have different shapes: {arr1.shape} vs {arr2.shape}")
    
    # Ensure arrays are of type float64
    arr1 = np.asarray(arr1, dtype=np.float64)
    arr2 = np.asarray(arr2, dtype=np.float64)
    
    # Calculate differences
    diff = arr1 - arr2
    abs_diff = np.abs(diff)
    
    # Calculate metrics
    results = {}
    
    if method in ["absolute", "all"]:
        results["max_absolute_diff"] = np.max(abs_diff)
        results["min_absolute_diff"] = np.min(abs_diff)
        
    if method in ["relative", "all"]:
        # Avoid division by zero by adding a small epsilon to denominator
        denominator = np.maximum(np.abs(arr2), epsilon)
        rel_diff = abs_diff / denominator
        results["max_relative_diff"] = np.max(rel_diff)
        results["mean_relative_diff"] = np.mean(rel_diff)
        
    if method in ["mse", "all"]:
        results["mse"] = np.mean(np.square(diff))
        
    if method in ["rmse", "all"]:
        results["rmse"] = np.sqrt(np.mean(np.square(diff)))
        
    if method in ["mae", "all"]:
        results["mae"] = np.mean(abs_diff)
    
    # Return appropriate result based on method
    if method == "all":
        print( results)
    elif method == "absolute":
        print( results["max_absolute_diff"])
    elif method == "relative":
        print( results["max_relative_diff"] )
    elif method == "mse":
        print( results["mse"])
    elif method == "rmse":
        print( results["rmse"])
    elif method == "mae":
        print( results["mae"])
    else:
        raise ValueError(f"Invalid method: {method}. Choose from 'absolute', 'relative', 'mse', 'rmse', 'mae', or 'all'")



def main():
  torch.cuda.empty_cache()

  TaichiQueue.init(ti.gpu, offline_cache=True, device_memory_GB=0.1)

  clear_test_images()

  args = parse_arguments()

  workspace = load_workspace(args.model_path)

  diff_renderer = DifferentiableRenderer(workspace=workspace, args=args)

  noise_adder = NoiseAdder()

  with torch.inference_mode():
      
      camera = diff_renderer.get_camera(20)

      w2c = diff_renderer.get_w2c(camera)

      rendering = diff_renderer.render(camera=camera)

      w2c_noisy = noise_adder.add_camera_pose_noise(w2c, pos_noise_scale=0.05, rot_noise_deg=0.0)

      camera_noisy = copy.deepcopy(camera)

      R_extracted, T_extracted = extract_rt(w2c_noisy)

      camera_noisy.position = T_extracted
      camera_noisy.rotation = R_extracted

      refined_pose = diff_renderer.optimize_camera_pose(target_image=rendering.image, camera=camera_noisy)

      pose_ndarray: NDArray[np.float64] = refined_pose.detach().cpu().numpy()

      compare_arrays(pose_ndarray, w2c)

      # refined_camera = copy.deepcopy(camera)

      # refined_camera.camera_t_world = refined_pose

      # refined_image = diff_renderer.render(camera=refined_camera)

      # save_image(aligned_image=refined_image, cam_num=20)

if __name__ == "__main__":
  main()  
