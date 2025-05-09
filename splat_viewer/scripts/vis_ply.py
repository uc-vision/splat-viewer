from pathlib import Path
import taichi as ti
import torch

from taichi_splatting.taichi_queue import TaichiQueue

import matplotlib.pyplot as plt
import open3d as o3d



def visualize_ply(file_path):
    # Read the PLY file
    pcd = o3d.io.read_point_cloud(file_path)  # or read_triangle_mesh for mesh data
    
    # Visualize the point cloud/mesh
    o3d.visualization.draw_geometries([pcd],
                                      window_name="PLY Viewer",
                                      width=800,
                                      height=600)

if __name__ == "__main__":
    
  BASE_DIR = Path(__file__).parent.parent 

  LOAD_DIR = BASE_DIR / "test_pcds"
  LOAD_DIR.mkdir(parents=True, exist_ok=True) 

  file_name = "pointcloud_20250506_135046"

  file_path = LOAD_DIR / f"{file_name}.ply"
  visualize_ply(file_path)