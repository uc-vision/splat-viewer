from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.scripts.noise_adder import NoiseAdder
from splat_viewer.camera.visibility import visibility
from scipy.spatial.transform import Rotation

from numpy.typing import NDArray

import open3d as o3d
import numpy as np
from argparse import Namespace
from splat_viewer.gaussians.data_types import Gaussians
from beartype import beartype

from time import perf_counter
from splat_viewer.gaussians.workspace import Workspace
import small_gicp

from splat_viewer.scripts.renderer_workspace import RenderingWorkspace
import torch

import copy 


class ModelICP(RenderingWorkspace):
  def __init__(
    self,
    workspace:Workspace,
    args:Namespace
  ):
    super().__init__(workspace=workspace, args=args)
    self.load_model_pcd(self.create_model_pcd(self.model))
    self.noise_adder = NoiseAdder()

  @beartype
  def create_model_pcd(
    self,
    model:Gaussians
  )->o3d.geometry.PointCloud:
    # Create open3d pointcloud of the model
    xyz_np = model.position.cpu().numpy().astype(np.float64)  # Shape [N, 3]

    model_pcd = o3d.geometry.PointCloud()
    model_pcd.points = o3d.utility.Vector3dVector(xyz_np)  

    colors_np = model.get_colors().cpu().numpy().astype(np.float64)
    model_pcd.colors = o3d.utility.Vector3dVector(colors_np)
    
    return model_pcd
  
  @property
  def num_cameras(self):
    return len(self.cameras)

  @property
  def model_pcd(self)->o3d.geometry.PointCloud:
    return self._model_pcd
  
  @property
  def model_pcd_t(self)->o3d.t.geometry.PointCloud:
    return self._model_pcd_t

  @beartype
  def load_model_pcd(
    self,
    model_pcd:o3d.geometry.PointCloud
  ):
    self._model_pcd = model_pcd

  @beartype
  def load_model_pcd_t(
    self
  ):
    self._model_pcd_t = o3d.t.geometry.PointCloud.from_legacy(self.model_pcd, device=o3d.core.Device("CUDA:0"))

  @beartype
  def define_intrinsics(
    self,
    image_colour:NDArray[np.uint8]
  )->o3d.camera.PinholeCameraIntrinsic:
    
    return o3d.camera.PinholeCameraIntrinsic(
        width=image_colour.shape[1],
        height=image_colour.shape[0],
        fx=self.intrinsics.focal_length[0],  
        fy=self.intrinsics.focal_length[1],  
        cx=self.intrinsics.principal_point[0],  
        cy=self.intrinsics.principal_point[1]   
    )
  
  @beartype
  def rgbd_from_depth_and_rgb(
    self,
    rgb:NDArray[np.uint8],
    depth:NDArray[np.float32]
  )->o3d.geometry.RGBDImage:
    
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False
    )
  
  @beartype
  def create_query_pcd(
    self,
    colour_image:NDArray[np.uint8],
    depth_image:NDArray[np.float32]
  )->o3d.geometry.PointCloud:
    
    rgbd = self.rgbd_from_depth_and_rgb(depth=depth_image, rgb=colour_image)

    query_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
                self.define_intrinsics(colour_image))
    
    query_pcd = self.scale_pcd(query_pcd, 1000)
    
    return query_pcd

  @beartype
  def scale_pcd(
    self,
    pcd:o3d.geometry.PointCloud,
    scale_factor:int
  )->o3d.geometry.PointCloud:
    
    return pcd.scale(scale_factor, [0,0,0])
  
  @beartype
  def apply_tGICP(
      self,
      query_pcd: o3d.t.geometry.PointCloud,
      w2c: o3d.core.Tensor = o3d.core.Tensor.eye(4)
  ) -> o3d.t.pipelines.registration.RegistrationResult:
      
      # print(f"cuda available: {o3d.core.cuda.is_available()}")  # Should return True
      # print(f"query dev: {query_pcd.device}") # returns CUDA:0
      # print(f"model dev: {model_pcd_t_copy.device}") # returns CUDA:0
      # print(f"init T dev: {w2c.device}") # returns CUDA:0
      # print("Transform dtype:", w2c.dtype)

      # print("Query PCD points:", len(query_pcd.point.positions)) # about 390k
      # print("Query dtype:", query_pcd.point.positions.dtype)

      # print("Model PCD points:", len(self.model_pcd_t.point.positions)) # about 1.2 mill
      # print("Model dtype:", self.model_pcd_t.point.positions.dtype)

      start = perf_counter()

      criteria = o3d.t.pipelines.registration.ICPConvergenceCriteria(
          max_iteration=50,
          relative_fitness=1e-6,
          relative_rmse=1e-6
      )

      reg_gicp = o3d.t.pipelines.registration.icp(
          source=query_pcd,
          target=self.model_pcd_t,
          init_source_to_target = w2c,
          criteria=criteria,
          voxel_size = 0.0075,
          max_correspondence_distance=0.3
      )
      end = perf_counter()
      print(f"Elapsed Time: {end-start} seconds")
      print("RMSE:", reg_gicp.inlier_rmse)
      # print("num_it:", reg_gicp.num_iterations)
      # print("converged: ", reg_gicp.converged)

      return reg_gicp

  @beartype
  def apply_GICP(
    self,
    query_pcd:o3d.geometry.PointCloud,
    w2c:NDArray[np.float64]=np.eye(4)
  )->o3d.pipelines.registration.RegistrationResult:

    print("Performing CPU o3e ICP")
    start = perf_counter()

    # estimate max_correspondence_distance
    bbox = self.model_pcd.get_axis_aligned_bounding_box().get_extent()
    threshold = 0.15 * np.max(bbox)  # 15% of largest cloud dimension

    print(threshold)

    distances = query_pcd.compute_nearest_neighbor_distance()
    radius = 2.5 * np.mean(distances)

    # estimate normals
    query_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))
    self.model_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))

    # gicp
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )

    print("Query PCD points:", len(query_pcd.points))
    print("Model PCD points:", len(self.model_pcd.points))


    reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
        query_pcd, self.model_pcd, threshold, w2c,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria
    )

    end = perf_counter()
    print(f"Elapsed Time: {end-start} seconds")

    # Results
    # print("Generalized ICP Transformation Matrix:\n", reg_gicp.transformation)
    # print("Fitness Score (inlier ratio):", reg_gicp.fitness)  # Range [0, 1], higher is better
    print("RMSE:", reg_gicp.inlier_rmse)  # Root Mean Square Error of inliers

    return reg_gicp

  @beartype
  def apply_smallGICP(
    self,
    query_pcd:o3d.geometry.PointCloud,
    w2c:NDArray[np.float64]=np.eye(4)
    )->NDArray[np.float64]:

    target_points = np.asarray(self.model_pcd.points)
    source_points = np.asarray(query_pcd.points)

    start = perf_counter()

    result = small_gicp.align(
      target_points,
      source_points,
      init_T_target_source=w2c,
      num_threads=8,
      max_correspondence_distance=0.1,
      verbose=False,
      registration_type="GICP"
      )

    end = perf_counter()

    print(f"Time elapsed: {end-start} seconds")
    print('--- registration result ---')
    print(result)
    
    return result.T_target_source
  
  @beartype
  def verify_result(
    self,
    T_target_source:NDArray[np.float64],
    gt_T_target_source:NDArray[np.float64]
  ):
    
    # Compute error transform
    error = np.linalg.inv(T_target_source) @ gt_T_target_source
    
    # Translation error (Euclidean distance)
    error_trans = np.linalg.norm(error[:3, 3])
    
    # Rotation error (in degrees)
    error_rot = Rotation.from_matrix(error[:3, :3]).magnitude()
    error_rot_deg = np.degrees(error_rot)  # Convert to degrees
    
    # Per-axis translation errors
    error_x = error[0, 3]
    error_y = error[1, 3]
    error_z = error[2, 3]
    
    # Print detailed error report
    print("\n--- Alignment Error Report ---")
    print(f"Total Translation Error: {error_trans:.4f} m")
    print(f"  X-axis Error: {error_x:.4f} m")
    print(f"  Y-axis Error: {error_y:.4f} m") 
    print(f"  Z-axis Error: {error_z:.4f} m")
    print(f"Rotation Error: {error_rot_deg:.2f}°")
    
    # Check thresholds (0.05m and ~2.86°)
    trans_pass = error_trans < 0.001
    rot_pass = error_rot < 0.025  # ~0.025 radians = 1.43°
    
    print("\n--- Verification ---")
    print(f"Translation Check: {'PASS' if trans_pass else f'FAIL (threshold: 0.001m)'}")
    print(f"Rotation Check: {'PASS' if rot_pass else f'FAIL (threshold: {np.degrees(0.025):.2f}°)'}")
    