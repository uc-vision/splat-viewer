from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.scripts.noise_adder import NoiseAdder
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility
from scipy.spatial.transform import Rotation

from numpy.typing import NDArray

from beartype.typing import List
import torch
import open3d as o3d
import numpy as np
from argparse import Namespace
from splat_viewer.gaussians.data_types import Gaussians, Rendering
from beartype import beartype

from time import perf_counter
from splat_viewer.gaussians.workspace import Workspace
import small_gicp

class CameraIntrinsics:
  def __init__(self, focal_length, principal_point):
    self.focal_length = focal_length
    self.principal_point = principal_point

@beartype
def load_model(
  workspace:Workspace
)-> Gaussians:
  model_name = workspace.latest_iteration()
  model_file = workspace.model_filename(model_name)
  model = read_gaussians(model_file)
  return model

class ModelICP:
  def __init__(
    self,
    workspace:Workspace,
    args:Namespace
  ):

    self.cameras:list[FOVCamera] = workspace.cameras
    model:Gaussians = load_model(workspace) 
    model = model.to(args.device)
    model = self.crop_model(model, self.cameras, args) 
    self.model = model

    self.intrinsics = CameraIntrinsics(self.cameras[0].focal_length, self.cameras[0].principal_point)

    self.load_model_pcd(self.create_model_pcd(self.model))
    
    self.renderer = GaussianRenderer()
    self.noise_adder = NoiseAdder()

  @beartype
  def crop_model(
    self,
    model:Gaussians,
    cameras:List[FOVCamera],
    args:Namespace
    )->Gaussians:

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
  def model_pcd(self):
    return self._model_pcd
  
  @beartype
  def load_model_pcd(
    self,
    model_pcd:o3d.geometry.PointCloud
  ):
    self._model_pcd = model_pcd

  @beartype
  def get_w2c(
    self,
    camera:FOVCamera
  )->NDArray[np.float64]:

    return camera.world_t_camera
  
  @beartype
  def get_c2w(
    self,
    camera:FOVCamera
  )->NDArray[np.float64]:
    return camera.camera_t_world

  @beartype
  def render(
    self,
    camera:FOVCamera
    )->Rendering:

    "Return a Rendering from arg:camera position"
    return self.renderer.render(self.renderer.pack_inputs(self.model), camera)
  
  @beartype
  def get_camera(
    self,
    index:int
  ):
    
    "returns camera of the index from the camera list"
    return self.cameras[index]
  
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
  def apply_GICP(
    self,
    query_pcd:o3d.geometry.PointCloud,
    w2c:NDArray[np.float64]=np.eye(4)
  )->o3d.pipelines.registration.RegistrationResult:

    # estimate max_correspondence_distance
    bbox = self._model_pcd.get_axis_aligned_bounding_box().get_extent()
    threshold = 0.15 * np.max(bbox)  # 15% of largest cloud dimension

    distances = query_pcd.compute_nearest_neighbor_distance()
    radius = 2.5 * np.mean(distances)

    # estimate normals
    query_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))
    self._model_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))

    # gicp
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=30,
        relative_fitness=1e-6,
        relative_rmse=1e-6
    )

    reg_gicp = o3d.pipelines.registration.registration_generalized_icp(
        query_pcd, self._model_pcd, threshold, w2c,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria
    )

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

    target, target_tree = small_gicp.preprocess_points(np.asarray(self.model_pcd.points), downsampling_resolution=0.001)
    source, source_tree = small_gicp.preprocess_points(np.asarray(query_pcd.points), downsampling_resolution=0.001)

    result = small_gicp.align(target, source, target_tree, init_T_target_source=w2c, num_threads=8, registration_type="GICP")

    # print('--- registration result ---')
    # print(result)
    
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
    