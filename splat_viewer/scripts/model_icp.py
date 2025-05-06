from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.scripts.noise_adder import NoiseAdder
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from beartype.typing import List
import torch
import open3d as o3d
import numpy as np

class CameraIntrinsics:
  def __init__(self, focal_length, principal_point):
    self.focal_length = focal_length
    self.principal_point = principal_point

def load_model(workspace):
  model_name = workspace.latest_iteration()
  model_file = workspace.model_filename(model_name)
  model = read_gaussians(model_file)
  return model

class ModelICP:
  def __init__(self, workspace, args):
    self.cameras = workspace.cameras
    model = load_model(workspace) 
    model = model.to(args.device)
    model = self.crop_model(model, self.cameras, args) 
    self.model = model

    self.intrinsics = CameraIntrinsics(self.cameras[0].focal_length, self.cameras[0].principal_point)

    self.load_model_pcd(self.create_model_pcd(self.model))
    
    self.renderer = GaussianRenderer()
    self.noise_adder = NoiseAdder()

  def crop_model(self, model, cameras:List[FOVCamera], args):
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
  
  def create_model_pcd(self, model):
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
  
  def load_model_pcd(self, model_pcd):
    self._model_pcd = model_pcd

  def get_w2c(self, camera):
    return camera.world_t_camera
  
  def get_c2w(self, camera):
    return camera.camera_t_world

  def render(self, camera):
    "Return a Rendering from arg:camera position"
    return self.renderer.render(self.renderer.pack_inputs(self.model), camera)

  def get_camera(self, index):
    "returns camera of the index from the camera list"
    return self.cameras[index]
  
  def define_intrinsics(self, image_colour):
    return o3d.camera.PinholeCameraIntrinsic(
        width=image_colour.shape[1],
        height=image_colour.shape[0],
        fx=self.intrinsics.focal_length[0],  
        fy=self.intrinsics.focal_length[1],  
        cx=self.intrinsics.principal_point[0],  
        cy=self.intrinsics.principal_point[1]   
    )

  def rgbd_from_depth_and_rgb(self, rgb, depth):
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(rgb),
        o3d.geometry.Image(depth),
        convert_rgb_to_intensity=False
    )

  def create_query_pcd(self, colour_image, depth_image):
    rgbd = self.rgbd_from_depth_and_rgb(depth=depth_image, rgb=colour_image)

    query_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd,
                self.define_intrinsics(colour_image))
    
    query_pcd = self.scale_pcd(query_pcd, 1000)
    
    return query_pcd

  def scale_pcd(self, pcd, scale_factor):
    return pcd.scale(scale_factor, [0,0,0])
  
  def apply_GICP(self, query_pcd, w2c=np.eye(4)):

    # Step 1: Auto-tune parameters
    bbox = self._model_pcd.get_axis_aligned_bounding_box().get_extent()
    threshold = 0.15 * np.max(bbox)  # 15% of largest cloud dimension

    distances = query_pcd.compute_nearest_neighbor_distance()
    radius = 2.5 * np.mean(distances)

    # Step 2: Estimate normals
    query_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))
    self._model_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
        radius=radius, max_nn=40))

    # Step 3: Run GICP
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        max_iteration=150,
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