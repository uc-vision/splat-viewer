from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from numpy.typing import NDArray

from beartype.typing import List
import torch
import numpy as np
from argparse import Namespace
from splat_viewer.gaussians.data_types import Gaussians, Rendering
from beartype import beartype

from time import perf_counter
from splat_viewer.gaussians.workspace import Workspace

@beartype
def load_model(
  workspace:Workspace
)-> Gaussians:
  model_name = workspace.latest_iteration()
  model_file = workspace.model_filename(model_name)
  model = read_gaussians(model_file)
  return model

class CameraIntrinsics:
  def __init__(self, focal_length, principal_point):
    self.focal_length = focal_length
    self.principal_point = principal_point

class RenderingWorkspace:
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
    self.renderer = GaussianRenderer()

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
  def get_camera(
    self,
    index:int
  ):
    
    "returns camera of the index from the camera list"
    return self.cameras[index]
  
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
  def render(
    self,
    camera:FOVCamera
    )->Rendering:

    "Return a Rendering from arg:camera position"
    return self.renderer.render(self.renderer.pack_inputs(self.model), camera)
  
  