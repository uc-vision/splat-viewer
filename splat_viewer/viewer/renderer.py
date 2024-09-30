from dataclasses import dataclass, replace
from functools import cached_property
from typing import Tuple
import weakref
from beartype import beartype


import cv2

import numpy as np
import pyrender
import torch
from splat_viewer.camera.fov import FOVCamera
#from splat_viewer.renderer.diff_gaussian_rasterization import DiffGaussianRenderer

from splat_viewer.editor.gaussian_scene import GaussianScene


from splat_viewer.gaussians.workspace import Workspace
from splat_viewer.gaussians import Gaussians, Rendering
from splat_viewer.viewer.scene_camera import to_pyrender_camera

    
from .mesh import make_camera_markers
from .settings import Settings, ViewMode

import plyfile

def plyfile_to_mesh(plydata:plyfile.PlyData):
  vertex = plydata['vertex']
  positions = torch.stack(
    [ torch.from_numpy(vertex[i].copy()) for i in ['x', 'y', 'z']], dim=-1)

  colors = torch.stack(
    [ torch.from_numpy(vertex[i]) for i in ['red', 'green', 'blue']], dim=-1)
  
  return pyrender.Mesh.from_points(positions, colors)


def get_cv_colormap(cmap):
  colormap = np.arange(256, dtype=np.uint8).reshape(1, -1)
  colormap = cv2.applyColorMap(colormap, cmap)
  return colormap.astype(np.uint8)


class PyrenderScene:
  
  def __init__(self,  workspace:Workspace):
    self.bbox_node = None
    self.renderer = None

    self.seed_points = workspace.load_seed_points()
    self.scene = pyrender.Scene()

    self.points = plyfile_to_mesh(self.seed_points)
    self.initial_node = self.scene.add(self.points, pose=np.eye(4))

    self.scene.ambient_light = np.array([1.0, 1.0, 1.0, 1.0])

    self.cameras = make_camera_markers(workspace.cameras, workspace.camera_extent / 50.)
    self.scene.add(self.cameras)
    

  def create_renderer(self, camera, settings:Settings):
    if self.renderer is None:
      self.renderer = pyrender.OffscreenRenderer(camera.image_size[0], camera.image_size[1], point_size=settings.point_size)
    else:
      self.renderer.viewport_width = camera.image_size[0]
      self.renderer.viewport_height = camera.image_size[1]
      self.renderer.point_size = settings.point_size

    return self.renderer

  def render(self, camera, settings:Settings):
    
    renderer = self.create_renderer(camera, settings)

    self.cameras.is_visible = settings.show.cameras
    self.points.is_visible = settings.show.initial_points


    node = to_pyrender_camera(camera)
    self.scene.add_node(node)

    image, depth = renderer.render(self.scene)
    self.scene.remove_node(node)

    return image, depth

@dataclass(frozen=True)
class RenderState:
  fixed_size : bool = False
  fixed_opacity : bool = False
  cropped : bool = False
  filtered_points: bool = False
  color_instances: bool = False

  selected_color: Tuple[float, float, float] = (1., 0., 0.)

  def update_setting(self, settings:Settings):
    return replace(self,
      fixed_size = settings.show.fixed_size,
      fixed_opacity = settings.show.fixed_opacity,
      cropped = settings.show.cropped,
      filtered_points = settings.show.filtered_points,
      color_instances = settings.show.color_instances)

def random_colors(n:int, device:torch.device):
  return (torch.randn(n, 3, device=device) * 2).sigmoid()


@dataclass(frozen=True)
class SceneView:
  scene: GaussianScene
  state: RenderState


  @staticmethod
  def empty(device:torch.device):
    return SceneView(GaussianScene.empty(device=device), RenderState())

  @cached_property
  def instance_colors(self) -> torch.Tensor:
    instances = self.scene.instances

    max_id = max(instances.keys(), default=0)

    selected_color = torch.tensor(self.state.selected_color)
    colors = torch.zeros(max_id + 1, 3, device=self.scene.device)

    for i in instances.values():
      colors[i.id] = i.color

    for i in self.scene.selected_instances:
      colors[i] = selected_color

    return colors.to(self.scene.device)


  @cached_property
  def instance_color_mask(self) -> tuple[torch.Tensor, torch.Tensor]:
    colors = self.instance_colors
    
    instance_label = self.scene.gaussians.instance_label.squeeze(1)

    mask = instance_label >= 0
    valid_labels = instance_label[mask].long()

    return colors[valid_labels], mask

  @cached_property
  def rendered_gaussians(self) -> Gaussians:
    gaussians = self.scene.gaussians
    state = self.state

    if state.fixed_size:
      gaussians = gaussians.with_fixed_scale(0.001)

    if state.fixed_opacity:
      
      alpha_logit = torch.full_like(gaussians.alpha_logit, 10.0)
      gaussians = gaussians.replace(alpha_logit=alpha_logit)

    if state.cropped and gaussians.foreground is not None:
      gaussians = gaussians[gaussians.foreground.squeeze()]

    if state.color_instances and gaussians.instance_label is not None:

      instance_colors, instance_mask = self.instance_color_mask
      gaussians = gaussians.with_colors(instance_colors, instance_mask)

    if state.filtered_points and gaussians.label is not None:
      gaussians = gaussians[gaussians.label.squeeze() > 0.3]

    return gaussians


class WorkspaceRenderer:
  def __init__(self, workspace:Workspace, gaussian_renderer, device:torch.device):
    self.workspace = workspace

    self.packed_gaussians = None
    self.scene_view = SceneView.empty(device)

    self.gaussian_renderer = gaussian_renderer
    self.pyrender_scene = PyrenderScene(workspace)

    self.rendering = None
    self.color_map = torch.from_numpy(get_cv_colormap(cv2.COLORMAP_TURBO)
                                      ).squeeze(0).to(device=device)

  @property
  def render_state(self) -> RenderState:
    return self.scene_view.state


  def render_gaussians(self, scene:GaussianScene, camera:FOVCamera, settings:Settings) -> Rendering:
    render_state = self.render_state.update_setting(settings)
    if self.scene_view.scene is not scene or self.packed_gaussians is None or self.render_state != render_state:

      self.scene_view = SceneView(scene, render_state)
      self.packed_gaussians = self.gaussian_renderer.pack_inputs(self.scene_view.rendered_gaussians)
    
    return self.gaussian_renderer.render(self.packed_gaussians, camera)


  def colormap_torch(self, depth, near=0.1):
    depth = depth.clone()
    depth[depth <= 0] = torch.inf

    inv_depth =  (near / depth).clamp(0, 1)
    inv_depth = (255 * inv_depth).to(torch.int)

    return (self.color_map[inv_depth])



  def render(self, scene:GaussianScene, camera:FOVCamera, settings:Settings):
    show = settings.show

    with torch.inference_mode():      
      self.rendering = self.render_gaussians(scene, camera, settings)   
    
    min_depth = camera.near * settings.depth_scale

    depth = self.rendering.depth

    if settings.view_mode == ViewMode.Depth:
      image_gaussian = self.colormap_torch(depth, near = min_depth).to(torch.uint8).cpu().numpy()
    else:
      image_gaussian = (self.rendering.image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    if any([show.initial_points, show.cameras, show.bounding_boxes]):

      image, depth = self.pyrender_scene.render(camera, settings)
      depth_gaussian = self.rendering.depth.cpu().numpy()
      depth_gaussian[depth_gaussian == 0] = np.inf

      mask = np.bitwise_and(depth_gaussian > depth, depth > 0)
      return np.where(np.expand_dims(mask, [-1]), image, image_gaussian)
    
    else:
      return image_gaussian




          



    

