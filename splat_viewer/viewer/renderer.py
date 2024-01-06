from dataclasses import dataclass, replace
import cv2

import numpy as np
import pyrender
import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.renderer.taichi_splatting import GaussianRenderer

from splat_viewer.gaussians.workspace import Workspace
from splat_viewer.gaussians import Gaussians, Rendering
from splat_viewer.viewer.scene_camera import to_pyrender_camera

    
from .mesh import make_camera_markers
from .settings import Settings, ViewMode


def get_cv_colormap(cmap):
  colormap = np.arange(256, dtype=np.uint8).reshape(1, -1)
  colormap = cv2.applyColorMap(colormap, cmap)
  return colormap.astype(np.uint8)


class PyrenderScene:
  
  def __init__(self,  workspace:Workspace):
    self.initial = workspace.load_initial_points()

    self.initial_scene = pyrender.Scene()

    self.points = pyrender.Mesh.from_points(
      self.initial.point['positions'].numpy(), self.initial.point['colors'].numpy())
    self.initial_node = self.initial_scene.add(self.points, pose=np.eye(4))

    self.initial_scene.ambient_light = np.array([1.0, 1.0, 1.0, 1.0])

    self.cameras = make_camera_markers(workspace.cameras, workspace.camera_extent / 50.)
    self.initial_scene.add(self.cameras)
    
    self.renderer = None


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
    scene =  self.initial_scene
    scene.add_node(node)

    image, depth = renderer.render(scene)
    scene.remove_node(node)

    return image, depth

@dataclass(frozen=True)
class RenderState:
  as_points : bool = False
  cropped : bool = False

  def update_setting(self, settings:Settings):
    return replace(self,
      as_points = settings.view_mode == ViewMode.Points,
      cropped = settings.show.cropped)


  def updated(self, gaussians:Gaussians) -> Gaussians:

    if self.as_points:
      gaussians = gaussians.with_fixed_scale(0.001)

    if self.cropped and gaussians.foreground is not None:
      gaussians = gaussians[gaussians.foreground.squeeze()]

    return gaussians

class Renderer:
  def __init__(self, workspace:Workspace, gaussians:Gaussians, gaussian_renderer):
    self.workspace = workspace

    self.gaussians = gaussians
    self.packed_gaussians = None
    self.render_state = RenderState()

    self.gaussian_renderer  = GaussianRenderer()
    # self.gaussian_renderer  =  TaichiRenderer()

    self.pyrender_scene = PyrenderScene(workspace)

    self.rendering = None
    self.color_map = torch.from_numpy(get_cv_colormap(cv2.COLORMAP_TURBO)
                                      ).squeeze(0).to(device=self.gaussians.device)


  def render_gaussians(self, camera, settings:Settings) -> Rendering:
    render_state = self.render_state.update_setting(settings)

    if self.packed_gaussians is None or self.render_state != render_state:
      self.packed_gaussians = self.gaussian_renderer.pack_inputs(
        render_state.updated(self.gaussians))
      
      self.render_state = render_state
    return self.gaussian_renderer.render(self.packed_gaussians, camera)


  def unproject_mask(self, camera:FOVCamera, 
                mask:torch.Tensor, alpha_multiplier=1.0, threshold=1.0):
    return self.gaussian_renderer.unproject_mask(self.gaussians, 
          camera, mask, alpha_multiplier, threshold)


  def colormap_torch(self, depth, near=0.1):
    depth = depth.clone()
    depth[depth == 0] = torch.inf

    min_depth = torch.clamp(depth, min=near).min()

    inv_depth =  (min_depth / depth).clamp(0, 1)
    inv_depth = (255 * inv_depth).to(torch.int)

    return (self.color_map[inv_depth])

  def colormap_np(self, depth, near=0.1):
    min_depth = np.clip(depth, a_min=near).min()

    inv_depth =  (min_depth / depth)
    inv_depth = (255 * inv_depth).astype(np.uint8)
    return cv2.applyColorMap(inv_depth, cv2.COLORMAP_TURBO)

  def render(self, camera, settings:Settings):
    show = settings.show

    with torch.inference_mode():      
      self.rendering = self.render_gaussians(camera, settings)   
    
    min_depth = self.workspace.camera_extent / 50.

    if settings.view_mode == ViewMode.Depth:
      image_gaussian = self.colormap_torch(self.rendering.depth, 
                  near = min_depth).to(torch.uint8).cpu().numpy()
    else:
      image_gaussian = (self.rendering.image.clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()

    if any([show.initial_points, show.cameras]):

      image, depth = self.pyrender_scene.render(camera, settings)
      depth_gaussian = self.rendering.depth.cpu().numpy()

      mask = np.bitwise_and(depth_gaussian > depth, depth > 0)
      return np.where(np.expand_dims(mask, [-1]), image, image_gaussian)
    
    else:
      return image_gaussian




          



    

