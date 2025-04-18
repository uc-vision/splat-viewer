from dataclasses import dataclass, replace
import cv2

import numpy as np
import pyrender
import torch
from splat_viewer.camera.fov import FOVCamera

from splat_viewer.gaussians.workspace import Workspace
from splat_viewer.gaussians import Gaussians, Rendering
from splat_viewer.viewer.scene_camera import to_pyrender_camera

    
from .mesh import make_camera_markers, make_bounding_box
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
  
  def __init__(self,  workspace:Workspace, gaussians: Gaussians):
    self.bbox_node = None
    self.renderer = None

    self.seed_points = workspace.load_seed_points()
    self.initial_scene = pyrender.Scene()

    self.points = plyfile_to_mesh(self.seed_points)
    self.initial_node = self.initial_scene.add(self.points, pose=np.eye(4))

    self.initial_scene.ambient_light = np.array([1.0, 1.0, 1.0, 1.0])

    self.cameras = make_camera_markers(workspace.cameras, workspace.camera_extent / 50.)
    self.initial_scene.add(self.cameras)
    
    self.update_gaussians(gaussians)


  
  def update_gaussians(self, gaussians:Gaussians):
    if self.bbox_node is not None:
      self.initial_scene.remove_node(self.bbox_node)

    if gaussians.instance_label is not None:
      bounding_boxes = make_bounding_box(gaussians)
      self.bbox_node = self.initial_scene.add(bounding_boxes)
    else:
      self.bbox_node = None


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

    if self.bbox_node is not None:
      self.bbox_node.mesh.is_visible = settings.show.bounding_boxes

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
  filtered_points: bool = False
  color_instances: bool = False

  def update_setting(self, settings:Settings):
    return replace(self,
      as_points = settings.view_mode == ViewMode.Points,
      cropped = settings.show.cropped,
      filtered_points = settings.show.filtered_points,
      color_instances = settings.show.color_instances)


  def updated(self, gaussians:Gaussians) -> Gaussians:

    if self.as_points:

      alpha_logit = torch.full_like(gaussians.alpha_logit, 10.0)
      gaussians = gaussians.with_fixed_scale(0.001).replace(alpha_logit=alpha_logit)

    if self.cropped and gaussians.foreground is not None:
      gaussians = gaussians[gaussians.foreground.squeeze()]

    if self.color_instances and gaussians.instance_label is not None:
    
      instance_mask = (gaussians.instance_label >= 0).squeeze()
      valid_instances = gaussians.instance_label[instance_mask].squeeze().long()

      unique_instance_labels = torch.unique(valid_instances)
      color_space = (torch.randn(unique_instance_labels.shape[0], 3, device=instance_mask.device) * 2).sigmoid()

      gaussians = gaussians.with_colors(color_space[valid_instances], instance_mask)

    if self.filtered_points and gaussians.label is not None:
      gaussians = gaussians[gaussians.label.squeeze() > 0.3]

    return gaussians

class WorkspaceRenderer:
  def __init__(self, workspace:Workspace, gaussians:Gaussians, gaussian_renderer):
    self.workspace = workspace

    self.gaussians = gaussians
    self.packed_gaussians = None
    self.render_state = RenderState()

    self.gaussian_renderer = gaussian_renderer

    self.pyrender_scene = PyrenderScene(workspace, gaussians)

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

  def update_gaussians(self, gaussians:Gaussians):
    self.gaussians = gaussians
    self.packed_gaussians = None
    
    self.pyrender_scene.update_gaussians(gaussians)

  def unproject_mask(self, camera:FOVCamera, 
                mask:torch.Tensor, alpha_multiplier=1.0, threshold=1.0):
    return self.gaussian_renderer.unproject_mask(self.gaussians, 
          camera, mask, alpha_multiplier, threshold)


  def colormap_torch(self, depth, near_point=0.2):
    depth = depth.clone()
    depth[depth <= 0] = torch.inf


    inv_depth =  (near_point / depth).clamp(0, 1)
    inv_depth = (255 * inv_depth).to(torch.int)

    return (self.color_map[inv_depth])

  def colormap_np(self, depth, near_point=0.2):

    inv_depth =  (near_point / depth)
    inv_depth = (255 * inv_depth).astype(np.uint8)
    return cv2.applyColorMap(inv_depth, cv2.COLORMAP_TURBO)

  def render(self, camera, settings:Settings):
    show = settings.show

    with torch.inference_mode():      
      self.rendering = self.render_gaussians(camera, settings)   
    
    near_point = (self.workspace.camera_extent / 3.) * settings.depth_scale
    depth = self.rendering.depth

    if settings.view_mode == ViewMode.Depth:
      image_gaussian = self.colormap_torch(depth, near_point = near_point).to(torch.uint8).cpu().numpy()
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




          



    

