from dataclasses import replace

from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent
from beartype import beartype
import cv2

import math

from pathlib import Path

import numpy as np
from splat_annotate.camera.visibility import visibility
import torch
from splat_annotate.camera.fov import FOVCamera

from splat_annotate.gaussians.workspace import Workspace
from splat_annotate.gaussians import Gaussians
from splat_annotate.viewer.interaction import Interaction
from splat_annotate.viewer.interactions.scribble import ScribbleGeometric
from splat_annotate.viewer.renderer import Rendering, WorkspaceRenderer

    
from .interactions.fly_control import FlyControl
from .scene_camera import SceneCamera
from .settings import Settings, ViewMode



class SceneWidget(QtWidgets.QWidget):
  def __init__(self, settings:Settings = Settings(), renderer=None, parent=None):
    super(SceneWidget, self).__init__(parent=parent)

    SceneWidget.instance = self

    self.camera_state = Interaction()
    self.interaction = ScribbleGeometric()

    self.camera = SceneCamera()
    self.settings = settings
    self.renderer = renderer

    self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    self.setMouseTracking(True)

    
    self.cursor_pos: np.ndarray = np.array([0, 0])
    self.modifiers = Qt.KeyboardModifier.NoModifier
    self.keys_down: set[Qt.Key] = set()

    self.dirty = True

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.update_camera_state)
    self.timer.start(int(1000 / Settings.update_rate))



  def update_setting(self, **kwargs):
    self.settings = replace(self.settings, **kwargs)
    self.dirty = True

  @property 
  def gaussians(self) -> Gaussians:
    return self.workspace_renderer.gaussians


  def median_point(self, points: list[np.ndarray]) -> int:

    stacked = np.vstack(points)  # shape: (n, 3)
    median_coords = np.median(stacked, axis=0)
    # Compute the Euclidean distances from each point to the median coordinates
    dists = np.linalg.norm(stacked - median_coords, axis=1)
    return int(np.argmin(dists))

  def load_workspace(self, workspace:Workspace, gaussians:Gaussians):
    self.workspace = workspace

    gaussians = gaussians.to(self.settings.device)
    if gaussians.foreground is None:
      foreground, depths = visibility(workspace.cameras, gaussians.position)

      q = torch.quantile(depths, 0.75)
      mask = (foreground > 0.05 * len(workspace.cameras)) & (depths < q)
      gaussians = gaussians.replace(foreground=mask.unsqueeze(1))

    
    self.workspace_renderer = WorkspaceRenderer(workspace, gaussians, self.renderer)
    self.keypoints = self.read_keypoints()

    centers = [c.position for c in workspace.cameras]
    self.set_camera_index(self.median_point(centers))
    self.camera_state.transition(FlyControl())



  def update_workspace(self, gaussians:Gaussians, index:int | None=None):
    self.load_workspace(self.workspace, gaussians)
    if index is not None:
      self.set_camera_index(index)
    self.show()

  def update_gaussians(self, gaussians:Gaussians):
    self.workspace_renderer.update_gaussians(gaussians.to(self.settings.device))
    self.dirty = True

  def set_dirty(self):
    self.dirty = True

  @property
  def camera_path_file(self):
    return self.workspace.model_path / "camera_path.npy"

  def write_keypoints(self):
    np.save(self.camera_path_file, np.array(self.keypoints))
    print(f"Saved {len(self.keypoints)} keypoints to {self.camera_path_file}")

  def read_keypoints(self):
    if self.camera_path_file.exists():
      kp = list(np.load(self.camera_path_file))
      print(f"Loaded {len(kp)} keypoints from {self.camera_path_file}")
      return kp

    return []



  def set_camera_index(self, index:int):
    self.camera_state.transition(None)

    camera = self.workspace.cameras[index]
    print(f'Showing view from camera {index}, {camera.image_name}')
    self.zoom = 1.0

    print(camera)

    self.camera.set_camera(camera)
    self.camera_index = index
    self.dirty = True


  @property
  def image_size(self):
    w, h = self.size().width(), self.size().height()

    return w, h
  
  def sizeHint(self):
    return QtCore.QSize(1024, 768)

  def event(self, event: QEvent):

    if (self.interaction.trigger_event(event) or 
        self.camera_state.trigger_event(event)):
      return True
      
    return super(SceneWidget, self).event(event)

    

  def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> None:
    self.modifiers = event.modifiers()
    key = Qt.Key(event.key())
    self.keys_down.discard(key)


    return super().keyPressEvent(event)
  
  def focusOutEvent(self, event: QtGui.QFocusEvent):
    self.keys_down.clear()
    return super().focusOutEvent(event)
  
  def showEvent(self, event: QtGui.QShowEvent):
    # Activate interactions when widget becomes visible
    if not self.camera_state.active:
      self.camera_state._activate()
    return super().showEvent(event)    
  
  def _key_press_event(self, event: QtGui.QKeyEvent) -> bool:
    self.modifiers = event.modifiers()
    key = Qt.Key(event.key())
    self.keys_down.add(key)

    view_modes = {
      Qt.Key.Key_1 : ViewMode.Normal,
      Qt.Key.Key_2 : ViewMode.Points,
      Qt.Key.Key_3 : ViewMode.Depth,
      Qt.Key.Key_4 : ViewMode.DepthVar,

    }

    enable_disable = {
      Qt.Key.Key_0 : 'cropped',
      Qt.Key.Key_9 : 'initial_points',
      Qt.Key.Key_8 : 'cameras',
      Qt.Key.Key_7 : 'bounding_boxes',
      Qt.Key.Key_6 : 'filtered_points',
      Qt.Key.Key_5 : 'color_instances'
    }

    if event.key() == Qt.Key.Key_Print:
      self.save_snapshot()
      return True
  
    elif event.key() == Qt.Key.Key_BraceLeft:
      self.set_camera_index((self.camera_index - 1) % len(self.workspace.cameras))
      return True
    elif event.key() == Qt.Key.Key_BraceRight:
      self.set_camera_index((self.camera_index + 1) % len(self.workspace.cameras))
      return True
    

    elif event.key() == Qt.Key.Key_Equal: 
      self.camera.zoom(self.settings.zoom_discrete)
      self.dirty = True
      return True
    elif event.key() == Qt.Key.Key_Minus:
      self.camera.zoom(1/self.settings.zoom_discrete)
      self.dirty = True
      return True


    elif event.key() == Qt.Key.Key_O: 
      shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
      self.update_setting(depth_near = self.settings.depth_near * (0.9 if shift else 1/0.9))
      self.dirty = True
      return True
    elif event.key() == Qt.Key.Key_P: 
      shift = event.modifiers() & Qt.KeyboardModifier.ShiftModifier
      self.update_setting(depth_far = self.settings.depth_far * (0.9 if shift else 1/0.9))
      self.dirty = True
      return True



    elif event.key() in enable_disable:
      k = enable_disable[key]
      update = {k: not getattr(self.settings.show, k)}
      
      self.update_setting(show = replace(self.settings.show, **update))
      return True
    
    elif event.key() in view_modes.keys():
      k = view_modes[key]
      self.update_setting(view_mode = k)
      return True
    
    
    elif event.key() == Qt.Key.Key_Space:
      self.keypoints.append(self.camera.view_matrix)


    elif event.key() == Qt.Key.Key_Return:
      if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
        if self.window().isFullScreen():
          self.window().showNormal()
        else:
          self.window().showFullScreen()


    return False
  
  def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
    if not self._key_press_event(event):
      super().keyPressEvent(event)



  def update_camera_state(self):
    self.camera_state._update(1 / self.settings.update_rate)
    self.repaint()

  def resizeEvent(self, event: QtGui.QResizeEvent):
    self.dirty = True
    return super().resizeEvent(event)

  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    p = event.localPos()
    self.cursor_pos = np.array([p.x(), p.y()])
    return super().mouseMoveEvent(event)

  @property
  def current_point_3d(self) -> np.ndarray:
    return self.lookup_point_3d(self.cursor_pos)
    
  @property
  def rendering(self) -> Rendering:
    if self.workspace_renderer.rendering is None:
      raise ValueError("No depth render available")

    return self.workspace_renderer.rendering

  def unproject_point(self, p:np.ndarray, depth:float) -> np.ndarray:
    return self.rendering.camera.unproject_pixel(p, depth)
  
  def unproject_radius(self, p:np.ndarray, depth:float, radius:float
                       ) -> tuple[np.ndarray, float]:
    
    p1 = self.unproject_point(p, depth)
    p2 = self.unproject_point(p + np.array([radius, 0]), depth)

    return p1, float(np.linalg.norm(p2 - p1))


  def lookup_depth(self, p:np.ndarray) -> float:
    render = self.rendering
    
    p_arr = np.round(p).astype(np.int32)
    x = np.clip(p_arr[0], 0, render.depth.shape[1] - 1)
    y = np.clip(p_arr[1], 0, render.depth.shape[0] - 1)

    return render.depth[y, x].item()

  def lookup_depths(self, p:np.ndarray) -> np.ndarray:
    assert len(p.shape) == 2 and p.shape[1] == 2, f"Expected Nx2 array, got {p.shape}"
    render = self.rendering
    
    p_arr = np.round(p).astype(np.int32)
    x = np.clip(p_arr[:, 0], 0, render.depth.shape[1] - 1)
    y = np.clip(p_arr[:, 1], 0, render.depth.shape[0] - 1)

    return render.depth[y, x].cpu().numpy()    

  def from_numpy(self, a:np.ndarray):
    return torch.from_numpy(a).to(device=self.settings.device)


  @beartype
  def test_depths(self, p:np.ndarray, depth:np.ndarray, tol=0.98) -> np.ndarray:

    return ((depth * tol <= self.lookup_depths(p)) & 
            (p[:, 0] >= 0) & (p[:, 0] < self.image_size[0] - 1) & 
            (p[:, 1] >= 0) & (p[:, 1] < self.image_size[1] - 1))


  @property
  def depth_map(self) -> torch.Tensor:
    render = self.rendering
    return render.depth

  def lookup_point_3d(self, p:np.ndarray) -> np.ndarray:
    return  self.rendering.camera.unproject_pixel(p.reshape(1, 2), self.lookup_depth(p))
    


  def render_camera(self) -> FOVCamera:
    return self.camera.resized(np.array(self.image_size))
  
  def render_scene(self):
    camera = self.render_camera()

    self.view_image = np.ascontiguousarray(
      self.workspace_renderer.render(camera, self.settings))
        
    self.dirty = False
    return self.view_image

      
  def paintEvent(self, event: QtGui.QPaintEvent):
    with QtGui.QPainter(self) as painter:
      dirty = self.dirty
      if dirty:
        self.render_scene()

      image = QtGui.QImage(self.view_image.data, 
                  self.view_image.shape[1], self.view_image.shape[0],
                  self.view_image.strides[0],  
                  QtGui.QImage.Format.Format_RGB888)
      

      painter.drawImage(0, 0, image)

      self.interaction.paintEvent(event, dirty)
      
            
  def snapshot_file(self):
    pictures = Path.home() / "Pictures"
    filename = pictures / "snapshot.jpg"

    i = 0
    while filename.exists():
      i += 1
      filename = pictures / f"snapshot_{i}.jpg"

    return filename


  def render_tiled(self, camera:FOVCamera):
    tile_size = self.settings.snapshot_tile
    nw, nh = [int(math.ceil(x / tile_size)) 
              for x in camera.image_size]
    
    full_image = np.zeros((nh * tile_size, nw * tile_size, 3), dtype=np.uint8)
    
    for x in range(0, nw):
      for y in range(0, nh):
        tile_camera = camera.crop_offset_size(np.array([x * tile_size, y * tile_size]), 
                             np.array([tile_size, tile_size]))
        
        image = self.workspace_renderer.render(tile_camera, self.settings)
        tile = full_image[y * tile_size:(y + 1) * tile_size, 
                          x * tile_size:(x + 1) * tile_size, :] 
        
        print(tile.shape, image.shape, x, y)
        tile[:] = image
        
    return full_image[:camera.image_size[1], :camera.image_size[0]]

  def save_snapshot(self):
    camera = self.camera.resized(np.array(self.settings.snapshot_size))
    filename = self.snapshot_file()

    w, h = camera.image_size
    print(f"Rendering snapshot ({w}x{h})...")
    print(camera)

    image = self.render_tiled(camera)

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(filename), image, [cv2.IMWRITE_JPEG_QUALITY, 92])

    print(f"Saved to {filename}")



  def move_camera(self, delta:np.ndarray):
    self.camera.move(delta)
    self.dirty = True

  def rotate_camera(self, delta:np.ndarray):
    self.camera.rotate(delta)
    self.dirty = True

  def set_camera_pose(self, r:np.ndarray, t:np.ndarray):
    self.camera.set_pose(r, t)
    self.dirty = True





    

