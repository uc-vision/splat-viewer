from dataclasses import replace
import traceback
from typing import List
from beartype.typing import Tuple, Optional

from PySide6 import QtGui, QtCore, QtWidgets
from PySide6.QtCore import Qt, QEvent, QObject, Signal

from beartype import beartype
import cv2

import math

from pathlib import Path

import numpy as np
import torch

from splat_viewer.camera.fov import FOVCamera
from splat_viewer.editor.gaussian_scene import GaussianScene

from splat_viewer.viewer.interaction import Interaction
from splat_viewer.viewer.renderer import WorkspaceRenderer

    
from .interactions.fly_control import FlyControl
from .scene_camera import SceneCamera
from .settings import Settings, ViewMode





class SceneWidget(QtWidgets.QWidget):
  settings_changed = QtCore.Signal(Settings)
  scene_changed = QtCore.Signal(GaussianScene, GaussianScene)
  
  def __init__(self, scene_renderer:WorkspaceRenderer, 
               settings:Settings = Settings(),         
               parent:Optional[QtWidgets.QWidget]=None):
    
    super(SceneWidget, self).__init__(parent=parent)

    SceneWidget.instance = self

    self.camera = SceneCamera()
    self.settings = settings
    self.scene_renderer = scene_renderer

    self.scene = GaussianScene.empty(device=self.settings.device)

    self.cursor_pos = (0, 0)
    self.modifiers = Qt.NoModifier
    self.keys_down = set()


    self.camera_state = FlyControl()
    self.tool = Interaction()

    self.setFocusPolicy(Qt.StrongFocus)
    self.setMouseTracking(True)

    self.timer = QtCore.QTimer(self)
    self.timer.timeout.connect(self.update)
    self.timer.start(1000 / Settings.update_rate)


  def update_setting(self, **kwargs):
    self.settings = replace(self.settings, **kwargs)

    self.settings_changed.emit(self.settings)

  
  def set_scene(self, previous:Optional[GaussianScene], scene:GaussianScene):
    self.scene = scene

    self.scene_changed.emit(previous, scene)
    self.tool.trigger_scene_changed(previous, scene)


  def set_cameras(self, cameras:List[FOVCamera]):
    self.cameras = cameras
    self.set_camera_index(0)



  def set_camera_index(self, index:int):
    self.camera_state.transition(None)

    camera = self.workspace.cameras[index]
    print('Showing view from camera', camera.image_name)
    self.zoom = 1.0

    self.camera.set_camera(camera)
    self.camera_index = index
    self.view_dirty = True


  @property
  def image_size(self):
    w, h = self.size().width(), self.size().height()

    return w, h
  
  def sizeHint(self):
    return QtCore.QSize(1024, 768)

  def event(self, event: QEvent):

    if (self.tool.trigger_event(event) or 
        self.camera_state.trigger_event(event)):
      return True
      
    return super(SceneWidget, self).event(event)

    

  def keyReleaseEvent(self, event: QtGui.QKeyEvent) -> bool:
    self.modifiers = event.modifiers()
    self.keys_down.discard(event.key())


    return super().keyPressEvent(event)
  
  def focusOutEvent(self, event: QtGui.QFocusEvent):
    self.keys_down.clear()
    return super().focusOutEvent(event)    
  
  def keyPressEvent(self, event: QtGui.QKeyEvent) -> bool:
    self.modifiers = event.modifiers()
    self.keys_down.add(event.key())

    view_modes = {
      Qt.Key_1 : ViewMode.Normal,
      Qt.Key_2 : ViewMode.Depth

    }

    enable_disable = {
      Qt.Key_0 : 'cropped',
      Qt.Key_9 : 'initial_points',
      Qt.Key_8 : 'cameras',
      Qt.Key_7 : 'bounding_boxes',
      Qt.Key_6 : 'filtered_points',
      Qt.Key_5 : 'color_instances'
    }

    if event.key() == Qt.Key_Print:
      self.save_snapshot()
      return True
  
    elif event.key() == Qt.Key_BraceLeft:
      self.set_camera_index((self.camera_index - 1) % len(self.workspace.cameras))
      return True
    elif event.key() == Qt.Key_BraceRight:
      self.set_camera_index((self.camera_index + 1) % len(self.workspace.cameras))
      return True
    

    elif event.key() == Qt.Key_Equal: 
      self.camera.zoom(self.settings.zoom_discrete)
      self.view_dirty = True
      return True
    elif event.key() == Qt.Key_Minus:
      self.camera.zoom(1/self.settings.zoom_discrete)
      self.view_dirty = True

      return True

    elif event.key() in enable_disable:
      k = enable_disable[event.key()]
      update = {k: not getattr(self.settings.show, k)}
      
      self.update_setting(show = replace(self.settings.show, **update))
      return True
    
    elif event.key() in view_modes:
      k = view_modes[event.key()]
      self.update_setting(view_mode = k)
      return True
    
  

    elif event.key() == Qt.Key_Return:
      if event.modifiers() & Qt.ShiftModifier:
        if self.window().isFullScreen():
          self.window().showNormal()
        else:
          self.window().showFullScreen()


    return super().keyPressEvent(event)
  

  def update(self):
    self.camera_state._update(1 / self.settings.update_rate)
    self.repaint()

  def resizeEvent(self, event: QtGui.QResizeEvent):
    self.view_dirty = True
    return super().resizeEvent(event)

  def mouseMoveEvent(self, event: QtGui.QMouseEvent):
    p = event.localPos()
    self.cursor_pos = (p.x(), p.y())
    return super().mouseMoveEvent(event)

  def current_point_3d(self) -> np.ndarray:
    return self.lookup_point_3d(self.cursor_pos)
    
  @property
  def rendering(self):
    if self.scene_renderer.rendering is None:
      raise ValueError("No depth render available")

    return self.scene_renderer.rendering

  def unproject_point(self, p:np.ndarray, depth:float) -> np.ndarray:
    render = self.rendering
    camera:FOVCamera = render.camera
    scene_point =  camera.unproject_pixel(*p, depth)
    return np.array([scene_point], dtype=np.float32)
  

  def unproject_radius(self, p:np.ndarray, depth:float, radius:float
                       ) -> Tuple[np.ndarray, float]:
    p = np.array(p)
    
    p1 = self.unproject_point(p, depth)
    p2 = self.unproject_point(p + np.array([radius, 0]), depth)

    return p1, np.linalg.norm(p2 - p1)


  def lookup_depth(self, p:Tuple[int, int]) -> np.ndarray:
    render = self.rendering
    
    p = np.round(p).astype(np.int32)
    x = np.clip(p[0], 0, render.depth.shape[1] - 1)
    y = np.clip(p[1], 0, render.depth.shape[0] - 1)

    return render.depth[y, x].item()

  def from_numpy(self, a:np.ndarray):
    return torch.from_numpy(a).to(device=self.settings.device)

  @beartype
  def lookup_depths(self, p:np.ndarray) -> np.ndarray:
    render = self.rendering
    p = np.round(p).astype(np.int32)

    x = np.clip(p[:, 0], 0, render.depth.shape[1] - 1)
    y = np.clip(p[:, 1], 0, render.depth.shape[0] - 1)

    x, y = self.from_numpy(x), self.from_numpy(y)
    return render.depth[y, x].cpu().numpy()

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
    render = self.rendering
    scene_point =  render.camera.unproject_pixel(*p, self.lookup_depth(p))
    return np.array([scene_point], dtype=np.float32)


  def render_camera(self) -> FOVCamera:
    return self.camera.resized(self.image_size)
  


  def render(self):
    camera = self.render_camera()

    self.view_image = np.ascontiguousarray(
      self.scene_renderer.render(self.scene , camera, self.settings))
        
    self.view_dirty = False
    return self.view_image

      
  def paintEvent(self, event: QtGui.QPaintEvent):
    view_dirty = self.view_dirty
    try:
      with QtGui.QPainter(self) as painter:
        self.render()

        image = QtGui.QImage(self.view_image.data, 
                    self.view_image.shape[1], self.view_image.shape[0],
                    self.view_image.strides[0],  
                    QtGui.QImage.Format_RGB888)
        

        painter.drawImage(0, 0, image)
        self.tool.trigger_paint(event, view_dirty)
    except Exception:
      traceback.print_exc()
      QtWidgets.QApplication.quit()
      
      
            
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
        tile_camera = camera.crop(np.array([x * tile_size, y * tile_size]), 
                             np.array([tile_size, tile_size]))
        
        image = self.scene_renderer.render(tile_camera, self.settings)
        tile = full_image[y * tile_size:(y + 1) * tile_size, 
                          x * tile_size:(x + 1) * tile_size, :] 
        
        print(tile.shape, image.shape, x, y)
        tile[:] = image
        
    return full_image[:camera.image_size[1], :camera.image_size[0]]

  def save_snapshot(self):
    camera = self.camera.resized(self.settings.snapshot_size)
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
    self.view_dirty = True

  def rotate_camera(self, delta:np.ndarray):
    self.camera.rotate(delta)
    self.view_dirty = True

  def set_camera_pose(self, r:np.ndarray, t:np.ndarray):
    self.camera.set_pose(r, t)
    self.view_dirty = True





    

