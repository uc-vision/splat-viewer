from copy import deepcopy
from dataclasses import replace
from typing import Tuple
import trimesh
import pyrender

import numpy as np
from scipy.spatial.transform import Rotation as R

from splat_viewer.camera.fov import FOVCamera, join_rt


def normalize(v):
  return v / np.linalg.norm(v)


def to_pyrender_camera(camera:FOVCamera):
    fx, fy = camera.focal_length
    cx, cy = camera.principal_point

    pr_camera = pyrender.IntrinsicsCamera(fx, fy, cx, cy,
        znear=camera.near, zfar=camera.far)
    
    rotation = camera.rotation  @ flip_yz 
    m = join_rt(rotation, camera.position)

    return pyrender.Node(camera=pr_camera, matrix=m)


def look_at(eye, target, up=np.array([0., 0., 1.])):
  forward = normalize(target - eye)
  left = normalize(np.cross(up, forward))
  true_up = np.cross(forward, left)
  return np.stack([left, true_up, forward])


def look_at_pose(eye, target, up=np.array([0., 0., 1.])):
  pose = np.eye(4)
  pose[:3, :3] = look_at(eye, target, up)
  pose[:3, 3] = eye
  return pose

def make_sphere(pos, color, radius):
  sphere = trimesh.creation.icosphere(radius=radius)
  sphere.visual.vertex_colors = color
  sphere_mesh = pyrender.Mesh.from_trimesh(sphere)
  node = pyrender.Node(mesh=sphere_mesh, translation=pos)
  return node

def fov_to_focal(fov, image_size):
  return image_size / (2 * np.tan(fov / 2))

flip_yz = np.array([
  [1, 0, 0],
  [0, -1, 0],
  [0, 0, -1]
])


class SceneCamera:
  def __init__(self):

    self._camera = FOVCamera(position=np.array([0, 0, 0]),
                            rotation=np.eye(3),
                            focal_length=fov_to_focal(60, (640, 480)),
                            image_size = np.array((640, 480)),
                            image_name="viewport")

   
  def set_camera(self, camera):
    self._camera = deepcopy(camera)
   
 
  def look_at(self, pos:np.array, target:np.ndarray, up:np.ndarray=np.array([0, 0, 1])):
    self._camera = self._camera.with_pose(look_at_pose(pos, target, up))


  def resized(self, size:Tuple[int, int]):
    scale_factor = size[1] / self._camera.image_size[1]
    return self._camera.scale_size(scale_factor).pad_to(np.array(size))
  
  @property
  def view_matrix(self):
    return self._camera.world_t_camera

  @property 
  def rotation(self):
    return self._camera.rotation
  
  @rotation.setter
  def rotation(self, value):
    return self.set_pose(value, self.pos)

  @property 
  def pos(self):
    return self._camera.position
  
  @pos.setter
  def pos(self, value):
    return self.set_pose(self.rotation, value)

  
  def set_pose(self, r, t):
    self._camera = replace(self._camera, rotation=r, position=t)


  def move(self, delta:np.ndarray):
    self.pos += self.rotation @ delta

  def rotate(self, ypr):
    m = self.view_matrix
    m[:3, :3] =  m[:3, :3] @ R.from_euler('yxz', ypr).as_matrix()
     
    self.rotation = m[:3, :3]


  def zoom(self, factor):
    self._camera = self._camera.zoom(factor)