from dataclasses import dataclass, replace
from numbers import Number

from camera_geometry import Camera

from pathlib import Path

import numpy as np
import json

from camera_geometry.transforms import unproject_pixels, project_points

from beartype import beartype
from beartype.typing import Tuple


num_pair = np.ndarray | Tuple[Number, Number]
int_pair = np.ndarray | Tuple[int, int]

Box = Tuple[int, int, int, int]


@beartype
@dataclass
class FOVCamera:
   
  position: np.ndarray
  rotation: np.ndarray
  focal_length : np.ndarray # 2
  image_size : np.ndarray # 2

  image_name: str
  principal_point : np.ndarray = np.array([0., 0.]) 

  near:float  = 0.1
  far :float  = 200.0

  
  @property
  def aspect(self):
    width, height = self.image_size
    return width / height
  
  @property
  def width(self):
    return self.image_size[0]
  
  @property
  def height(self):
    return self.image_size[1]
  
  def scale_size(self, scale_factor) -> 'FOVCamera':

    return replace(self,
      image_size=np.round(self.image_size * scale_factor).astype(np.int32),
      focal_length=self.focal_length * scale_factor,
      principal_point=self.principal_point * scale_factor
    )
    
  def crop_offset_size(self, offset:num_pair, size:num_pair) -> 'FOVCamera':
    offset, size = np.array(offset), np.array(size)
    return replace(self,
      image_size=size.astype(np.int32),
      principal_point=self.principal_point - offset
    )
  
  def crop_extent(self, centre:num_pair, size:num_pair) -> 'FOVCamera':
      centre, size = np.array(centre), np.array(size)
      return self.crop_offset_size(centre - size / 2, size)

  def crop_box(self, box:Box) -> 'FOVCamera':
    x_min, x_max, y_min, y_max = box
    return self.crop_offset_size(
      np.array([x_min, y_min]),
      np.array([x_max - x_min, y_max - y_min])
    )
    
  def pad_to(self, image_size:num_pair) -> 'FOVCamera':
    image_size = np.array(image_size)
    return replace(self,
      image_size=image_size.astype(np.int32),
      principal_point=self.principal_point + (image_size - self.image_size) / 2
    )
    

  def resize_shortest(self, min_size, max_size) -> 'FOVCamera':
    shortest = min(self.image_size)
    scale = (min_size / shortest if shortest < min_size 
             else max_size / shortest)
        
    return self.scale_size(scale)

  def zoom(self, zoom_factor) -> 'FOVCamera':
    return replace(self, focal_length=self.focal_length * zoom_factor)


  @property
  def world_t_camera(self):
    return join_rt(self.rotation, self.position)
  
  @property
  def camera_t_world(self):
    return np.linalg.inv(self.world_t_camera)
  
  def __repr__(self):
    w, h = self.image_size
    fx, fy = self.focal_length
    cx, cy = self.principal_point
    return f"FOVCamera(name={self.image_name}@{w}x{h} pos={self.position}, z={self.forward}, fx={fx} fy={fy}, cx={cx} cy={cy})"

  def __str__(self):
    return repr(self)

  
  @property
  def right(self):
    return self.rotation[0]

  @property
  def up(self):
    return -self.rotation[1]

  @property
  def forward(self):
    return self.rotation[2]
  

  @property
  def fov(self):
    return np.arctan2(self.image_size, self.focal_length * 2) * 2 
  
  @property
  def intrinsic(self):
  
    cx, cy = self.principal_point
    fx, fy = self.focal_length

    return np.array(
      [[fx, 0,  cx],
        [0, fy, cy],
        [0, 0,  1]]
    )
  
  def unproject_pixels(self, xy:np.ndarray, depth:np.ndarray):
     return unproject_pixels(self.world_t_image, xy, depth)

  def project_points(self, points:np.ndarray):
    return project_points(self.image_t_world, points)
    

  def unproject_pixel(self, x, y, depth):
    points = self.unproject_pixels(np.array([[x, y]]), np.array([[depth]]))
    return tuple(points[0])
  
  def project_point(self, x, y, z):
    xy, depth = self.project_points(np.array([[x, y, z]]))
    return tuple([*xy[0], *depth[0]])

  @property
  def image_t_camera(self):
    m44 = np.eye(4)
    m44[:3, :3] = self.intrinsic
    
    return m44
  
  @property
  def image_t_world(self):
    return self.image_t_camera @ self.camera_t_world
  
  @property
  def world_t_image(self):
    return np.linalg.inv(self.image_t_world)

  @property
  def projection(self):
    return self.image_t_world
  
  @property
  def ncd_t_camera(self):
    """ OpenGL projection - Camera to Normalised Device Coordinates (NDC)
    """
    w, h = self.image_size

    cx, cy = self.principal_point
    fx, fy = self.focal_length
    n, f = self.near, self.far

    return np.array([[
          2.0 * fx / w,   0,              1.0 - 2.0 * cx / w,   0,
          0,              2.0 * fy / h,   2.0 * cy / h - 1.0,   0,
          0,              0,              (f + n) / (n - f),    (2 * f * n) / (n - f),
          0,              0,              -1.0,                 0
      ]], dtype=np.float32)
  
  @property
  def gl_camera_t_image(self):
    return np.linalg.inv(self.ncd_t_camera)

  @property
  def gl_camera_t_world(self):

    flip_yz = np.array([
      [1, 0, 0],
      [0, -1, 0],
      [0, 0, -1]
    ])

    rotation = self.rotation  @ flip_yz 
    return join_rt(rotation, self.position)

  @property
  def ndc_t_world(self):
    return self.ncd_t_camera @ self.gl_camera_t_world


def join_rt(R, T):
  Rt = np.zeros((4, 4))
  Rt[:3, :3] = R
  Rt[:3, 3] = T
  Rt[3, 3] = 1.0

  return Rt
         

def split_rt(Rt):
  R = Rt[:3, :3]
  T = Rt[:3, 3]
  return R, T



def camera_to_fov(camera:Camera) -> FOVCamera:
  assert camera.has_distortion == False, "Simple FOV camera does not have distortion"
  R, T = split_rt(camera.camera_t_parent)

  return FOVCamera(
    position = T,
    rotation = R,
    focal_length = np.array(camera.focal_length),
    principal_point = np.array(camera.principal_point),
    image_size = np.array(camera.image_size, dtype=np.int32),
  )


def from_json(camera_info) -> FOVCamera:
  pos = np.array(camera_info['position'])
  rotation = np.array(camera_info['rotation']).reshape(3, 3)
  w, h = (camera_info['width'], camera_info['height'])
  cx, cy = (camera_info.get('cx', w/2.), camera_info.get('cy', h/2.))
  

  return FOVCamera(
    position=pos,
    rotation=rotation,
    image_size=np.array([w, h], dtype=np.int32),
    focal_length=np.array([camera_info['fx'], camera_info['fy']]),
    principal_point=np.array([cx, cy]),
    image_name=camera_info['img_name']
  )



def load_camera_json(filename:Path):
  cameras = sorted(json.loads(filename.read_text()), key=lambda x: x['id'])

  return {camera_info['id']: from_json(camera_info) for camera_info in cameras}
  
