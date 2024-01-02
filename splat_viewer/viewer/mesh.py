import math
from typing import List

from camera_geometry.transforms import batch_transform_points
import trimesh

import numpy as np
import pyrender

from splat_viewer.camera.fov import FOVCamera

def instance_meshes(mesh:trimesh.Trimesh, transforms:np.array):
  vertices = batch_transform_points(transforms, mesh.vertices)
  n = transforms.shape[0]

  offsets = np.arange(n).reshape(n, 1, 1) * mesh.vertices.shape[0] 
  faces = mesh.faces.reshape(1, -1, 3) + offsets

  return trimesh.Trimesh(vertices=vertices.reshape(-1, 3), 
                         faces=faces.reshape(-1, 3))

def camera_marker(camera:FOVCamera, scale):
  fov = camera.fov

  x = math.tan(fov[0] / 2)
  y = math.tan(fov[1] / 2) 

  points = np.array([
    [0, 0, 0],
    [-x,  y, 1],
    [x,   y, 1],
    [x,  -y, 1],
    [-x, -y, 1]
    ])
  

  triangles = np.array([
    [0, 1, 2],
    [0, 2, 3],
    [0, 3, 4],
    [0, 4, 1],

    [1, 2, 3],
    [1, 3, 4]
  ], dtype=np.int32) 

  return trimesh.Trimesh(vertices=points * scale, faces=triangles, process=False) 
    

def make_camera_markers(cameras:List[FOVCamera], scale:float):
  mesh = camera_marker(cameras[0], scale)

  markers = instance_meshes(mesh, np.array([cam.world_t_camera for cam in cameras], dtype=np.float32))
  markers = pyrender.Mesh.from_trimesh(markers, wireframe=True, smooth=False,
                  material=pyrender.MetallicRoughnessMaterial
                  (doubleSided=True, wireframe=True, smooth=False, baseColorFactor=(255, 0, 0, 255)))


  return markers


def make_sphere(radius=1.0, subdivisions=3, color=(0.0, 0.0, 1.0)):
  sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

  material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    baseColorFactor=[*color, 1.0],
  )
  return pyrender.Mesh.from_trimesh(sphere, smooth=True, material=material)