import math
from beartype.typing import List
import trimesh

import numpy as np
import pyrender

import torch

from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.transforms import batch_transform_points
from splat_viewer.gaussians import Gaussians


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


def extract_instance_corner_points(gaussians: Gaussians):

  mask = gaussians.instance_label != -1

  valid_labels = gaussians.instance_label[mask]
  unique_labels = torch.unique(valid_labels)

  corner_points = []
  for label in unique_labels:

    positions = gaussians.position[(gaussians.instance_label == label).squeeze()]
    corner_points.append((torch.min(positions, dim=0)[0], torch.max(positions, dim=0)[0]))

  return corner_points


def make_bounding_box(gaussians: Gaussians):
  all_vertices = []
  all_indices = []
  current_vertex_count = 0
  
  for (min_coords, max_coords) in extract_instance_corner_points(gaussians):
    min_x, min_y, min_z = min_coords.cpu().numpy()
    max_x, max_y, max_z = max_coords.cpu().numpy()

    vertices = np.array([
      [min_x, min_y, min_z], 
      [max_x, min_y, min_z], 
      [max_x, max_y, min_z], 
      [min_x, max_y, min_z], 
      [min_x, min_y, max_z], 
      [max_x, min_y, max_z], 
      [max_x, max_y, max_z], 
      [min_x, max_y, max_z]
    ])

    edges = np.array([
      [0, 1], [1, 2], [2, 3], [3, 0], 
      [4, 5], [5, 6], [6, 7], [7, 4],
      [0, 4], [1, 5], [2, 6], [3, 7]
    ], dtype=np.uint32) + current_vertex_count

    current_vertex_count += len(vertices)

    all_vertices.append(vertices)
    all_indices.append(edges)

  all_vertices = np.vstack(all_vertices)
  all_indices = np.vstack(all_indices)

  primitive = pyrender.Primitive(
    positions=all_vertices,
    indices=all_indices,
    mode=1,
    material=pyrender.MetallicRoughnessMaterial
              (doubleSided=True, wireframe=True, smooth=False, baseColorFactor=(255, 255, 0, 255))
  )

  mesh = pyrender.Mesh([primitive])

  return mesh



def make_sphere(radius=1.0, subdivisions=3, color=(0.0, 0.0, 1.0)):
  sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

  material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    baseColorFactor=[*color, 1.0],
  )
  return pyrender.Mesh.from_trimesh(sphere, smooth=True, material=material)