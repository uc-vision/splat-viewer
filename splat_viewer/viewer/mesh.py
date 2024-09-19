from dataclasses import dataclass
import math
import beartype
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

@dataclass
class BoundingBox:
  lower: torch.Tensor
  upper: torch.Tensor

@beartype
def instance_boxes(gaussians: Gaussians) -> List[BoundingBox]:
  boxes = []

  mask = gaussians.instance_label != -1
  valid_labels = gaussians.instance_label[mask]
  unique_labels = torch.unique(valid_labels)

  for label in unique_labels:
    positions = gaussians.position[(gaussians.instance_label == label).squeeze()]
    boxes.append(BoundingBox(torch.min(positions, dim=0).values, torch.max(positions, dim=0).values))

  return boxes

@beartype
def make_solid_box(box:BoundingBox):
  min_coords = box.lower.cpu().numpy()
  max_coords = box.upper.cpu().numpy()

  box_vertices = np.array([
    [min_coords[0], min_coords[1], min_coords[2]],
    [max_coords[0], min_coords[1], min_coords[2]],
    [max_coords[0], max_coords[1], min_coords[2]],
    [min_coords[0], max_coords[1], min_coords[2]],
    [min_coords[0], min_coords[1], max_coords[2]],
    [max_coords[0], min_coords[1], max_coords[2]],
    [max_coords[0], max_coords[1], max_coords[2]],
    [min_coords[0], max_coords[1], max_coords[2]]
  ])

  box_faces = np.array([
    [0, 1, 2], [0, 2, 3],  # bottom face
    [4, 5, 6], [4, 6, 7],  # top face
    [0, 1, 5], [0, 5, 4],  # front face
    [2, 3, 7], [2, 7, 6],  # back face
    [0, 3, 7], [0, 7, 4],  # left face
    [1, 2, 6], [1, 6, 5]   # right face
  ])

  return pyrender.Mesh.from_trimesh(trimesh.Trimesh(vertices=box_vertices, faces=box_faces), material=pyrender.MetallicRoughnessMaterial(
    doubleSided=True, wireframe=True, smooth=False, baseColorFactor=(255, 255, 0, 255)
  ))



@beartype
def make_wire_boxes(boxes: List[BoundingBox]):
    if len(boxes) == 0:
        return pyrender.Mesh([])

    vertices = []
    indices = []

    for i, box in enumerate(boxes):
        min_coords = box.lower.cpu().numpy()
        max_coords = box.upper.cpu().numpy()

        box_vertices = np.array([
            min_coords,
            [max_coords[0], min_coords[1], min_coords[2]],
            [max_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], max_coords[1], min_coords[2]],
            [min_coords[0], min_coords[1], max_coords[2]],
            [max_coords[0], min_coords[1], max_coords[2]],
            max_coords,
            [min_coords[0], max_coords[1], max_coords[2]]
        ])

        box_indices = np.array([
            [0, 1], [1, 2], [2, 3], [3, 0], 
            [4, 5], [5, 6], [6, 7], [7, 4],
            [0, 4], [1, 5], [2, 6], [3, 7]
        ]) + i * 8

        vertices.append(box_vertices)
        indices.append(box_indices)

    all_vertices = np.vstack(vertices)
    all_indices = np.vstack(indices)

    primitive = pyrender.Primitive(
        positions=all_vertices,
        indices=all_indices,
        mode=1,
        material=pyrender.MetallicRoughnessMaterial(
            doubleSided=True, wireframe=True, smooth=False, baseColorFactor=(255, 255, 0, 255)
        )
    )

    return pyrender.Mesh([primitive])



def make_sphere(radius=1.0, subdivisions=3, color=(0.0, 0.0, 1.0)):
  sphere = trimesh.creation.icosphere(radius=radius, subdivisions=subdivisions)

  material = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    baseColorFactor=[*color, 1.0],
  )
  return pyrender.Mesh.from_trimesh(sphere, smooth=True, material=material)