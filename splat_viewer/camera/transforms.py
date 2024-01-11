import numpy as np


def expand_identity(m, shape=(4, 4)):
  expanded = np.eye(*shape)
  expanded[0:m.shape[0], 0:m.shape[1]] = m
  return expanded


def join_rt(r, t):
  assert t.ndim == r.ndim - 1 and t.shape[-1] == 3 and r.shape[-2:] == (3, 3),\
    'got r.shape:' + str(r.shape) + ' and t.shape:' + str(t.shape)

  d = t.ndim
  m_34 = np.concatenate([r, np.expand_dims(t, d)], axis=d)
  row = np.broadcast_to(np.array([[0, 0, 0, 1]]), (*r.shape[:-2], 1, 4))
  return np.concatenate([m_34, row], axis=d - 1)


def split_rt(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3], m[..., :3, 3]


def translation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, 3]


def rotation(m):
  assert m.shape[-2:] == (4, 4)
  return m[..., :3, :3]


def cross_matrix(vec):
  return np.cross(vec, np.identity(vec.shape[0]) * -1)


def make_grid(width, height):
  return np.meshgrid(np.arange(0, width, dtype=np.float32),
                     np.arange(0, height, dtype=np.float32))


def make_grid_centred(width, height):
  return np.meshgrid(np.arange(0.5, width, dtype=np.float32),
                     np.arange(0.5, height, dtype=np.float32))


def transform_grid_homog(x, y, z, w, transform):
  points = np.stack([x, y, z, w], axis=2)
  transformed = points.reshape(
      (-1, 4)) @ np.transpose(transform.astype(np.float32))
  return transformed.reshape(*z.shape, 4)


def make_homog(points):
  shape = list(points.shape)
  shape[-1] = 1
  return np.concatenate([points, np.ones(shape, dtype=np.float32)], axis=-1)


def transform_grid(x, y, z, transform):
  """  transform points of (x, y, z) by 4x4 matrix """
  return transform_grid_homog(x, y, z, np.ones(z.shape, dtype=np.float32),
                              transform)


def uproject_invdepth(invdepth: np.ndarray, depth_t_disparity: np.ndarray):
  """ perspective transform points of (x, y, 1/depth) by 4x4 matrix """
  x, y = make_grid(invdepth.shape[1], invdepth.shape[0])
  return transform_grid_homog(x, y, np.ones(x.shape, dtype=x.dtype), invdepth,
                              depth_t_disparity)


def uproject_depth(depth, transform):
  x, y = make_grid(depth.shape[1], depth.shape[0])
  return transform_grid_homog(x * depth, y * depth, depth,
                              np.ones(x.shape, dtype=x.dtype), transform)


def uproject_disparity(disparity, transform):
  x, y = make_grid(disparity.shape[1], disparity.shape[0])
  return transform_grid_homog(x, y, disparity, np.ones(x.shape, dtype=x.dtype),
                              transform)


def transform_invdepth(invdepth, depth_t_disparity):
  """ transform image grid of inverse-depth image by 4x4 matrix
    returns: inverse-depth image in new coordinate system 
    """
  points = uproject_invdepth(invdepth, depth_t_disparity)
  return points[:, :, 3] / points[:, :, 2]


def transform_depth(depth, depth_t_disparity):
  """ transform depth image by 4x4 matrix
    returns: image in new coordinate system 
    """
  points = uproject_invdepth(1 / depth, depth_t_disparity)
  return points[:, :, 2] / points[:, :, 3]


def _batch_transform(transforms, points):
  assert points.shape[
      -1] == 3 and points.ndim == 2, 'transform_points: expected 3d points of Nx3, got:' + str(
          points.shape)
  assert transforms.shape[-2:] == (
      4, 4
  ) and transforms.ndim == 3, 'transform_points: expected Mx4x4, got:' + str(
      transforms.shape)

  homog = make_homog(points)
  transformed = transforms.reshape(transforms.shape[0], 1, 4, 4) @ homog.reshape(
      1, *homog.shape, 1)

  return transformed[..., 0].reshape([transforms.shape[0], -1, 4])


def batch_transform_points(transforms, points):
  return _batch_transform(transforms, points)[..., 0:3]


def batch_project_points(transforms, points):
  homog = _batch_transform(transforms, points) 
  return homog[..., 0:3] / homog[..., 3:4]


def _transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  homog = make_homog(points).reshape([-1, 4, 1])
  transformed = transform.reshape([1, 4, 4]) @ homog
  return transformed[..., 0].reshape(-1, 4)


def transform_points(transform, points):
  return _transform_points(transform, points)[..., 0:3]

def unproject_pixels(transform, xy, depth):
  points = np.concatenate([xy * depth, depth], axis=-1)

  homog = _transform_points(transform, points)
  return  homog[..., 0:3] / homog[..., 3:4]

def project_points(transform, xyz):
  homog = _transform_points(transform, xyz)
  depth = homog[..., 2:3]
  xy = homog[..., 0:2] 
  return (xy / depth), depth

def affine_transform_points(transform, points):
  assert points.shape[
      -1] == 3, 'affine_transform_points: expected 3d points of ...x3, got:' + str(
          points.shape)

  r, t = split_rt(transform)
  transformed = (r.reshape(1, 3, 3) @ points.reshape(-1, 3, 1)).reshape(-1,
                                                                        3) + t

  return transformed.reshape(points.shape)


def translate_33(tx, ty):
  return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def scale_33(sx, sy):
  return np.diag([sx, sy, 1])


def translate_44(tx, ty, tz):
  return np.array([
      [1, 0, 0, tx],
      [0, 1, 0, ty],
      [0, 0, 1, tz],
      [0, 0, 0, 1],
  ])


def scale_44(sx, sy, sz):
  return np.diag([sx, sy, sz, 1])


def dict_subset(dict, keys):
  return {k: dict[k] for k in keys}


def check_size(image, expected_size):
  size = image.shape[1], image.shape[0]
  assert size == tuple(
      expected_size), 'got: ' + str(size) + ', should be: ' + str(expected_size)


def estimate_rigid_transform(ref_points, points):
  assert ref_points.shape == points.shape and points.shape[0] >= 3,\
     'estimate_transform: expected at least 3 points, got:' + str(points.shape)

  centroid_ref = np.mean(ref_points, axis=0)
  centroid_points = np.mean(points, axis=0)

  centered_ref = ref_points - centroid_ref
  centered_points = points - centroid_points

  s = centered_ref.T @ centered_points

  u, s, vh = np.linalg.svd(s)
  r = np.dot(vh.T, u.T)

  if np.linalg.det(r) < 0:
    vh[-1, :] *= -1
    r = np.dot(vh.T, u.T)

  t = centroid_points - np.dot(r, centroid_ref)
  transform = join_rt(r, t)

  err = np.linalg.norm(transform_points(transform, ref_points) - points, axis=1)
  return transform, err



def normalize(v):
  return v / np.linalg.norm(v, axis=-1)
