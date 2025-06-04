from typing import List
from splat_viewer.gaussians import Workspace
from splat_trainer.util.pointcloud import PointCloud
import open3d as o3d
import argparse
from torch.utils.dlpack import to_dlpack
import numpy as np
import torch
import torchvision.transforms.functional as F

from pykeops.torch import LazyTensor

def rgb_to_hsv(image: torch.Tensor) -> torch.Tensor:
    r, g, _ = image.unbind(dim=-1)

    # Implementation is based on
    # https://github.com/python-pillow/Pillow/blob/4174d4267616897df3746d315d5a2d0f82c656ee/src/libImaging/Convert.c#L330
    minc, maxc = torch.aminmax(image, dim=-1)

    # The algorithm erases S and H channel where `maxc = minc`. This avoids NaN
    # from happening in the results, because
    #   + S channel has division by `maxc`, which is zero only if `maxc = minc`
    #   + H channel has division by `(maxc - minc)`.
    #
    # Instead of overwriting NaN afterwards, we just prevent it from occurring so
    # we don't need to deal with it in case we save the NaN in a buffer in
    # backprop, if it is ever supported, but it doesn't hurt to do so.
    eqc = maxc == minc

    channels_range = maxc - minc
    # Since `eqc => channels_range = 0`, replacing denominator with 1 when `eqc` is fine.
    ones = torch.ones_like(maxc)
    s = channels_range / torch.where(eqc, ones, maxc)
    # Note that `eqc => maxc = minc = r = g = b`. So the following calculation
    # of `h` would reduce to `bc - gc + 2 + rc - bc + 4 + rc - bc = 6` so it
    # would not matter what values `rc`, `gc`, and `bc` have here, and thus
    # replacing denominator with 1 when `eqc` is fine.
    channels_range_divisor = torch.where(eqc, ones, channels_range).unsqueeze_(dim=-1)
    rc, gc, bc = ((maxc.unsqueeze(dim=-1) - image) / channels_range_divisor).unbind(dim=-1)

    mask_maxc_neq_r = maxc != r
    mask_maxc_eq_g = maxc == g

    hg = rc.add(2.0).sub_(bc).mul_(mask_maxc_eq_g & mask_maxc_neq_r)
    hr = bc.sub_(gc).mul_(~mask_maxc_neq_r)
    hb = gc.add_(4.0).sub_(rc).mul_(mask_maxc_neq_r.logical_and_(mask_maxc_eq_g.logical_not_()))

    h = hr.add_(hg).add_(hb)
    h = h.mul_(1.0 / 6.0).add_(1.0).fmod_(1.0)
    return torch.stack((h, s, maxc), dim=-1)




def smooth_colors_spatial(points: torch.Tensor, colors: torch.Tensor, k: int = 16) -> torch.Tensor:
    """Smooth colors by averaging with k nearest spatial neighbors"""
    
    N, D = points.shape
    x_i = LazyTensor(points.view(N, 1, D))  # (N, 1, D) samples
    x_j = LazyTensor(points.view(1, N, D))  # (1, N, D) samples

    # Compute pairwise squared distances
    D_ij = ((x_i - x_j) ** 2).sum(-1)  # (N, N) symbolic squared distances
    
    # Find k nearest neighbors using PyKeOps
    knn_indices = D_ij.argKmin(K=k, dim=1)  # (N, k) tensor of neighbor indices
    
    # Gather colors of k nearest neighbors
    knn_colors = colors[knn_indices]  # (N, k, 3)
    
    # Average the colors
    smoothed_colors = knn_colors.mean(dim=1)  # (N, 3)
    return smoothed_colors

def is_vegetation(colors: torch.Tensor, green_hue_tolerance: float = 0.1) -> torch.Tensor:
    hsv = rgb_to_hsv(colors)
    h, s, v = hsv.unbind(dim=-1)
    
    # Debug: show hue distribution
    q05, q25, q50, q75, q95 = torch.quantile(h, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    print(f"Hue quantiles: 5%={q05:.3f}, 25%={q25:.3f}, 50%={q50:.3f}, 75%={q75:.3f}, 95%={q95:.3f}")
    print(f"Hue mean: {h.mean():.3f}, std: {h.std():.3f}")
    
    green_center = 0.2 # Around 72 degrees - more yellow-green
    hue_min = green_center - green_hue_tolerance
    hue_max = green_center + green_hue_tolerance
    
    print(f"Looking for hues between {hue_min:.3f} and {hue_max:.3f}")
    
    mask = (h > hue_min) & (h < hue_max) & (s > 0.2)
    
    return mask


def load_cloud(path:str):
  workspace = Workspace.load(model_path=path)
  gaussians3d = workspace.load_model()

  gaussians3d = gaussians3d.crop_foreground()
  gaussians3d = gaussians3d[gaussians3d.alpha().squeeze() > 0.5]

  return PointCloud(
      points=gaussians3d.position,   
      colors=gaussians3d.get_colors(),
      batch_size=(gaussians3d.batch_size[0]))


def to_o3d(cloud:PointCloud, o3d_device:str = "CUDA:0"):
  device = o3d.core.Device(o3d_device)
  pcd = o3d.t.geometry.PointCloud(device=device)

  assert torch.all(cloud.points.isfinite()) and torch.all(cloud.colors.isfinite())
  
  # Move tensors to CUDA device matching o3d_device
  torch_device = f"cuda:{o3d_device.split(':')[1]}"
  points_tensor = cloud.points.to(torch_device).contiguous()
  colors_tensor = cloud.colors.to(torch_device).contiguous()

  pcd.point["positions"] = o3d.core.Tensor.from_dlpack(to_dlpack(points_tensor))
  pcd.point["colors"] = o3d.core.Tensor.from_dlpack(to_dlpack(colors_tensor))
  return pcd



def vis_clouds(clouds:List[PointCloud], o3d_device:str = "CUDA:0"):
  o3d_clouds = [to_o3d(cloud, o3d_device) for cloud in clouds]

  for i in range(1, len(o3d_clouds)):

    result = icp(o3d_clouds[i], o3d_clouds[0])
    print(result)
    print("Transformation matrix:")
    print(result.transformation.cpu().numpy())

    # Apply transformation to align first cloud to second
    o3d_clouds[i] = o3d_clouds[i].transform(result.transformation)

  print("Visualizing aligned clouds...")
  o3d.visualization.draw(o3d_clouds)


def icp(cloud1, cloud2, max_distance=0.05, max_iterations=300, init_transform=np.identity(4)):
    # Convert init_transform to tensor (Float32 like their example)
    init_transform_tensor = o3d.core.Tensor.eye(4, o3d.core.Dtype.Float32)
    
    # Use multi-scale ICP matching their example
    voxel_sizes = o3d.utility.DoubleVector([0.01, 0.005, 0.001])
    
    # Use stricter convergence criteria to actually converge
    criteria_list = [
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-4, max_iteration=200),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-5, max_iteration=100),
        o3d.t.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6, max_iteration=50)
    ]
    
    # Use their max correspondence distances
    max_correspondence_distances = o3d.utility.DoubleVector([0.1, 0.05, 0.01])
    
    icp_result = o3d.t.pipelines.registration.multi_scale_icp(
        source=cloud1,
        target=cloud2,
        voxel_sizes=voxel_sizes,
        criteria_list=criteria_list,
        max_correspondence_distances=max_correspondence_distances,
        init_source_to_target=init_transform_tensor,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPoint()
    )
    
    return icp_result

def filter_vegetation(cloud:PointCloud, green_hue_tolerance: float = 0.1, use_smoothing: bool = True) -> PointCloud:
  colors = cloud.colors
  if use_smoothing:
    colors = smooth_colors_spatial(cloud.points, cloud.colors, k=16)
  
  is_leaf = is_vegetation(colors, green_hue_tolerance)

  # # color the vegetation red
  # cloud.colors[is_leaf] = torch.tensor([1., 0., 0.])

  # return cloud
  return cloud[~is_leaf]

def main():


  parser = argparse.ArgumentParser()
  parser.add_argument("workspace_paths", type=str,  nargs="+")
  parser.add_argument("--o3d_device", type=str, default="CUDA:0")
  parser.add_argument("--green_hue_tolerance", type=float, default=0.1)
  args = parser.parse_args()

  clouds = [load_cloud(workspace_path) for workspace_path in args.workspace_paths]
  print(clouds)
  
  clouds = [filter_vegetation(cloud, args.green_hue_tolerance) for cloud in clouds]
  print(clouds)


  vis_clouds(clouds, args.o3d_device)

if __name__ == "__main__":
  main()