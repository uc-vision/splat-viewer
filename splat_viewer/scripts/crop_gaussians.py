import argparse
from dataclasses import replace
from pathlib import Path
from typing import List

import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from splat_viewer.gaussians.loading import  read_gaussians, to_pcd, write_gaussians
from splat_viewer.gaussians.workspace import load_workspace


def crop_model(model, cameras:List[FOVCamera], args):
  is_near, is_visible = visibility(cameras, model.positions, near = args.near, far = args.far)

  min_views = max(1, len(cameras) * args.min_percent / 100)
  n = (is_visible > 0).sum(dtype=torch.int32)
  n_near = (is_near >= min_views).sum(dtype=torch.int32)

  print(f"Cropped model from {model.batch_shape} to {n} visible points, {n_near} near (at least {min_views} views)")
  model = replace(model, labels=(is_near >= min_views).to(torch.int32).reshape(-1, 1))

  model = model[is_visible > 0]
  return model

def main():

  parser = argparse.ArgumentParser(description="Extract a point cloud from a gaussian splatting model")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  parser.add_argument("--scan", type=str,  help="Input scan file")
  
  parser.add_argument("--far", default=torch.inf, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.2, type=float, help="Min depth to determine the visible ROI")


  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')

  parser.add_argument("--statistical_outliers", type=float, default=None)
  parser.add_argument("--radius_outliers", type=float, default=None)
  parser.add_argument("--knn", type=int, default=100)

  
  parser.add_argument("--write", action="store_true", help="Write the cropped model to a file")
  parser.add_argument("--show", action="store_true")

  args = parser.parse_args()

  assert args.show or args.write, "Nothing to do. Please specify --show or --write"


  workspace = load_workspace(args.model_path)

  with torch.inference_mode():
    workspace = load_workspace(args.model_path)

    model_name = workspace.latest_iteration()
    model_file = workspace.model_filename(model_name)
    model = read_gaussians(model_file)

  
    model = model.to(args.device)
    model = crop_model(model, workspace.cameras, args)
    
    idx_crop = torch.nonzero(model.labels.squeeze() > 0).squeeze()

    def filtered(mask):
      mask = torch.from_numpy(mask.numpy()).to(model.device)
      model.labels[idx_crop[~mask], :] = 0

    if any([args.radius_outliers, args.statistical_outliers]):
      pcd = to_pcd(model[idx_crop].cpu())

      if args.statistical_outliers is not None:
        _, keep = pcd.remove_statistical_outliers(nb_neighbors=args.knn, std_ratio=args.statistical_outliers)
        filtered(keep)

      if args.radius_outliers is not None:
        _, keep = pcd.remove_radius_outliers(nb_points=args.knn, search_radius=args.radius_outliers)
        filtered(keep)


      num_removed = idx_crop.shape[0] - (model.labels > 0).sum()
      print(f"Found {num_removed} outliers")

    if args.write:    
      write_gaussians(model_file, model)
      print(f"Wrote {model} to {model_file}")

    if args.show:
      from splat_viewer.viewer.viewer import show_workspace
      show_workspace(workspace, model)



if __name__ == "__main__":
  main()  






