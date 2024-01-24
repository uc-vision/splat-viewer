import argparse
from dataclasses import replace
from pathlib import Path
from beartype.typing import List

import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from splat_viewer.gaussians.loading import  read_gaussians, write_gaussians
from splat_viewer.gaussians.workspace import load_workspace


def crop_model(model, cameras:List[FOVCamera], args):
  is_near, is_visible = visibility(cameras, model.position, near = args.near, far = args.far)

  min_views = max(1, len(cameras) * args.min_percent / 100)
  n = (is_visible > 0).sum(dtype=torch.int32)
  n_near = (is_near >= min_views).sum(dtype=torch.int32)

  print(f"Cropped model from {model.batch_size[0]} to {n} visible points, {n_near} near (at least {min_views} views)")
  model = model.replace(foreground=(is_near >= min_views).reshape(-1, 1))

  model = model[is_visible > 0]
  return model

def main():

  parser = argparse.ArgumentParser(description="Add a 'foreground' annotation to a .ply gaussian splatting file")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  parser.add_argument("--scan", type=str,  help="Input scan file")
  
  parser.add_argument("--far", default=torch.inf, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.2, type=float, help="Min depth to determine the visible ROI")


  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')

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
    

    if args.write:    
      write_gaussians(model_file, model)
      print(f"Wrote {model} to {model_file}")

    if args.show:
      from splat_viewer.viewer.viewer import show_workspace
      show_workspace(workspace, model)



if __name__ == "__main__":
  main()  






