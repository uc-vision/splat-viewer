import argparse
from dataclasses import replace
from pathlib import Path
from beartype.typing import List

import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from splat_viewer.gaussians.loading import  read_gaussians, write_gaussians
from splat_viewer.gaussians.workspace import load_workspace


def label_model(model, cameras:List[FOVCamera], args):
  num_visible, min_distance = visibility(cameras, model.position, near = args.near)

  min_views = max(1, len(cameras) * args.min_percent / 100)

  is_near = (min_distance < args.far) & (num_visible > min_views)
  n_near = is_near.sum(dtype=torch.int32)

  print(f"Labelled {n_near} points as near ({100.0 * n_near / model.batch_size[0]:.2f}%)")
  model = model.replace(foreground=is_near.reshape(-1, 1))

  return model

def main():

  parser = argparse.ArgumentParser(description="Add a 'foreground' annotation to a .ply gaussian splatting file")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  
  parser.add_argument("--far", default=torch.inf, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.01, type=float, help="Min depth to determine the visible ROI")

  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')

  parser.add_argument("--write", action="store_true", help="Write the labelled moel back to the file")
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
    model = label_model(model, workspace.cameras, args)
    

    if args.write:    
      write_gaussians(model_file, model)
      print(f"Wrote {model} to {model_file}")

    if args.show:
      from splat_viewer.viewer.viewer import show_workspace
      show_workspace(workspace, model)



if __name__ == "__main__":
  main()  






