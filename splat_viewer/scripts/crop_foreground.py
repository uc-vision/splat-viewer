import argparse
from pathlib import Path
from typing import List

from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility
import torch

from splat_viewer.gaussians.loading import  read_gaussians, write_gaussians
from splat_viewer.gaussians.workspace import load_workspace



def crop_model(model, cameras:List[FOVCamera], args):
  num_visible, min_distance = visibility(cameras, model.position, near = args.near)

  min_views = max(1, len(cameras) * args.min_percent / 100)

  is_near = (min_distance < args.far) & (num_visible > min_views)
  n_near = is_near.sum(dtype=torch.int32)

  print(f"Cropped model to {n_near} points from {model.batch_size[0]} points")
  return model[is_near].to(model.device)


def main():

  parser = argparse.ArgumentParser(description="Add a 'foreground' annotation to a .ply gaussian splatting file")
  parser.add_argument("model_path", type=Path, help="Path to the gaussian splatting workspace")
  
  parser.add_argument("--far", default=torch.inf, type=float, help="Max depth to determine the visible ROI")
  parser.add_argument("--near", default=0.01, type=float, help="Min depth to determine the visible ROI")

  parser.add_argument("--min_percent", type=float, default=0, help="Minimum percent of views to be included")
  parser.add_argument("--device", default='cuda:0')

  parser.add_argument("--write_to", type=Path, help="Write the model to a ply file")
  parser.add_argument("--show", action="store_true")

  args = parser.parse_args()

  assert args.show or args.write_to, "Nothing to do. Please specify --show or --write_to"


  workspace = load_workspace(args.model_path)

  with torch.inference_mode():
    workspace = load_workspace(args.model_path)

    model_name = workspace.latest_iteration()
    model_file = workspace.model_filename(model_name)
    model = read_gaussians(model_file)

    model = model.to(args.device)
    model = crop_model(model, workspace.cameras, args)
    

    if args.write_to:    
      write_gaussians(args.write_to, model)
      print(f"Wrote {model} to {args.write_to}")

    if args.show:
      from splat_viewer.viewer.viewer import show_workspace
      show_workspace(workspace, model)



if __name__ == "__main__":
  main()  






