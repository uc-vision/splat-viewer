import argparse
from natsort import natsorted
from pathlib import Path
from typing import List
import shutil


def main():
  parser = argparse.ArgumentParser(description="Export model")
  parser.add_argument("model_path", type=Path)
  parser.add_argument("output", type=Path)

  args = parser.parse_args()

  clouds = natsorted([file for file in args.model_path.glob("point_cloud/iteration_*/point_cloud.ply")])

  if len(clouds) == 0:
    raise Exception("No point clouds found in {}".format(args.model_path))
  
  cloud_file = clouds[-1]
  print("Using point cloud {}".format(cloud_file))

  camera_file = args.model_path/"cameras.json"
  scene_file = args.model_path/"scene.json"
  input_file = args.model_path/"input.ply"
  cfg_file = args.model_path/"cfg_args"
  
  files:List[Path] = [camera_file, input_file, cloud_file, cfg_file, scene_file]
  
  for file in files:
    if not file.exists():
      raise Exception(f"File not found {file}")

  
  args.output.mkdir(parents=True, exist_ok=True)
  for file in files:
    filename = file.relative_to(args.model_path)
    out_filename = args.output/filename

    print(f"Copying {file} to {out_filename}")
    out_filename.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(file, out_filename)







if __name__=="__main__":
  main()
