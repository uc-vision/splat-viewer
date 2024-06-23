import argparse
from natsort import natsorted
from pathlib import Path
from beartype.typing import List
import shutil


def main():
  parser = argparse.ArgumentParser(description="Export a trained workspace (only the minimum files required)")
  parser.add_argument("model_path", type=Path)
  parser.add_argument("output", type=Path)

  args = parser.parse_args()

  clouds = natsorted([file for file in args.model_path.glob("point_cloud/iteration_*/point_cloud.ply")])

  if len(clouds) == 0:
    raise Exception("No point clouds found in {}".format(args.model_path))
  
  cloud_file = clouds[-1]
  print("Using point cloud {}".format(cloud_file))

  camera_file = args.model_path/"cameras.json"
  input_file = args.model_path/"input.ply"

  scene_file = args.model_path/"scene.json"
  cfg_file = args.model_path/"cfg_args"
  
  files:List[Path] = [camera_file, input_file, cloud_file, cfg_file, scene_file]

  for file in [camera_file, input_file]:
    if not file.exists():
      raise Exception("Missing file {}".format(file))
  
  
  args.output.mkdir(parents=True, exist_ok=True)
  for file in files:
    if not file.exists():
      continue

    filename = file.relative_to(args.model_path)
    out_filename = args.output/filename

    print(f"Copying {file} to {out_filename}")
    out_filename.parent.mkdir(parents=True, exist_ok=True)

    shutil.copyfile(file, out_filename)







if __name__=="__main__":
  main()
