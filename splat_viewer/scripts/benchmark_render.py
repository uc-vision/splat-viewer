
import argparse
import itertools
import math
from pathlib import Path
import time

import numpy as np
import torch
from splat_viewer.camera.fov import FOVCamera
from tqdm import tqdm
from splat_viewer.gaussians.taichi_renderer import TaichiRenderer
from splat_viewer.gaussians.gaussian_renderer import GaussianRenderer

from splat_viewer.gaussians.workspace import load_workspace
import taichi as ti
from torch.profiler import profile, record_function, ProfilerActivity


def pad_to_tile(camera:FOVCamera, m:int):
  def next_mult(x):
      return int(math.ceil(x / m) * m)
      
  w, h = camera.image_size

  round_size = (next_mult(w), next_mult(h))
  return camera.pad_to(np.array(round_size))


def bench_forward(inputs, renderer, cameras):
    start_time = time.time()
    with torch.no_grad():
      for camera in tqdm(cameras):
        renderer.render_inputs(inputs, camera)
    torch.cuda.synchronize()
    ti.sync()
    return time.time() - start_time

def bench_backward(inputs, renderer, cameras):
    start_time = time.time()

    for camera in tqdm(cameras):
      inputs = [x.requires_grad_(True) for x in inputs]

      output = renderer.render_inputs(inputs, camera)
      loss = output.image.sum()
      loss.backward()

    torch.cuda.synchronize()
    ti.sync()
    return time.time() - start_time

def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("model_path", type=Path)
  parser.add_argument("--device", type=str, default="cuda:0")
  parser.add_argument("--model", type=str)
  parser.add_argument("--profile", action="store_true")
  parser.add_argument("--profile_torch", action="store_true")
  
  parser.add_argument("--debug", action="store_true")
  parser.add_argument("--n", type=int, default=1000)
  parser.add_argument("--tile_size", type=int, default=16)
  parser.add_argument("--backward", action="store_true")
  parser.add_argument("--taichi", action="store_true")


  args = parser.parse_args()

  ti.init(arch=ti.cuda, offline_cache=True, 
          debug=args.debug, kernel_profiler=args.profile, device_memory_GB=0.1)


  workspace = load_workspace(args.model_path)
  if args.model is None:
      args.model = workspace.latest_iteration()
    
  gaussians = workspace.load_model(args.model).to(args.device)


  renderer = TaichiRenderer() if args.taichi else GaussianRenderer() 
  renderer.update_settings(tile_size=args.tile_size)

  cameras = workspace.cameras # [pad_to_tile(camera, args.tile_size) for camera in workspace.cameras]

  def n_cameras(n):
    return list(itertools.islice(itertools.cycle(cameras), n))

  print(f"Benchmarking {args.model_path} with {gaussians.batch_shape[0]} points")

  image_sizes = set([tuple(camera.image_size) for camera in workspace.cameras])
  print(f"Cameras: {len(workspace.cameras)}, Image sizes: {image_sizes}")


  inputs = renderer.make_inputs(gaussians)
  bench_renders = bench_backward if args.backward else bench_forward

  print(f"Warmup @ {args.n // 10} cameras")
  bench_renders(inputs, renderer, n_cameras(args.n // 10))

  if args.profile:
    print("Profiling...")
    ti.profiler.clear_kernel_profiler_info()

  print(f"Benchmark @ {args.n} cameras:")

  if args.profile_torch:
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_inference"):
        ellapsed = bench_renders(inputs, renderer, n_cameras(args.n))
    result = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                       row_limit=15, max_name_column_width=45)
    print(result)
  else:
    ellapsed = bench_renders(inputs, renderer, n_cameras(args.n))
  print(f"{args.n} images in {ellapsed:.2f}s at {args.n / ellapsed:.2f} images/s")

  if args.profile:
    ti.profiler.print_kernel_profiler_info("count")

    

if __name__ == "__main__":
  main()  

