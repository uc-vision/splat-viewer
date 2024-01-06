
import argparse
import itertools
from pathlib import Path
import time

import torch
from tqdm import tqdm


from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.gaussians.workspace import load_workspace

import taichi as ti
from torch.profiler import profile, record_function, ProfilerActivity




def bench_forward(gaussians, renderer, cameras, **kwargs):
    inputs = renderer.pack_inputs(gaussians)
    torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
      for camera in tqdm(cameras):
        renderer.render(inputs, camera, **kwargs)
    torch.cuda.synchronize()
    return time.time() - start_time

def bench_backward(gaussians, renderer, cameras, **kwargs):
    inputs = renderer.pack_inputs(gaussians, requires_grad=True)
    torch.cuda.synchronize()

    start_time = time.time()
    for camera in tqdm(cameras):

      output = renderer.render(inputs, camera, **kwargs)
      loss = output.image.sum()
      loss.backward()

    torch.cuda.synchronize()
    return time.time() - start_time

def main():

  parser = argparse.ArgumentParser()

  parser.add_argument("model_path", type=Path, help="workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")
  parser.add_argument("--device", type=str, default="cuda:0", help="torch device to use")
  parser.add_argument("--model", type=str, default=None, help="model iteration to load from point_clouds folder")
  parser.add_argument("--profile", action="store_true", help="enable profiling")
  
  parser.add_argument("--debug", action="store_true", help="enable taichi kernels in debug mode")
  parser.add_argument("-n", type=int, default=500, help="number of iterations to render")
  parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterizer")
  parser.add_argument("--backward", action="store_true", help="benchmark backward pass")
  parser.add_argument("--sh_degree", type=int, default=None, help="modify spherical harmonics degree")
  parser.add_argument("--no_sort", action="store_true", help="disable sorting by scale (sorting makes tilemapping faster)")
  parser.add_argument("--depth", action="store_true", help="render depth maps")
  
  parser.add_argument("--taichi", action="store_true", help="use taichi renderer")
  

  args = parser.parse_args()

  ti.init(arch=ti.cuda, offline_cache=True, 
          debug=args.debug, kernel_profiler=args.profile, device_memory_GB=0.1)


  workspace = load_workspace(args.model_path)
  if args.model is None:
      args.model = workspace.latest_iteration()
    
  gaussians:Gaussians = workspace.load_model(args.model).to(args.device)
  if args.sh_degree is not None:
    gaussians = gaussians.with_sh_degree(args.sh_degree)

  # sorting by scale makes the tile mapping algorithm more efficient
  if not args.no_sort:
    gaussians = gaussians.sorted()

  if args.taichi:
    from splat_viewer.renderer.taichi_3d_gaussian_splatting import TaichiRenderer
    renderer = TaichiRenderer()
  else:
    renderer =  GaussianRenderer(tile_size=args.tile_size) 

  cameras = workspace.cameras # [pad_to_tile(camera, args.tile_size) for camera in workspace.cameras]

  def n_cameras(n):
    return list(itertools.islice(itertools.cycle(cameras), n))

  print(f"Benchmarking {args.model_path} with {gaussians.batch_size[0]} points")

  image_sizes = set([tuple(camera.image_size) for camera in workspace.cameras])
  print(f"Cameras: {len(workspace.cameras)}, Image sizes: {image_sizes}")


  bench_renders = bench_backward if args.backward else bench_forward

  warmup_size = args.n // 10
  print(f"Warmup @ {warmup_size} cameras")
  bench_renders(gaussians, renderer, n_cameras(warmup_size))


  print(f"Benchmark @ {args.n} cameras:")

  if args.profile:
    with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
      with record_function("model_inference"):
        ellapsed = bench_renders(gaussians, renderer, n_cameras(args.n), render_depth=args.depth)
    result = prof.key_averages().table(sort_by="self_cuda_time_total", 
                                       row_limit=25, max_name_column_width=70)
    print(result)
  else:
    ellapsed = bench_renders(gaussians, renderer, n_cameras(args.n), render_depth=args.depth)
  print(f"{args.n} images in {ellapsed:.2f}s at {args.n / ellapsed:.2f} images/s")

    

if __name__ == "__main__":
  main()  

