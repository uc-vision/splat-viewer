
from dataclasses import replace
import itertools
from pathlib import Path
import time
from typing import List

import torch
from tqdm import tqdm
from splat_viewer.benchmark.arguments import BenchmarkArgs
from splat_viewer.camera.fov import FOVCamera


from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.gaussians.workspace import load_workspace
from splat_viewer.renderer.arguments import renderer_from_args

from torch.profiler import profile, record_function, ProfilerActivity


def bench_forward(gaussians, renderer, cameras, profiler=None, **kwargs):

    inputs = renderer.pack_inputs(gaussians)
    torch.cuda.synchronize()

    start_time = time.time()
    with torch.no_grad():
      for camera in tqdm(cameras):
        renderer.render(inputs, camera, **kwargs)

        if profiler is not None:
          profiler.step()

    torch.cuda.synchronize()
    return time.time() - start_time

def bench_backward(gaussians, renderer, cameras, profiler=None, **kwargs):
    inputs = renderer.pack_inputs(gaussians, requires_grad=True)
    torch.cuda.synchronize()

    start_time = time.time()
    for camera in tqdm(cameras):

      output = renderer.render(inputs, camera, **kwargs)
      loss = output.image.sum()
      loss.backward()

      if profiler is not None:
        profiler.step()

    torch.cuda.synchronize()
    return time.time() - start_time


def load_workspace_with(model_path:Path, args:BenchmarkArgs):
  workspace = load_workspace(model_path)
  if args.model is None:
      args.model = workspace.latest_iteration()
    
  gaussians:Gaussians = workspace.load_model(args.model).to(args.device)
  if args.sh_degree is not None:
    gaussians = gaussians.with_sh_degree(args.sh_degree)

  # sorting by scale makes the tile mapping algorithm more efficient
  if not args.no_sort:
    gaussians = gaussians.sorted()

  cameras = workspace.cameras
  if args.image_size is not None:
    cameras = [camera.resize_longest(args.image_size) for camera in cameras]

  return replace(workspace, cameras=cameras), gaussians


def tile_info(gaussians:Gaussians, camera:FOVCamera):
  from splat_viewer.renderer.tile_mapping import tile_mapper, RasterConfig
  from splat_viewer.renderer.perspective import to_camera_params
  from splat_viewer.renderer.tile_mapping import map_to_tiles, tile_mapper
  from splat_viewer.renderer.perspective import perspective

  config = RasterConfig(tile_size=16, tight_culling=True)
  camera_params = to_camera_params(camera, device=gaussians.device)

  mask = perspective.frustum_culling(gaussians, camera_params, margin_pixels=50)
  gaussians2d, depth = perspective.project_to_image(gaussians[mask], camera_params)
    
  overlap_to_point, tile_ranges = map_to_tiles(
    gaussians2d, depth, camera_params.image_size, config)
  
  image_size = pad_to_tile(camera.image_size, config.tile_size)

  mapper = tile_mapper.tile_mapper(config)
  overlap_offsets, total_overlap = mapper.generate_tile_overlaps(
    gaussians2d, image_size)

  cum_overlap_counts = torch.cat([overlap_offsets.cpu(), torch.tensor([total_overlap])])

  return dict(
    max_overlap = torch.max(cum_overlap_counts[1:] - cum_overlap_counts[:-1]).item(),
    num_tiles = (tile_ranges[:, 1] - tile_ranges[:, 0]).sum().item(),
  )


def model_info(model_path:Path, args:BenchmarkArgs):
  workspace, gaussians = load_workspace_with(model_path, args)

  info = [tile_info(gaussians, camera) for camera in workspace.cameras]

  return dict(
    num_points = gaussians.batch_size[0],
  )



def benchmark_model(model_path:Path, args:BenchmarkArgs):
  workspace, gaussians = load_workspace_with(model_path, args)
  renderer = renderer_from_args(args.renderer)

  print("")
  print(f"{model_path}: benchmarking with {gaussians.batch_size[0]} points")

  return benchmark_gaussians(gaussians, workspace.cameras, renderer, args)

def benchmark_gaussians(gaussians:Gaussians, cameras: List[FOVCamera],
                    renderer, args:BenchmarkArgs):


  def n_cameras(n):
    return list(itertools.islice(itertools.cycle(cameras), n))

  image_sizes = set([tuple(camera.image_size) for camera in cameras])
  print(f"Cameras: {len(cameras)}, Image sizes: {image_sizes}")

  bench_renders = bench_backward if args.backward else bench_forward


  warmup_size = 10
  print(f"Warmup @ {warmup_size} cameras")
  bench_renders(gaussians, renderer, n_cameras(warmup_size), render_depth=args.depth)

  render_cameras = cameras if args.n is None else n_cameras(args.n)
  num_cameras = len(render_cameras)
  print(f"Benchmark @ {len(render_cameras)} cameras:")

  if args.profile or args.trace:
    with profile(activities=[ProfilerActivity.CUDA], 
                record_shapes=True, with_stack=True) as prof:
      with record_function("model_inference"):
        ellapsed = bench_renders(gaussians, renderer, render_cameras, profiler=prof, render_depth=args.depth)
    result = prof.key_averages(group_by_stack_n=4).table(sort_by="self_cuda_time_total", 
                                      row_limit=25, max_name_column_width=100)
    
    if args.trace:
      print(f"Writing chrome trace file to {args.trace}")
      prof.export_chrome_trace(args.trace)

    if args.profile:
      print(result)

  else:
    ellapsed = bench_renders(gaussians, renderer, render_cameras, render_depth=args.depth)
  print(f"{num_cameras} images in {ellapsed:.2f}s at {num_cameras / ellapsed:.2f} images/s")

  return dict(image_sizes=image_sizes, ellapsed=ellapsed, num_cameras=num_cameras, rate=num_cameras / ellapsed)




