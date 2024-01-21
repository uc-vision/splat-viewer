import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional
from beartype import beartype

import torch
from splat_viewer.renderer.arguments import RendererArgs, add_render_arguments, make_renderer_args

@beartype
@dataclass 
class BenchmarkArgs:
  model_paths:List[Path]
  device:torch.device = torch.device('cuda', 0)
  model:Optional[str] = None

  profile:bool = False
  trace:Optional[str] = None
  backward:bool = False

  n:Optional[int] = None
  sh_degree:Optional[int] = None

  no_sort:bool = False
  depth:bool = False
  image_size:Optional[int] = None

  debug:bool = False
  renderer:RendererArgs = RendererArgs()
  

def benchmark_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("model_paths", type=Path, nargs='+', help="workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")

  parser.add_argument("--device", type=str, default="cuda:0", help="torch device to use")
  parser.add_argument("--model", type=str, default=None, help="model iteration to load from point_clouds folder")

  parser.add_argument("--profile", action="store_true", help="enable profiling")
  parser.add_argument("--trace", type=str, default=None, help="enable profiling with tensorboard trace for profiled data")
  parser.add_argument("--backward", action="store_true", help="benchmark backward pass")

  add_render_arguments(parser)

  parser.add_argument("--debug", action="store_true", help="enable taichi kernels in debug mode")
  parser.add_argument("-n", type=int, default=None, help="number of iterations to render (default all cameras))")

  parser.add_argument("--sh_degree", type=int, default=None, help="modify spherical harmonics degree")
  parser.add_argument("--no_sort", action="store_true", help="disable sorting by scale (sorting makes tilemapping faster)")

  parser.add_argument("--depth", action="store_true", help="render depth maps")
  parser.add_argument("--image_size", type=int, default=None, help="resize longest edge of camera image sizes")

  return parser


def make_benchmark_args(args) -> BenchmarkArgs:

  return BenchmarkArgs(
    model_paths = args.model_paths,
    device = torch.device(args.device),
    model = args.model,
    profile = args.profile,

    trace = args.trace,
    backward = args.backward,
    n = args.n,
    sh_degree = args.sh_degree,
    no_sort = args.no_sort,
    depth = args.depth,
    image_size = args.image_size,

    debug = args.debug,
    renderer = make_renderer_args(args))


def parse_benchmark_args():
  return make_benchmark_args(benchmark_args().parse_args())