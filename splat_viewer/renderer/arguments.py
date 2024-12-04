


from dataclasses import dataclass
from enum import Enum
from typing import Tuple


def add_render_arguments(parser):
  parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterizer")
  parser.add_argument("--antialias", action="store_true", help="enable analytic antialiasing")
  parser.add_argument("--depth16", action="store_true", help="use 16 bit depth in sorting (default is 32 bit)")


  parser.add_argument("--pixel_stride", type=str, default="2,2", help="pixel tile size for rasterizer, e.g. 2,2")
  return parser

  



@dataclass(frozen=True)
class RendererArgs:
  tile_size: int = 16
  pixel_stride: Tuple[int, int] = (2, 2)
  antialias: bool = False
  depth16: bool = False


def renderer_from_args(args:RendererArgs):
    from splat_viewer.renderer.taichi_splatting import GaussianRenderer
    return GaussianRenderer(tile_size=args.tile_size, 
                                 antialias=not args.antialias,
                                 use_depth16=args.depth16,
                                 pixel_stride=args.pixel_stride) 

def make_renderer_args(args):


  pixel_stride = tuple(map(int, args.pixel_stride.split(','))) 

  return RendererArgs(
    tile_size=args.tile_size,
    antialias=args.antialias,
    depth16=args.depth16,
    pixel_stride=pixel_stride,
  )
  