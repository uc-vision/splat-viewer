


from dataclasses import dataclass
from enum import Enum
from typing import Tuple


def add_render_arguments(parser):
  parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterizer")
  parser.add_argument("--no_tight_culling", action="store_true", help="disable tight (OBB) culling")
  parser.add_argument("--depth16", action="store_true", help="use 16 bit depth in sorting (default is 32 bit)")

  parser.add_argument("--taichi", action="store_true", help="use taichi renderer")
  parser.add_argument("--diff_gaussian", action="store_true", help="use diff gaussian renderer")

  parser.add_argument("--pixel_stride", type=str, default="2,2", help="pixel tile size for rasterizer, e.g. 2,2")
  return parser

  


class RendererImpl(Enum):
  TaichiSplatting = "taichi_splatting"
  Taichi3DGS      = "taichi_3d_gaussian_splatting"
  DiffGaussian    = "diff_gaussian_rasterization"


@dataclass(frozen=True)
class RendererArgs:
  tile_size: int = 16
  pixel_stride: Tuple[int, int] = (2, 2)
  no_tight_culling: bool = False
  depth16: bool = False

  impl: RendererImpl = RendererImpl.TaichiSplatting

def renderer_from_args(args:RendererArgs):
  if args.impl == RendererImpl.Taichi3DGS:
    from splat_viewer.renderer.taichi_3d_gaussian_splatting import TaichiRenderer
    return TaichiRenderer()
  elif args.impl == RendererImpl.DiffGaussian:
    from splat_viewer.renderer.diff_gaussian_rasterization import DiffGaussianRenderer
    return DiffGaussianRenderer()
  else:
    from splat_viewer.renderer.taichi_splatting import GaussianRenderer
    return GaussianRenderer(tile_size=args.tile_size, 
                                 tight_culling=not args.no_tight_culling,
                                 use_depth16=args.depth16,
                                 pixel_stride=args.pixel_stride) 

def make_renderer_args(args):
  if args.taichi:
    impl = RendererImpl.Taichi3DGS
  elif args.diff_gaussian:
    impl = RendererImpl.DiffGaussian
  else:
    impl = RendererImpl.TaichiSplatting

  pixel_stride = tuple(map(int, args.pixel_stride.split(','))) 

  return RendererArgs(
    tile_size=args.tile_size,
    no_tight_culling=args.no_tight_culling,
    depth16=args.depth16,
    pixel_stride=pixel_stride,
    impl=impl
  )