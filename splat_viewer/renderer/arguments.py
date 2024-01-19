


def add_render_arguments(parser):
  parser.add_argument("--tile_size", type=int, default=16, help="tile size for rasterizer")
  parser.add_argument("--no_tight_culling", action="store_true", help="disable tight (OBB) culling")
  parser.add_argument("--depth16", action="store_true", help="use 16 bit depth in sorting (default is 32 bit)")

  parser.add_argument("--taichi", action="store_true", help="use taichi renderer")
  parser.add_argument("--diff_gaussian", action="store_true", help="use diff gaussian renderer")

  return parser


def renderer_from_args(args):
  if args.taichi:
    from splat_viewer.renderer.taichi_3d_gaussian_splatting import TaichiRenderer
    return TaichiRenderer()
  if args.diff_gaussian:
    from splat_viewer.renderer.diff_gaussian_rasterization import DiffGaussianRenderer
    return DiffGaussianRenderer()
  else:
    from splat_viewer.renderer.taichi_splatting import GaussianRenderer
    return GaussianRenderer(tile_size=args.tile_size, 
                                 tight_culling=not args.no_tight_culling,
                                 use_depth16=args.depth16) 
