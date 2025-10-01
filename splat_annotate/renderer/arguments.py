


from dataclasses import dataclass


def add_render_arguments(parser):
  parser.add_argument("--antialias", action="store_true", help="enable analytic antialiasing")
  parser.add_argument("--blur_cov", type=float, default=0.3, help="add isotropic gaussian blur with given covariance")
  return parser


@dataclass(frozen=True)
class RendererArgs:
  antialias: bool = False
  blur_cov: float = 0.3

def renderer_from_args(args:RendererArgs):
    from splat_annotate.renderer.taichi_splatting import GaussianRenderer

    return GaussianRenderer(antialias=args.antialias,
      blur_cov=args.blur_cov if not args.antialias else 0.0) 

def make_renderer_args(args):


  return RendererArgs(
    antialias=args.antialias,
    blur_cov=args.blur_cov
  )
  
