from dataclasses import replace
import numpy as np
import pandas as pd
import torch

import taichi as ti
from splat_viewer.benchmark.arguments import benchmark_args, make_benchmark_args
from splat_viewer.benchmark.run import benchmark_model
from splat_viewer.renderer.arguments import RendererArgs, RendererImpl

def benchmark_models(args):
  summary = {}

  for model_path in args.model_paths:
    summary[model_path] = benchmark_model(model_path, args)
 
  print("Model\t Rate(images/sec)")
  for model_path, info in summary.items():
    print(f"{model_path}({info['num_cameras']})\t {info['rate']:.2f}")

  models = list(summary.keys())
  rates = [info['rate'] for info in summary.values()] 

  return models, rates


def main():
  torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)

  parser = benchmark_args()
  parser.add_argument("--write_to", type=str, default="benchmark.csv", help="output filename")

  cmd_args = parser.parse_args()
  args = make_benchmark_args(cmd_args)


  ti.init(arch=ti.cuda, offline_cache=True, log_level=ti.INFO,
          debug=args.debug, device_memory_GB=0.1)

  results = {}
  models = []

  for image_size in [1024, 2048, 4096]:
    impls = {}
    impls['taichi_16'] = RendererArgs(
          tile_size=16,
          depth16=True,
          impl=RendererImpl.TaichiSplatting
        )

    impls['diff_gaussian'] = RendererArgs(impl=RendererImpl.DiffGaussian)

    if image_size <= 2048:
      impls['taichi_3dgs'] = RendererArgs(impl = RendererImpl.Taichi3DGS)

    if image_size >= 2048:
      impls['taichi_32'] = RendererArgs(
          tile_size=32,
          depth16=True,
          impl=RendererImpl.TaichiSplatting
        )
    
    for impl_name, impl_args in impls.items():
      for backward in [False, True]:
          run_args = replace(args, 
                         renderer=impl_args,
                         image_size=image_size,
                         backward=backward)

          key = (impl_name, image_size, backward)

          print("------------------------------------")
          print(f"{impl_name}: image_size={image_size} {'forward+backward' if backward else 'forward'}")
          print("------------------------------------")

          models, results[key] = benchmark_models(run_args)
          


    index = pd.MultiIndex.from_tuples(
      results.keys(), names=['impl', 'image_size', 'backward'])

    df = pd.DataFrame(index=index, data=np.array(list(results.values())), columns=models)

    print("Results:")
    print(df)

    df.to_csv(cmd_args.write_to)
    print(f"Saved to {cmd_args.write_to}")

    

if __name__ == "__main__":
  main()  

