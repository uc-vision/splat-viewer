import torch
import taichi as ti

from splat_viewer.benchmark.arguments import parse_benchmark_args
from splat_viewer.benchmark.run import benchmark_model


def main():
  torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)
    
  args = parse_benchmark_args()

  ti.init(arch=ti.cuda, offline_cache=True, log_level=ti.INFO,
          debug=args.debug, device_memory_GB=0.1)

  summary = {}

  for model_path in args.model_paths:
    summary[model_path] = benchmark_model(model_path, args)
 
  print("Model\t Rate(images/sec)")
  for model_path, info in summary.items():
    print(f"{model_path}({info['num_cameras']})\t {info['rate']:.2f}")

if __name__ == "__main__":
  main()  

