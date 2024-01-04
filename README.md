# Overview

A viewer and some tools to work with guassian splatting reconstructions. Currently will open .ply files and gaussian splatting workspaces from the original guassian-splatting implementation. Intended primarily for testing [taichi-splatting](https://github.com/uc-vision/taichi-splatting)

# splat-viewer 

A gaussian splatting viewer.

## Arguments

```
usage: splat-viewer [-h] [--model MODEL] [--device DEVICE] [--debug] model_path

positional arguments:
  model_path       workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models

options:
  -h, --help       show this help message and exit
  --model MODEL    load model from point_clouds folder, default is latest iteration
  --device DEVICE  torch device to use
  --debug          enable taichi kernels in debug mode
```

## Keyboard Controls


###  Switch View mode 
* 1: normal rendering
* 2: render gaussian centers as points
* 3: render depth map

### Show/hide
* 0 : cropped foreground
* 9 : initial points
* 8 : camera markers
    
### Misc
 * prntsc: save high-res snapshot into workspace directory
 * shift return: toggle fullscreen
 
### Camera 
 * '[' : Prev camera
 * ']' : Next camera

 * '=' : zoom in
 * '-' : zoom out

 * w/s a/d q/e : forward/backward left/right up/down
 * keypad plus/minus: navigate faster/slower


### Animation
 * space: add current viewpoint to animaiton sequence
 * control-space: save current animation sequence to workspace folder
 * return: animate current sequence
 * shift plus/minus: animation speed faster/slower


# splat-benchmark

A benchmark to test forward and backward passes of differentiable renderers. 

## Arguments

```
usage: splat-benchmark [-h] [--device DEVICE] [--model MODEL] [--profile] [--debug] [-n N] [--tile_size TILE_SIZE] [--backward] [--sh_degree SH_DEGREE] [--no_sort] model_path

positional arguments:
  model_path            workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models

options:
  -h, --help            show this help message and exit
  --device DEVICE       torch device to use
  --model MODEL         model iteration to load from point_clouds folder
  --profile             enable profiling
  --debug               enable taichi kernels in debug mode
  -n N                  number of iterations to render
  --tile_size TILE_SIZE
                        tile size for rasterizer
  --backward            benchmark backward pass
  --sh_degree SH_DEGREE
                        modify spherical harmonics degree
  --no_sort             disable sorting by scale (sorting makes tilemapping faster)
```