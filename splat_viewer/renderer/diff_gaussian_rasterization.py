from dataclasses import asdict, dataclass, replace
import math
import torch
import torch.nn.functional as F

from beartype import beartype
from splat_viewer.camera.fov import FOVCamera

from splat_viewer.gaussians.data_types import Gaussians, Rendering
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

def raster_settings(camera:FOVCamera, sh_degree=3, device=torch.device("cuda:0")):

    proj = camera.ndc_t_camera @ camera.camera_t_world
    view, proj, pos = [torch.from_numpy(t).to(device=device, dtype=torch.float32) 
            for t in (camera.camera_t_world, proj, camera.position)]

    width, height = camera.image_size
    fovW, fovH = camera.fov

    return GaussianRasterizationSettings(
        image_height=int(height),
        image_width=int(width),

        tanfovx=math.tan(fovW * 0.5),
        tanfovy=math.tan(fovH * 0.5),
        bg=torch.tensor([0.0, 0.0, 0.0], device=device),
        scale_modifier=1.0,
        viewmatrix=view.transpose(0, 1),
        projmatrix=proj.transpose(0, 1),
        sh_degree=sh_degree,
        campos=pos,
        prefiltered=False,
        debug=False
    )

  
@dataclass
class PackedGaussians:
  means2D : torch.Tensor
  means3D : torch.Tensor

  shs : torch.Tensor
  alphas : torch.Tensor
  scales : torch.Tensor
  rotations : torch.Tensor

  def requires_grad_(self, requires_grad):
    d = {k:t.requires_grad_(requires_grad) for k, t in asdict(self).items()}
      
    return PackedGaussians(**d)

class DiffGaussianRenderer:

  def __init__(self):
    pass

  @beartype
  def pack_inputs(self, gaussians:Gaussians, requires_grad=False):
      means2D = torch.zeros_like(gaussians.position, device=gaussians.device)

      return PackedGaussians(
        means2D=means2D,
        means3D=gaussians.position,
        shs=gaussians.sh_feature.transpose(1, 2).contiguous(),
        alphas=gaussians.alpha(),
        scales=gaussians.scale(),

        # necessary as we store rotation in xyzw - this uses wxyz
        rotations=torch.roll(F.normalize(gaussians.rotation, dim=1), 1, 1)
      ).requires_grad_(requires_grad)

  
  def update_settings(self, **kwargs):
    pass

  @beartype
  def render(self, inputs:PackedGaussians, camera:FOVCamera, render_depth:bool = True):
    
    sh_degree = int(math.sqrt(inputs.shs.shape[1])) - 1
    settings = raster_settings(camera, sh_degree=sh_degree, device=inputs.means2D.device)
    rasterizer = GaussianRasterizer(raster_settings=settings)

    image, radii = rasterizer.forward(
        means2D=inputs.means2D, means3D=inputs.means3D, shs=inputs.shs, 
        opacities=inputs.alphas, scales=inputs.scales, rotations=inputs.rotations)

    return Rendering(
      image=image.permute(1, 2, 0),
      camera=camera)

