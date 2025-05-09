from splat_viewer.gaussians.loading import  read_gaussians
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.camera.visibility import visibility

from numpy.typing import NDArray

from beartype.typing import List
import torch
import open3d as o3d
import numpy as np
from argparse import Namespace
from splat_viewer.gaussians.data_types import Gaussians, Rendering
from beartype import beartype

from time import perf_counter
from splat_viewer.gaussians.workspace import Workspace

from splat_viewer.scripts.renderer_workspace import RenderingWorkspace

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Dict, Optional, Tuple, List

def extract_rt(Rt: NDArray[np.float64]) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    """
    Extract rotation matrix R and translation vector T from a 4x4 transformation matrix.
    This function is the inverse of join_rt.
    
    Parameters:
    -----------
    Rt : NDArray[np.float64]
        4x4 transformation matrix
    
    Returns:
    --------
    Tuple[NDArray[np.float64], NDArray[np.float64]]
        (R, T) where R is a 3x3 rotation matrix and T is a (3,) translation vector
    
    Raises:
    -------
    ValueError
        If the input matrix is not 4x4 or doesn't have the expected structure
    """
    # Input validation
    if Rt.shape != (4, 4):
        raise ValueError(f"Expected 4x4 transformation matrix, got shape {Rt.shape}")
    
    if abs(Rt[3, 3] - 1.0) > 1e-6 or np.any(np.abs(Rt[3, :3]) > 1e-6):
        raise ValueError("Invalid transformation matrix: Last row should be [0, 0, 0, 1]")
    
    # Extract rotation matrix (top-left 3x3)
    R = Rt[:3, :3].copy()
    
    # Extract translation vector (rightmost column, first 3 elements)
    T = Rt[:3, 3].copy()
    
    return R, T


class DifferentiableRenderer(RenderingWorkspace):
  def __init__(
    self,
    workspace:Workspace,
    args:Namespace
  ):
    super().__init__(workspace=workspace, args=args)

  def optimize_camera_pose(self, 
                        target_image: torch.Tensor, 
                        camera: FOVCamera,
                        num_iterations: int = 100,
                        learning_rate: float = 0.01) -> torch.Tensor:
    """
    Optimize camera pose to match target image using differentiable rendering.
    
    Args:
        target_image: Target image to match
        camera: Initial camera guess, contains camera extrinsics
        num_iterations: Number of optimization iterations
        learning_rate: Learning rate for optimization
        
    Returns:
        Optimized camera pose as a torch.Tensor
    """
    # Convert initial camera pose to tensor and ensure it requires gradients
    initial_camera_pose = camera.world_t_camera
    camera_pose = torch.from_numpy(initial_camera_pose).float().clone()
    camera_pose.requires_grad_(True)
    
    # Create optimizer with the actual parameter to optimize
    optimizer = torch.optim.Adam([camera_pose], lr=learning_rate)
    
    # Store original camera state to restore at the end
    original_position = camera.position.copy()
    original_rotation = camera.rotation.copy()
    
    # Optimization loop
    losses = []
    progress_bar = tqdm(range(num_iterations))
    for i in progress_bar:
        # Update camera with current pose estimate
        with torch.no_grad():
            Rt = camera_pose.detach().cpu().numpy()
            R_extracted, T_extracted = extract_rt(Rt)
            camera.position = T_extracted
            camera.rotation = R_extracted
        
        # Forward pass - render image
        rendering = self.render(camera)
        rendered_image = rendering.image
        
        # Compute loss
        loss = F.mse_loss(rendered_image, target_image)
        losses.append(loss.item())
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Update parameters
        optimizer.step()
        
        # Orthogonalize the rotation part (maintain valid rotation matrix)
        with torch.no_grad():
            R = camera_pose[:3, :3]
            U, _, V = torch.svd(R)
            R_new = U @ V.T
            camera_pose[:3, :3] = R_new
        
        # Update progress bar
        progress_bar.set_description(f"Loss: {loss.item():.6f}")
        
        # Optional: Early stopping if loss is very small
        if loss.item() < 1e-6:
            print(f"Converged at iteration {i}")
            break
    
    # Return optimized pose
    final_pose = camera_pose.detach()
    
    # Update camera with final optimized pose
    Rt = final_pose.cpu().numpy()
    R_extracted, T_extracted = extract_rt(Rt)
    camera.position = T_extracted
    camera.rotation = R_extracted
    
    # Optionally: plot loss curve
    if num_iterations > 1:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 5))
        plt.plot(losses)
        plt.title('Optimization Loss')
        plt.xlabel('Iteration')
        plt.ylabel('MSE Loss')
        plt.yscale('log')
        plt.grid(True)
        plt.show()
    
    return final_pose