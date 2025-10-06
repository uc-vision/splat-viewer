import argparse
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import torch
from splat_viewer.camera.fov import FOVCamera
from splat_viewer.gaussians.workspace import Workspace
from splat_viewer.renderer.arguments import add_render_arguments, make_renderer_args, renderer_from_args
from splat_viewer.viewer.renderer import WorkspaceRenderer
from splat_viewer.viewer.scene_camera import SceneCamera
from splat_viewer.viewer.settings import Settings
from taichi_splatting import TaichiQueue
import taichi as ti

# Import shared utilities
from splat_viewer.camera.camera_path import read_camera_path
from splat_viewer.camera.camera_animation import CameraPathAnimator, AnimationConfig
from .argument_parser import add_settings_arguments, create_settings_from_args


def setup_renderer(args):
    """Setup the Taichi renderer."""
    TaichiQueue.init(ti.gpu, offline_cache=True, debug=getattr(args, 'debug', False), device_memory_GB=0.1)
    renderer_args = make_renderer_args(args)
    return renderer_from_args(renderer_args)


def get_next_frame_number(output_dir: Path) -> int:
    """Find the next available frame number by checking existing files."""
    if not output_dir.exists():
        return 0
    
    pattern = re.compile(r'frame_(\d+)\.jpg$')
    frame_numbers = [
        int(match.group(1))
        for f in output_dir.glob("frame_*.jpg")
        if (match := pattern.match(f.name))
    ]
    
    return max(frame_numbers) + 1 if frame_numbers else 0


def create_output_directory(output_dir: Path) -> Path:
    """Create output directory for rendered frames."""
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def render_frame(workspace_renderer: WorkspaceRenderer,
                 camera: FOVCamera,
                 settings: Settings,
                 frame_number: int,
                 output_path: Path) -> str:
    """Render a single frame and save it to disk as high-quality JPEG."""
    image = workspace_renderer.render(camera, settings)
    filename = output_path / f"frame_{frame_number:06d}.jpg"

    success = cv2.imwrite(str(filename), cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])

    if not success:
        raise RuntimeError(f"Failed to save frame {frame_number} to {filename}")

    return str(filename)


def main():
    parser = argparse.ArgumentParser(
        description="Render frames from a splat workspace using a saved camera path")

    # Input arguments
    parser.add_argument('workspace_path', help="Path to workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")
    parser.add_argument('--model', default=None, help="Load model from point_clouds folder, default is latest iteration")
    parser.add_argument('--camera-path', default=None, help="Path to camera path file (default: workspace_path/camera_path.npy)")

    # Output arguments  
    parser.add_argument('-o', '--output-dir', required=True, help="Output directory for rendered frames")

    # Animation arguments
    parser.add_argument('--image-size', nargs=2, type=int, default=[1920, 1080],
                       help="Output image size as width height")
    parser.add_argument('--frame-rate', type=float, default=30.0, help="Frame rate for animation")
    parser.add_argument('--loop', action='store_true', help="Loop the camera path animation")

    parser.add_argument('--device', default='cuda:0', type=torch.device, help="Device to use for rendering")

    # Settings and renderer arguments
    add_settings_arguments(parser)
    add_render_arguments(parser)
    parser.add_argument('--debug', action='store_true', help="Enable taichi kernels in debug mode")

    args = parser.parse_args()

    torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)

    print(f"Loading workspace from {args.workspace_path}")
    workspace = Workspace.load(args.workspace_path)

    if args.model is None:
        args.model = workspace.latest_iteration()

    print(f"Loading model {args.model}")
    gaussians = workspace.load_model(args.model)
    print(f"Loaded model: {gaussians}")

    gaussians = gaussians.to(args.device)

    # Determine camera path location
    if args.camera_path:
        camera_path_file = Path(args.camera_path)
        if not camera_path_file.exists():
            print(f"Camera path file not found: {camera_path_file}")
            sys.exit(1)
        keypoints = list(np.load(camera_path_file))
        print(f"Loaded {len(keypoints)} keypoints from {camera_path_file}")
    else:
        keypoints = read_camera_path(Path(args.workspace_path))
        if not keypoints:
            print("No camera path found. Please save a camera path in the viewer first.")
            sys.exit(1)

    print(f"Found {len(keypoints)} camera poses in path")

    print("Setting up renderer...")
    gaussian_renderer = setup_renderer(args)
    workspace_renderer = WorkspaceRenderer(workspace, gaussians, gaussian_renderer)

    settings = create_settings_from_args(args)
    output_path = create_output_directory(Path(args.output_dir))
    start_frame = get_next_frame_number(output_path)
    
    if start_frame > 0:
        print(f"Output directory: {output_path} (appending from frame {start_frame})")
    else:
        print(f"Output directory: {output_path}")

    animation_config = AnimationConfig(
        animate_speed=args.animate_speed,
        animate_pausing=args.animate_pausing
    )
    animator = CameraPathAnimator(
        keypoints,
        loop=args.loop,
        config=animation_config
    )

    scene_camera = SceneCamera()
    if workspace.cameras:
        scene_camera.set_camera(workspace.cameras[0])

    dt = 1.0 / args.frame_rate
    
    # Calculate animation duration based on path length
    animation_duration = animator.total / args.animate_speed
    
    if args.loop:
        total_frames = int(animation_duration * args.frame_rate)
        print(f"Rendering {total_frames} frames for one complete loop cycle")
        print(f"Loop duration: {animation_duration:.1f}s at speed {args.animate_speed}")
    else:
        # For non-looping, render until animation finishes naturally
        print(f"Rendering animation (estimated {animation_duration:.1f}s at speed {args.animate_speed})")

    frame_idx = 0
    while True:
        frame_number = start_frame + frame_idx
        
        r, t, finished = animator.get_camera_pose(dt)
        scene_camera.set_pose(r, t)
        camera = scene_camera.resized(tuple(args.image_size))

        render_frame(
            workspace_renderer, camera, settings,
            frame_number, output_path
        )
        
        frame_idx += 1
        
        if args.loop:
            print(f"Rendering frame {frame_number + 1} ({frame_idx}/{total_frames})", end='\r')
            if frame_idx >= total_frames:
                break
        else:
            print(f"Rendering frame {frame_number + 1}", end='\r')
            # For non-looping, stop when animation finishes
            if finished:
                break

    print(f"\nRendering complete! Frames saved to {output_path}")


if __name__ == "__main__":
    main()