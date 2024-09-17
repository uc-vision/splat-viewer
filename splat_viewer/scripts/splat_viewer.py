import argparse
import sys

from PySide6.QtWidgets import QApplication


from splat_viewer.gaussians.workspace import load_workspace
from splat_viewer.renderer.arguments import  add_render_arguments, make_renderer_args, renderer_from_args

from splat_viewer.viewer.main_window import create_window

import signal
import taichi as ti
import torch

from splat_viewer.viewer.scene_widget import SceneWidget
from splat_viewer.viewer.settings import Settings

def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path', help="workspace folder containing cameras.json, input.ply and point_cloud folder with .ply models")  # positional argument
    parser.add_argument('--model', default=None, help="load model from point_clouds folder, default is latest iteration") 
    parser.add_argument('--device', default='cuda:0', help="torch device to use")
    parser.add_argument('--debug', action='store_true', help="enable taichi kernels in debug mode")

    add_render_arguments(parser)    

    parsed_args, unparsed_args = parser.parse_known_args()
    return parsed_args, unparsed_args

def sigint_handler(*args):
    QApplication.quit()



def main():
    signal.signal(signal.SIGINT, sigint_handler)
    torch.set_printoptions(precision=5, sci_mode=False, linewidth=120)


    parsed_args, unparsed_args = process_cl_args()
    workspace = load_workspace(parsed_args.model_path)

    if parsed_args.model is None:
      parsed_args.model = workspace.latest_iteration()

    gaussians = workspace.load_model(parsed_args.model)
    print(f"Loaded model {parsed_args.model}: {gaussians}")

    ti.init(ti.gpu, offline_cache=True, debug=parsed_args.debug, device_memory_GB=0.1)


    qt_args = sys.argv[:1] + unparsed_args
    app = QApplication(qt_args)
  
    renderer = renderer_from_args(make_renderer_args(parsed_args))
    print(renderer)
    
    scene_widget = SceneWidget(
       settings=Settings(device=parsed_args.device),
       renderer = renderer
    )

    window = create_window(scene_widget)
    
    scene_widget.load_workspace(workspace, gaussians)
    window.show()
    
    sys.exit(app.exec())

