import sys
import argparse

from PySide6 import  QtWidgets
from PySide6.QtWidgets import QApplication

from splat_viewer.gaussians.workspace import Workspace, load_workspace
from splat_viewer.gaussians.gaussians import Gaussians

from splat_viewer.scene_widget import SceneWidget, Settings

import signal
import taichi as ti
import torch


def show_workspace(workspace:Workspace, gaussians:Gaussians = None):
  import taichi as ti
  ti.init(ti.gpu, offline_cache=False, device_memory_GB=0.1)

  from splat_viewer.viewer.main import sigint_handler
  signal.signal(signal.SIGINT, sigint_handler)

  app = QtWidgets.QApplication(["viewer"])
  widget = SceneWidget()

  if gaussians is None:
    gaussians = workspace.load_model(workspace.latest_iteration())

  print(f"Showing model from {workspace.model_path}: {gaussians}")


  widget.load_workspace(workspace, gaussians)
  widget.show()
  app.exec_()



def process_cl_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_path')  # positional argument
    parser.add_argument('--model', default=None)  # positional argument
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--debug', action='store_true')
    

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
    

    window = QtWidgets.QMainWindow()

    scene_widget = SceneWidget(settings=Settings(device=parsed_args.device))
    scene_widget.load_workspace(workspace, gaussians)

    window.setCentralWidget(scene_widget)

    window.show()
    sys.exit(app.exec())

  
if __name__ == '__main__':
  main()