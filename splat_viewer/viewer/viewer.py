

import signal

from PySide6 import QtWidgets
from PySide6 import QtCore
from torch.multiprocessing import Process, Queue, get_start_method

from splat_viewer.gaussians import Gaussians
from splat_viewer.gaussians.workspace import Workspace
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.viewer.scene_widget import SceneWidget, Settings


def init_viewer(workspace:Workspace, 
                   gaussians:Gaussians = None, 
                   settings:Settings=Settings()):

  app = QtWidgets.QApplication.instance()
  if app is None:
    app = QtWidgets.QApplication(["viewer"])
    

  widget = SceneWidget(settings=settings, 
                       renderer = GaussianRenderer())

  if gaussians is None:
    gaussians = workspace.load_model(workspace.latest_iteration())
  widget.load_workspace(workspace, gaussians)

  window = QtWidgets.QMainWindow()
  window.setCentralWidget(widget)
  window.show()

  # widget.show()
  return app, window, widget

def show_workspace(workspace:Workspace, 
                   gaussians:Gaussians = None, 
                   settings:Settings=Settings()):
  from splat_viewer.viewer.viewer import sigint_handler
  signal.signal(signal.SIGINT, sigint_handler)

  print(f"Showing model from {workspace.model_path}: {gaussians}")
  app, _, _ = init_viewer(workspace, gaussians, settings)
  app.exec()


def sigint_handler(*args):
    QtWidgets.QApplication.quit()


def run_process(workspace:Workspace, 
                   update_queue:Queue,

                   gaussians:Gaussians = None, 
                   settings:Settings=Settings()):
  
  import taichi as ti
  ti.init(ti.gpu, offline_cache=True, device_memory_GB=0.1)

  from splat_viewer.viewer.viewer import sigint_handler
  signal.signal(signal.SIGINT, sigint_handler)

  app, _, widget = init_viewer(workspace, gaussians, settings)


  def on_timer(): 
    if not update_queue.empty():
      update = update_queue.get()

      if update is None:
        app.quit()
        return

      if isinstance(update, Gaussians):
        widget.update_gaussians(update)
      elif isinstance(update, dict):
        widget.update_gaussians(widget.gaussians.replace(**update))
      else:
        raise TypeError(f"Unknown type of update: {type(update)}")

  timer = QtCore.QTimer(widget)
  timer.timeout.connect(on_timer)

  timer.start(10)

  app.exec()


class Viewer:
  def __init__(self):
    pass

  def quit(self):
    pass

  def update_gaussians(self, gaussians:Gaussians):
    pass

  def __enter__(self):
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    pass

  def start(self):
    pass

  def close(self):
    pass

class ViewerProcess(Viewer):
  def __init__(self, workspace:Workspace, 
               gaussians:Gaussians, 
               settings:Settings, 
               queue_size=1):
    
    assert get_start_method() == "spawn", "For ViewerProcess, torch multiprocessing must be started with spawn"
    self.update_queue = Queue(queue_size)
    self.view_process = Process(target=run_process, 
                           args=(workspace, self.update_queue, gaussians, settings))

    
  def quit(self):
    self.update_queue.put(None)
    self.join()

  def update_gaussians(self, gaussians:Gaussians):
    self.update_queue.put(gaussians)

  def __enter__(self):
    self.start()
    return self
  
  def __exit__(self, exc_type, exc_value, traceback):
    self.join()

  def start(self):
    self.view_process.start()

  def join(self):
    while not self.update_queue.empty():
      pass

    self.view_process.join()
    self.update_queue.close()



