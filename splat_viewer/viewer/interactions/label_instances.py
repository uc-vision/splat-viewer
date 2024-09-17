from dataclasses import replace
from typing import List

from PySide6 import QtGui
from PySide6.QtCore import Qt
import numpy as np

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.interaction import Interaction

from .scribble import Instance, ScribbleGeometric


class LabelInstances(Interaction):
  def __init__(self):
    super(LabelInstances, self).__init__()

    self.instances:List[Instance] = []
    self.new_scribble()


  def new_scribble(self):
    self.pop()

    self.scribble = ScribbleGeometric()

    self.push(self.scribble)

  def on_activate(self):
    self.new_scribble()


  
  def colorize_instances(self, gaussians: Gaussians):
    for instance in self.instances:
      gaussians = gaussians.set_colors(instance.color, instance.points)
      
    return gaussians



  def keyPressEvent(self, event: QtGui.QKeyEvent):

    if event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ControlModifier:

      if self.scribble.valid:
        instance = self.scribble.instance

        color = np.random.rand(3).tolist()
        self.instances.append(replace(instance, color=color))

      self.new_scribble()

      self.update_gaussians(self.colorize_instances(self.gaussians))
      return True

    return super().keyPressEvent(event)
