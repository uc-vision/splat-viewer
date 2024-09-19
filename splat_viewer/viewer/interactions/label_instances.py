from dataclasses import replace
from typing import List

from PySide6 import QtGui
from PySide6.QtCore import Qt
import numpy as np

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.interaction import Interaction

from .select_radius import Instance, SelectRadius


def colorize_instances(instances:List[Instance], gaussians: Gaussians):
  for instance in instances:
    gaussians = gaussians.set_colors(instance.color, instance.points)
    
  return gaussians


class LabelInstances(Interaction):
  def __init__(self):
    super(LabelInstances, self).__init__()

    self.instances:List[Instance] = []
    self.new_scribble()


  def new_scribble(self):
    self.pop()

    self.scribble = SelectRadius()   

    self.push(self.scribble)

  def on_activate(self):
    self.new_scribble()




  def renderEvent(self, gaussians:Gaussians):
    return colorize_instances(self.instances, gaussians)

  def keyPressEvent(self, event: QtGui.QKeyEvent):

    if event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ControlModifier:

      if self.scribble.valid:
        instance = self.scribble.instance

        color = np.random.rand(3).tolist()
        self.instances.append(replace(instance, color=color))

      self.new_scribble()

      self.mark_dirty()
      return True

    return super().keyPressEvent(event)
