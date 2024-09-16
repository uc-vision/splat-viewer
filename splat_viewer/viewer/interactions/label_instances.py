import math
from typing import Callable, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
import numpy as np
import torch

from splat_viewer.gaussians.data_types import Gaussians
from splat_viewer.viewer.interaction import Interaction

from .scribble import ScribbleGeometric


class LabelInstances(Interaction):
  def __init__(self):
    super(LabelInstances, self).__init__()

    self.instances = []
    self.new_scribble()


  def new_scribble(self):
    self.pop()

    color = np.random.rand(3).tolist()
    self.scribble = ScribbleGeometric(color=color)

    self.push(self.scribble)

  def on_activate(self):
    self.new_scribble()
    
  
  def keyPressEvent(self, event: QtGui.QKeyEvent):

    if event.key() == Qt.Key.Key_Return and event.modifiers() & Qt.KeyboardModifier.ControlModifier:

      if self.scribble.valid:
        self.instances.append(self.scribble.instance)

      self.new_scribble()
      return True

    return super().keyPressEvent(event)
