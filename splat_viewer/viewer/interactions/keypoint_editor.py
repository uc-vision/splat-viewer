import math
from typing import Callable, Tuple

from PySide6 import QtGui, QtCore
from PySide6.QtCore import Qt
import numpy as np
import torch

from splat_viewer.viewer.interaction import Interaction
from splat_viewer.viewer.interactions.animate import animate_to_loop



class KeyPointEditor(Interaction):
  def __init__(self):
    super(KeyPointEditor, self).__init__()


  def keyPressEvent(self, event: QtGui.QKeyEvent):
    scene = self.scene_widget

    if event.key() == Qt.Key_Space and event.modifiers() & Qt.ControlModifier:  
      scene.write_keypoints()

    elif event.key() == Qt.Key_Space:
      scene.keypoints.append(scene.camera.view_matrix)

    elif event.key() == Qt.Key_Return & Qt.ControlModifier:
      if len(self.keypoints) > 0:
        animate_to_loop(scene.camera_state, 
                        scene.camera.view_matrix, self.keypoints)
    
    return super().keyPressEvent(event)




