from dataclasses import replace
from functools import cached_property
from pathlib import Path
from typing import List, Optional, Set

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon

from immutables import Map
import qtawesome as qta

from splat_viewer.editor.editor import EditorState
from splat_viewer.editor.gaussian_scene import GaussianScene, Instance
from splat_viewer.editor.util import load_ui
from splat_viewer.editor.interactions.draw_instances import InstanceEditor
from splat_viewer.renderer.taichi_splatting import GaussianRenderer
from splat_viewer.viewer.renderer import WorkspaceRenderer
from splat_viewer.viewer.scene_widget import SceneWidget, Settings
from splat_viewer.viewer.settings import ViewMode

from PySide6.QtGui import QColor
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QAbstractItemView, QListWidget

from PySide6.QtCore import QObject

# def load_workspace(self, workspace:Workspace, gaussians:Gaussians):

#   scene = GaussianScene.from_gaussians(gaussians.to(self.settings.device), class_labels=["apple"])
#   editor.set_scene(scene)

#   self.scene_renderer = WorkspaceRenderer(workspace, self.renderer, self.settings.device)
#   self.keypoints = self.read_keypoints()

#   self.set_camera_index(0)

#   self.camera_state = FlyControl()
#   self.editor.scene_changed.connect(self.emit_scene_changed)


class EditorApp(QObject):
  def __init__(self, parent:Optional[QObject]=None, settings:Settings=Settings()):
    super().__init__(parent)

    window = load_ui(Path("main_window.ui"))
    self.window = window


    self.renderer = GaussianRenderer()
    self.scene_renderer = WorkspaceRenderer(self.renderer, settings.device)
    self.scene_widget = SceneWidget(self.scene_renderer, settings, parent=self.window)

    self.window.setCentralWidget(self.scene_widget)
    self.editor = EditorState(GaussianScene.empty(device=self.settings.device))

    self.setup_icons()
    self.setup_settings()
    self.setup_instances()



  @property
  def settings(self) -> Settings:
    return self.scene_widget.settings
  
  @property
  def scene(self) -> GaussianScene:
    return self.editor.scene

  def setup_actions(self):
    window = self.window
      
    window.setWindowIcon(QIcon(qta.icon('mdi.cube-scan')))
    window.actionDraw.setIcon(QIcon(qta.icon('mdi.draw')))
    window.actionEraser.setIcon(QIcon(qta.icon('mdi.eraser')))
    window.actionSAM.setIcon(QIcon(qta.icon('mdi.label')))

    window.actionUndo.setIcon(QIcon(qta.icon('mdi.undo')))
    window.actionRedo.setIcon(QIcon(qta.icon('mdi.redo')))
    window.actionDelete.setIcon(QIcon(qta.icon('mdi.delete')))

      
    window.actionUndo.triggered.connect(self.editor.undo)
    window.actionRedo.triggered.connect(self.editor.redo)

  @cached_property
  def checkboxes(self):
    window = self.window
    return dict(
        initial_points = window.show_initial_points,
        cameras = window.show_cameras,
        cropped = window.show_cropped,
        bounding_boxes = window.show_bounding_boxes,
        filtered_points = window.show_filtered_points,
        color_instances = window.show_color_instances,

        fixed_size = window.fixed_size,
        fixed_opacity = window.fixed_opacity,
    )

  
  def update_setting(self, key, value):
    self.scene_widget.update_setting(key, value)

  def on_settings_changed(self, settings:Settings):
    for key, cb in self.checkboxes.items():
        checked = getattr(settings.show, key)
        if checked != cb.isChecked():
            cb.setChecked(checked)

    if settings.view_mode.name != self.window.view_mode.currentText():
      self.window.view_mode.setCurrentText(settings.view_mode.name)
      

  def show(self, key:str, checked:bool):
    show = replace(self.scene_widget.settings.show, **{key: checked})
    self.scene_widget.update_setting(show=show)
                    
  def on_checked(self, key):
    return lambda checked: self.show(key, checked)
  
  def on_view_mode_changed(self, index):
    mode = ViewMode[self.window.view_mode.currentText()]
    self.scene_widget.update_setting(view_mode=mode)

  def setup_settings(self):
    for key, value in self.checkboxes.items():
      # Need to use the funciton otherwise 'key' is not captured properly
      value.stateChanged.connect(self.on_checked(key))

    self.scene_widget.settings_changed.connect(self.on_settings_changed)
    for mode in ViewMode:
      self.window.view_mode.addItem(mode.name)

    self.window.view_mode.currentIndexChanged.connect(self.on_view_mode_changed)



  def update_class_labels(self):
    window = self.window
    window.class_combo.clear()
    for label in self.scene.class_labels:
      window.class_combo.addItem(label)

  def update_selected_instance(self):
    inst = self.scene.selected_instance
    selected_ids = self.scene.selected_instance_ids


    if len(selected_ids) == 1:
      self.window.class_combo.setCurrentText(inst.label)
      self.window.name_edit.setText(inst.name)  
      self.window.name_edit.setEnabled(True)
    elif len(selected_ids) > 1:
      self.window.name_edit.setText("(multiple selected)")
      self.window.name_edit.setDisabled(True)

    inst_list = self.window.instances_list
    if len(selected_ids) > 0:
      for i in self.scene.selected_instance_ids:
        item = self.find_row(inst_list, i)
        if item is not None:
          item.setSelected(True)
          inst_list.setCurrentItem(item)      

  def update_instance_list(self):
    inst_list = self.window.instances_list
    inst_list.clear()

    for k, instance in self.scene.instances.items():
        # Create a color icon
        color_pixmap = QPixmap(20, 20)
        color_pixmap.fill(QColor(*[int(c * 255) for c in instance.color]))
        color_icon = QIcon(color_pixmap)

        # Create a list item with icon and text
        item = QListWidgetItem(color_icon, instance.name)
        
        # Set the instance id as item data for later reference
        item.setData(Qt.UserRole, k)
        inst_list.addItem(item)

    inst_list.setSelectionMode(QAbstractItemView.SingleSelection)
    
  def find_row(inst_list:QListWidget, instance_id:int):
    for i in range(inst_list.count()):
      item = inst_list.item(i)
      if item.data(Qt.UserRole) == instance_id:
        return item
    return None


  def scene_changed(self, previous:GaussianScene, current:GaussianScene):
    if current.class_labels != previous.class_labels:
      self.update_class_labels(current.class_labels)


    if previous.selected_instance_ids != current.selected_instance_ids:
      self.update_selected_instance(current)

    if previous.instances.keys() != current.instances.keys():
      self.update_instance_list(current)

    has_selection = len(current.selected_instance_ids) > 0
    self.window.actionDelete.setEnabled(has_selection)
    
    self.window.actionUndo.setEnabled(self.editor.can_undo)
    self.window.actionRedo.setEnabled(self.editor.can_redo)


      
  def on_instance_clicked(self, item:Optional[QListWidgetItem]):
    if item is None:
      self.editor.modify_scene(self.editor.scene.with_unselected())
    else:
        instance_id = item.data(Qt.UserRole)
        self.editor.modify_scene(self.editor.scene.with_selected({instance_id}))  


  def setup_instances(self):
    window = self.window
    window.instances_list.itemClicked.connect(self.on_instance_changed)

    self.update_instance_list()
    self.update_selected_instance()



  



  
