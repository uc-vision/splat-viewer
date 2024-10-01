from dataclasses import replace
from pathlib import Path
from typing import Optional

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon

import qtawesome as qta

from splat_viewer.editor.gaussian_scene import GaussianScene, Instance
from splat_viewer.editor.util import load_ui
from splat_viewer.viewer.interactions.instance_editor import InstanceEditor
from splat_viewer.viewer.scene_widget import SceneWidget, Settings
from splat_viewer.viewer.settings import ViewMode

from PySide6.QtGui import QColor
from PySide6.QtGui import QPixmap
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QListWidgetItem, QAbstractItemView, QListWidget
from PySide6.QtGui import QIcon

from PySide6.QtGui import QKeySequence


def create_window(scene_widget:SceneWidget):
  window = load_ui(Path("main_window.ui"))

  show_checkboxes = dict(
      initial_points = window.show_initial_points,
      cameras = window.show_cameras,
      cropped = window.show_cropped,
      bounding_boxes = window.show_bounding_boxes,
      filtered_points = window.show_filtered_points,
      color_instances = window.show_color_instances,

      fixed_size = window.fixed_size,
      fixed_opacity = window.fixed_opacity,
  )

  def update_setting(key, value):
      scene_widget.update_setting(key, value)

  def on_settings_changed(settings:Settings):
    for key, cb in show_checkboxes.items():
        checked = getattr(settings.show, key)
        if checked != cb.isChecked():
            cb.setChecked(checked)

    if settings.view_mode.name != window.view_mode.currentText():
      window.view_mode.setCurrentText(settings.view_mode.name)
      

  window.setWindowIcon(QIcon(qta.icon('mdi.cube-scan')))
  window.actionInstances.setIcon(QIcon(qta.icon('mdi.draw')))
  window.actionErase_labels.setIcon(QIcon(qta.icon('mdi.eraser')))
  window.actionUndo.setIcon(QIcon(qta.icon('mdi.undo')))
  window.actionRedo.setIcon(QIcon(qta.icon('mdi.redo')))
  window.actionDelete.setIcon(QIcon(qta.icon('mdi.delete')))


  def show(key:str, checked:bool):
    show = replace(scene_widget.settings.show, **{key: checked})
    scene_widget.update_setting(show=show)
                    
  def on_checked(key):
      return lambda checked: show(key, checked)
  
  for key, value in show_checkboxes.items():
    # Need to use the funciton otherwise 'key' is not captured properly
    value.stateChanged.connect(on_checked(key))

  scene_widget.settings_changed.connect(on_settings_changed)


  for mode in ViewMode:
    window.view_mode.addItem(mode.name)

  def on_view_mode_changed(index):
    mode = ViewMode[window.view_mode.currentText()]
    scene_widget.update_setting(view_mode=mode)

  window.view_mode.currentIndexChanged.connect(on_view_mode_changed)
  window.setCentralWidget(scene_widget)

  scene_widget.tool = InstanceEditor()
  editor = scene_widget.editor

  def set_class_labels(labels:list[str]):
    selected = window.class_combo.currentText()
    window.class_combo.clear()
    for label in labels:
      window.class_combo.addItem(label)

    print(selected)
    window.class_combo.setCurrentText(selected)

  def set_instances(scene:GaussianScene):
    inst_list = window.instances_list
    inst_list.clear()

    for k, instance in scene.instances.items():
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


  def update_instaces(previous:GaussianScene, current:GaussianScene):
    if current.class_labels != previous.class_labels:
      set_class_labels(current.class_labels)

    if previous.instances.keys() != current.instances.keys():
      set_instances(current)
    
    inst_list = window.instances_list
    has_selection = len(current.selected_instances) > 0
    if has_selection:

      for i in current.selected_instances:
        item = find_row(inst_list, i)
        if item is not None:
          item.setSelected(True)
          inst_list.setCurrentItem(item)


    window.actionDelete.setEnabled(has_selection)
    
    window.actionUndo.setEnabled(editor.can_undo)
    window.actionRedo.setEnabled(editor.can_redo)

    
    
  def on_instance_changed(item:Optional[QListWidgetItem]):
    if item is None:
      editor.modify_scene(editor.scene.with_unselected())
    else:
      instance_id = item.data(Qt.UserRole)
      editor.modify_scene(editor.scene.with_selected({instance_id}))


  window.instances_list.itemClicked.connect(on_instance_changed)
  

  window.actionUndo.triggered.connect(editor.undo)
  window.actionRedo.triggered.connect(editor.redo)


  scene_widget.editor.scene_changed.connect(update_instaces)
  scene_widget.setFocus()

  return window



  