from dataclasses import replace
from pathlib import Path

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon

import qtawesome as qta

from splat_viewer.viewer.scene_widget import SceneWidget, Settings
from splat_viewer.viewer.settings import ViewMode


def create_window(scene_widget:SceneWidget):
  ui_path = Path(__file__).parent.parent / "ui"

  loader = QUiLoader()
  window = loader.load(ui_path / "main_window.ui")

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
  window.actionEdit_Foreground.setIcon(QIcon(qta.icon('mdi.eraser')))


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


  scene_widget.setFocus()

  return window



  