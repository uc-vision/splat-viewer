from dataclasses import replace
from pathlib import Path

from PySide6.QtUiTools import QUiLoader
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QMainWindow

import qtawesome as qta

from splat_viewer.viewer.scene_widget import SceneWidget, Settings


class MainWindow(QMainWindow):

    @staticmethod
    def load(scene_widget:SceneWidget):
        ui_path = Path(__file__).parent.parent / "ui"

        loader = QUiLoader()
        window = loader.load(ui_path / "main_window.ui")

        window.initialise(scene_widget)

        return window

    def settings_changed(self, settings:Settings):
        for key, cb in self.show_checkboxes.items():
            checked = getattr(settings.show, key)
            if checked != cb.isChecked():
                cb.setChecked(checked)

    def initialise(self, scene_widget:SceneWidget):
        

        self.setWindowIcon(QIcon(qta.icon('mdi.cube-scan')))
        self.actionInstances.setIcon(QIcon(qta.icon('mdi.draw')))
        self.actionEdit_Foreground.setIcon(QIcon(qta.icon('mdi.eraser')))

        self.show_checkboxes = dict(
            initial_points = self.show_initial_points,
            cameras = self.show_cameras,
            cropped = self.show_cropped,
            bounding_boxes = self.show_bounding_boxes,
            filtered_points = self.show_filtered_points,
            color_instances = self.show_color_instances
        )

        def show(key:str, checked:bool):
            show = replace(self.scene_widget.settings.show, **{key: checked})
            self.scene_widget.update_setting(show=show)
                        

        for key, value in self.show_checkboxes.items():
            value.stateChanged.connect(lambda checked: show(key, checked))

        scene_widget.settings_changed.connect(self.settings_changed)

        self.setCentralWidget(scene_widget)

    def update_setting(self, key, value):
        self.scene_widget.update_setting(key, value)

    