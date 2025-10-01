from pathlib import Path
from typing import Optional
from PySide6.QtUiTools import QUiLoader
from PySide6 import QtWidgets

def load_ui(path:Path, parent:Optional[QtWidgets.QWidget] = None):
  ui_path = Path(__file__).parent.parent / "ui"

  loader = QUiLoader()
  return loader.load(ui_path / path, parent)

