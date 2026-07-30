"""SBM Alignment GUI — main entry point."""
import sys

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import QApplication, QMainWindow, QTabWidget, QStatusBar

from ._setup_tab import SetupTab
from ._energy_tab import EnergyTab
from ._align_tab import AlignTab


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBM Alignment")
        self.resize(1300, 820)

        self._settings = QSettings("ID15A", "align-sbm")

        self._setup_tab = SetupTab(self._settings)
        self._energy_tab = EnergyTab(self._settings)
        self._align_tab = AlignTab(self._setup_tab, self._energy_tab)

        tabs = QTabWidget()
        tabs.addTab(self._setup_tab, "Setup")
        tabs.addTab(self._energy_tab, "Energy Table")
        tabs.addTab(self._align_tab, "Alignment")
        tabs.setCurrentIndex(2)
        self.setCentralWidget(tabs)

        sb = QStatusBar()
        self.setStatusBar(sb)
        self._align_tab.status_message.connect(sb.showMessage)

        self._build_menu()

    def _build_menu(self):
        mb = self.menuBar()

        file_menu = mb.addMenu("&File")

        save_act = QAction("&Save Config", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.triggered.connect(self._save_all)
        file_menu.addAction(save_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

    def _save_all(self):
        self._setup_tab.save_settings()
        self._energy_tab.save_settings()

    def closeEvent(self, event):
        self._save_all()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("align-sbm")
    app.setOrganizationName("ID15A")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
