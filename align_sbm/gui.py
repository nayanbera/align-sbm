"""SBM Alignment GUI — main entry point."""
import sys

import traceback

from PyQt6.QtCore import QSettings
from PyQt6.QtGui import QAction, QKeySequence
from PyQt6.QtWidgets import (
    QApplication, QFileDialog, QMainWindow, QMessageBox, QTabWidget, QStatusBar,
)

from ._setup_tab import SetupTab
from ._energy_tab import EnergyTab
from ._align_tab import AlignTab
from ._help_dialog import DocsDialog, AboutDialog


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SBM Alignment")
        self.resize(1300, 820)

        self._settings = QSettings("ID15A", "align-sbm")

        self._setup_tab = SetupTab(self._settings)
        self._energy_tab = EnergyTab(self._settings)
        self._align_tab = AlignTab(self._setup_tab, self._energy_tab, settings=self._settings)

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

        # ── File ────────────────────────────────────────────────────────────
        file_menu = mb.addMenu("&File")

        save_act = QAction("&Save Config", self)
        save_act.setShortcut(QKeySequence.StandardKey.Save)
        save_act.setStatusTip("Save current PV names and scan parameters")
        save_act.triggered.connect(self._save_all)
        file_menu.addAction(save_act)

        save_as_act = QAction("Save Config &As…", self)
        save_as_act.setStatusTip("Export configuration to an .ini file for sharing")
        save_as_act.triggered.connect(self._save_config_as)
        file_menu.addAction(save_as_act)

        load_cfg_act = QAction("&Load Config…", self)
        load_cfg_act.setStatusTip("Import configuration from an .ini file")
        load_cfg_act.triggered.connect(self._load_config_from)
        file_menu.addAction(load_cfg_act)

        file_menu.addSeparator()

        load_et_act = QAction("&Load Energy Table…", self)
        load_et_act.setStatusTip("Load energy table rows from a CSV file")
        load_et_act.triggered.connect(self._energy_tab._load_csv)
        file_menu.addAction(load_et_act)

        save_et_act = QAction("Save &Energy Table…", self)
        save_et_act.setStatusTip("Save the current energy table to a CSV file")
        save_et_act.triggered.connect(self._energy_tab._save_csv)
        file_menu.addAction(save_et_act)

        file_menu.addSeparator()

        quit_act = QAction("&Quit", self)
        quit_act.setShortcut(QKeySequence.StandardKey.Quit)
        quit_act.setStatusTip("Exit the application")
        quit_act.triggered.connect(self.close)
        file_menu.addAction(quit_act)

        # ── Help ─────────────────────────────────────────────────────────────
        help_menu = mb.addMenu("&Help")

        docs_act = QAction("&Documentation", self)
        docs_act.setShortcut(QKeySequence.StandardKey.HelpContents)
        docs_act.setStatusTip("Open the built-in documentation")
        docs_act.triggered.connect(self._show_docs)
        help_menu.addAction(docs_act)

        help_menu.addSeparator()

        about_act = QAction("&About align-sbm", self)
        about_act.setStatusTip("About this application")
        about_act.triggered.connect(self._show_about)
        help_menu.addAction(about_act)

    def _save_config_as(self):
        self._save_all()   # flush current UI state into QSettings first
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Config As", "align_sbm_config.ini",
            "Config files (*.ini);;All files (*)")
        if not path:
            return
        if not path.endswith(".ini"):
            path += ".ini"
        file_cfg = QSettings(path, QSettings.Format.IniFormat)
        for key in self._settings.allKeys():
            file_cfg.setValue(key, self._settings.value(key))
        file_cfg.sync()
        self.statusBar().showMessage(f"Config saved to {path}", 5000)

    def _load_config_from(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Config", "",
            "Config files (*.ini);;All files (*)")
        if not path:
            return
        file_cfg = QSettings(path, QSettings.Format.IniFormat)
        if not file_cfg.allKeys():
            QMessageBox.warning(self, "Load Config",
                                "No settings found in the selected file.")
            return
        for key in file_cfg.allKeys():
            self._settings.setValue(key, file_cfg.value(key))
        # Refresh all tabs from the updated QSettings
        self._setup_tab.reload_settings()
        self._energy_tab.reload_settings()
        self._align_tab.reload_settings()
        self.statusBar().showMessage(f"Config loaded from {path}", 5000)

    def _show_docs(self):
        try:
            dlg = DocsDialog(self)
            dlg.exec()
        except Exception:
            tb = traceback.format_exc()
            print(tb)
            QMessageBox.critical(self, "Documentation error",
                                 f"Could not open documentation:\n\n{tb}")

    def _show_about(self):
        dlg = AboutDialog(self)
        dlg.exec()

    def _save_all(self):
        self._setup_tab.save_settings()
        self._energy_tab.save_settings()
        self._align_tab.save_settings()

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
