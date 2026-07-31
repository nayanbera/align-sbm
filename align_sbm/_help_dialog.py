"""Documentation and About dialogs for the Help menu."""
import importlib.resources
import pathlib

from PyQt6.QtCore import Qt, QUrl
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QDialogButtonBox,
    QTextBrowser, QLabel, QSizePolicy,
)

_README_CSS = """
body  { font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
        font-size: 14px; line-height: 1.6; color: #212121;
        margin: 24px 32px; }
h1    { font-size: 1.9em; border-bottom: 2px solid #1565c0;
        padding-bottom: 6px; color: #1565c0; }
h2    { font-size: 1.35em; border-bottom: 1px solid #bbb;
        padding-bottom: 4px; margin-top: 28px; color: #263238; }
h3    { font-size: 1.1em; margin-top: 20px; color: #37474f; }
code  { background: #f5f5f5; border-radius: 3px;
        padding: 1px 5px; font-family: "Menlo", "Consolas", monospace;
        font-size: 0.9em; color: #c62828; }
pre   { background: #f5f5f5; border-left: 4px solid #1565c0;
        padding: 12px 16px; border-radius: 4px; overflow-x: auto;
        font-family: "Menlo", "Consolas", monospace; font-size: 0.88em;
        line-height: 1.5; }
pre code { background: none; padding: 0; color: inherit; }
table { border-collapse: collapse; width: 100%; margin: 12px 0; }
th    { background: #e3f2fd; color: #1565c0; font-weight: 600;
        text-align: left; padding: 8px 12px;
        border: 1px solid #90caf9; }
td    { padding: 6px 12px; border: 1px solid #ccc; }
tr:nth-child(even) td { background: #fafafa; }
a     { color: #1565c0; text-decoration: none; }
a:hover { text-decoration: underline; }
blockquote { border-left: 4px solid #90caf9; margin: 12px 0;
             padding: 4px 16px; color: #546e7a; }
hr    { border: none; border-top: 1px solid #e0e0e0; margin: 20px 0; }
"""


def _read_readme() -> str:
    """Return the content of README.md as a string."""
    # Try the package directory first, then the repo root
    here = pathlib.Path(__file__).parent
    for candidate in [here.parent / "README.md", here / "README.md"]:
        if candidate.exists():
            return candidate.read_text(encoding="utf-8")
    return "README.md not found."


def _render_markdown(text: str) -> str:
    try:
        import markdown
        return markdown.markdown(
            text,
            extensions=["tables", "fenced_code", "toc"],
        )
    except Exception:
        import html
        return f"<pre>{html.escape(text)}</pre>"


class DocsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SBM Alignment — Documentation")
        self.resize(920, 720)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)

        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)

        md_text = _read_readme()
        body_html = _render_markdown(md_text)
        full_html = (
            "<html><head>"
            '<meta charset="utf-8">'
            f"<style>{_README_CSS}</style>"
            f"</head><body>{body_html}</body></html>"
        )
        try:
            browser.setHtml(full_html)
        except Exception:
            import html as _html
            browser.setPlainText(_html.unescape(md_text))
        layout.addWidget(browser)

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        btn_box.rejected.connect(self.accept)
        layout.addWidget(btn_box)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About align-sbm")
        self.setFixedSize(420, 240)
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(32, 24, 32, 16)

        title = QLabel("<b style='font-size:18px'>align-sbm</b>")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Side-Bounce Monochromator Alignment GUI")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        layout.addSpacing(8)

        details = QLabel(
            "Automated multi-step SBM alignment protocol<br>"
            "with EPICS motor control and simulation mode.<br><br>"
            "<b>Beamline:</b> ID15A2<br>"
            "<b>Framework:</b> PyQt6 + pyepics"
        )
        details.setAlignment(Qt.AlignmentFlag.AlignCenter)
        details.setWordWrap(True)
        layout.addWidget(details)

        layout.addStretch()

        btn_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok)
        btn_box.accepted.connect(self.accept)
        layout.addWidget(btn_box)
