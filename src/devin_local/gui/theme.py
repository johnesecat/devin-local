"""Qt stylesheet for the devin-local desktop GUI.

A single dark theme tuned for readability of chat + code. We keep this as
pure QSS text rather than QPalette because QSS is more expressive and easier
to iterate on. The colors come from the "tokyo-night" palette family because
it's friendly to syntax highlighting and HiDPI displays.
"""

from __future__ import annotations

DARK_PALETTE = {
    "bg_0": "#0d0f17",  # window
    "bg_1": "#13151f",  # panels
    "bg_2": "#1a1b26",  # cards
    "bg_3": "#24283b",  # composer / hover surface
    "bg_4": "#2f3549",  # active hover
    "fg_0": "#c0caf5",  # primary text
    "fg_1": "#9aa5ce",  # secondary text
    "fg_2": "#565f89",  # muted / disabled
    "accent": "#7aa2f7",  # primary brand
    "accent_2": "#bb9af7",
    "accent_3": "#7dcfff",
    "ok": "#9ece6a",
    "ok_bg": "#1f2a1f",
    "warn": "#e0af68",
    "warn_bg": "#2d2516",
    "err": "#f7768e",
    "err_bg": "#2a1620",
    "border": "#2f3549",
    "border_strong": "#3b4261",
}


def stylesheet() -> str:
    p = DARK_PALETTE
    return f"""
* {{
    font-family: "Segoe UI Variable", "Segoe UI", "Inter", "Helvetica Neue", Arial, sans-serif;
    font-size: 14px;
    color: {p["fg_0"]};
}}

QMainWindow, QDialog {{
    background-color: {p["bg_0"]};
}}

QWidget#Sidebar {{
    background-color: {p["bg_1"]};
    border-right: 1px solid {p["border"]};
}}

QWidget#Inspector {{
    background-color: {p["bg_1"]};
    border-left: 1px solid {p["border"]};
}}

QWidget#StatusBar {{
    background-color: {p["bg_1"]};
    border-top: 1px solid {p["border"]};
}}

QWidget#Composer {{
    background-color: {p["bg_3"]};
    border-top: 1px solid {p["border"]};
}}

QLabel#H1 {{
    font-size: 17px;
    font-weight: 700;
    color: {p["fg_0"]};
    letter-spacing: -0.2px;
}}

QLabel#H2 {{
    font-size: 11px;
    font-weight: 700;
    color: {p["fg_2"]};
    letter-spacing: 1.1px;
}}

QLabel#Muted {{
    color: {p["fg_2"]};
}}

QLabel#BubbleAvatar {{
    color: {p["fg_2"]};
    font-size: 12px;
    font-weight: 600;
    letter-spacing: 0.5px;
}}

QLabel#ToolIcon {{
    font-size: 14px;
    min-width: 18px;
}}

QLabel#ToolName {{
    font-weight: 600;
    color: {p["fg_0"]};
}}

QPushButton {{
    background-color: {p["bg_3"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border_strong"]};
    padding: 6px 12px;
    border-radius: 6px;
}}

QPushButton:hover {{
    background-color: {p["bg_4"]};
}}

QPushButton#Primary {{
    background-color: {p["accent"]};
    color: #0a0c14;
    border: none;
    font-weight: 600;
}}

QPushButton#Primary:hover {{
    background-color: #98b6f9;
}}

QPushButton#Ghost {{
    background-color: transparent;
    border: none;
    color: {p["fg_1"]};
    padding: 4px 8px;
}}

QPushButton#Ghost:hover {{
    color: {p["accent"]};
    background-color: {p["bg_2"]};
    border-radius: 4px;
}}

QPushButton:disabled {{
    color: {p["fg_2"]};
    background-color: {p["bg_1"]};
    border: 1px solid {p["border"]};
}}

QLineEdit, QPlainTextEdit, QTextEdit, QTextBrowser, QComboBox {{
    background-color: {p["bg_2"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {p["accent"]};
    selection-color: #0a0c14;
}}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QTextBrowser:focus, QComboBox:focus {{
    border: 1px solid {p["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border_strong"]};
    selection-background-color: {p["accent"]};
    selection-color: #0a0c14;
    padding: 4px;
}}

QListWidget, QTreeView, QListView {{
    background-color: {p["bg_1"]};
    border: none;
    color: {p["fg_0"]};
    outline: none;
    alternate-background-color: {p["bg_2"]};
}}

QListWidget::item, QTreeView::item {{
    padding: 6px 10px;
    border-radius: 4px;
}}

QListWidget#PlanList::item {{
    padding: 4px 6px;
}}

QListWidget::item:hover, QTreeView::item:hover {{
    background-color: {p["bg_3"]};
}}

QListWidget::item:selected, QTreeView::item:selected {{
    background-color: {p["accent"]};
    color: #0a0c14;
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    width: 10px;
    height: 10px;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {p["bg_4"]};
    border-radius: 5px;
    min-height: 24px;
    min-width: 24px;
}}

QScrollBar::handle:hover {{
    background: {p["fg_2"]};
}}

QScrollBar::add-line, QScrollBar::sub-line {{
    background: none;
    height: 0px;
    width: 0px;
}}

QFrame#ToolCard {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border"]};
    border-radius: 10px;
}}

QFrame#MessageBubbleUser {{
    background-color: {p["bg_3"]};
    border: 1px solid {p["border_strong"]};
    border-radius: 12px;
}}

QFrame#MessageBubbleAssistant {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border"]};
    border-radius: 12px;
}}

QFrame#CodeBlock {{
    background-color: {p["bg_1"]};
    border: 1px solid {p["border"]};
    border-radius: 8px;
}}

QWidget#CodeBlockHeader {{
    background-color: {p["bg_2"]};
    border-top-left-radius: 8px;
    border-top-right-radius: 8px;
    border-bottom: 1px solid {p["border"]};
}}

QPlainTextEdit#CodeBlockBody, QTextBrowser#CodeBlockBody {{
    background-color: {p["bg_1"]};
    border: none;
    padding: 8px 10px;
    color: {p["fg_0"]};
}}

QFrame#PlanPane, QFrame#SandboxPanel {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border"]};
    border-radius: 8px;
}}

QLabel#PillRunning {{
    background-color: {p["warn_bg"]};
    color: {p["warn"]};
    font-weight: 600;
    font-size: 11px;
    border: 1px solid {p["warn"]};
    border-radius: 9px;
    padding: 1px 8px;
    max-height: 18px;
}}

QLabel#PillOk {{
    background-color: {p["ok_bg"]};
    color: {p["ok"]};
    font-weight: 600;
    font-size: 11px;
    border: 1px solid {p["ok"]};
    border-radius: 9px;
    padding: 1px 8px;
    max-height: 18px;
}}

QLabel#PillErr {{
    background-color: {p["err_bg"]};
    color: {p["err"]};
    font-weight: 600;
    font-size: 11px;
    border: 1px solid {p["err"]};
    border-radius: 9px;
    padding: 1px 8px;
    max-height: 18px;
}}

QLabel#PillWarn {{
    background-color: {p["warn_bg"]};
    color: {p["warn"]};
    font-weight: 600;
    font-size: 11px;
    border: 1px solid {p["warn"]};
    border-radius: 9px;
    padding: 1px 8px;
    max-height: 18px;
}}

QLabel#BadgeOk {{
    color: {p["ok"]};
    font-weight: 600;
}}

QLabel#BadgeErr {{
    color: {p["err"]};
    font-weight: 600;
}}

QLabel#BadgeWarn {{
    color: {p["warn"]};
    font-weight: 600;
}}

QProgressBar {{
    border: 1px solid {p["border_strong"]};
    border-radius: 4px;
    background-color: {p["bg_2"]};
    text-align: center;
    height: 14px;
}}

QProgressBar::chunk {{
    background-color: {p["accent"]};
    border-radius: 3px;
}}

QSplitter::handle {{
    background-color: {p["border"]};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}

QMenuBar {{
    background-color: {p["bg_1"]};
    border-bottom: 1px solid {p["border"]};
}}

QMenuBar::item {{
    padding: 6px 10px;
}}

QMenuBar::item:selected {{
    background-color: {p["bg_3"]};
}}

QMenu {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border_strong"]};
    padding: 4px;
}}

QMenu::item:selected {{
    background-color: {p["accent"]};
    color: #0a0c14;
    border-radius: 4px;
}}

QToolTip {{
    background-color: {p["bg_3"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border_strong"]};
    padding: 4px 6px;
}}

/* Settings dialog ---------------------------------------------------- */

QTabWidget::pane {{
    background-color: {p["bg_1"]};
    border: 1px solid {p["border"]};
    border-radius: 8px;
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {p["fg_1"]};
    padding: 8px 16px;
    margin-right: 4px;
    border: 1px solid transparent;
    border-bottom: none;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
}}

QTabBar::tab:hover {{
    color: {p["fg_0"]};
    background-color: {p["bg_2"]};
}}

QTabBar::tab:selected {{
    background-color: {p["bg_1"]};
    color: {p["accent"]};
    border: 1px solid {p["border"]};
    border-bottom: 1px solid {p["bg_1"]};
}}

QLabel#Hint {{
    color: {p["fg_2"]};
    font-size: 12px;
}}

QFormLayout QLabel {{
    color: {p["fg_1"]};
}}

QListWidget {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
    padding: 4px;
}}

QListWidget::item {{
    padding: 6px 8px;
    border-radius: 4px;
}}

QListWidget::item:hover {{
    background-color: {p["bg_3"]};
}}

QListWidget::item:selected {{
    background-color: {p["accent"]};
    color: #0a0c14;
}}

QDoubleSpinBox, QSpinBox {{
    background-color: {p["bg_3"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border"]};
    border-radius: 4px;
    padding: 4px 6px;
}}
"""
