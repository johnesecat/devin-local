"""Qt stylesheet for the devin-local desktop GUI.

A single dark theme tuned for readability of chat + code. We keep this as
pure QSS text rather than QPalette because QSS is more expressive and easier
to iterate on. The colors come from the "tokyo-night" palette family because
it's friendly to syntax highlighting and HiDPI displays.
"""

from __future__ import annotations

DARK_PALETTE = {
    "bg_0": "#16161e",  # window
    "bg_1": "#1a1b26",  # panels
    "bg_2": "#24283b",  # cards / composer
    "bg_3": "#2f3549",  # hover
    "fg_0": "#c0caf5",  # primary text
    "fg_1": "#9aa5ce",  # secondary text
    "fg_2": "#565f89",  # disabled / muted
    "accent": "#7aa2f7",  # primary brand
    "accent_2": "#bb9af7",  # secondary highlight
    "ok": "#9ece6a",
    "warn": "#e0af68",
    "err": "#f7768e",
    "border": "#3b4261",
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
    background-color: {p["bg_2"]};
    border-top: 1px solid {p["border"]};
}}

QLabel#H1 {{
    font-size: 18px;
    font-weight: 600;
    color: {p["fg_0"]};
}}

QLabel#H2 {{
    font-size: 13px;
    font-weight: 600;
    color: {p["fg_1"]};
    letter-spacing: 0.5px;
}}

QLabel#Muted {{
    color: {p["fg_2"]};
}}

QPushButton {{
    background-color: {p["bg_2"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border"]};
    padding: 6px 12px;
    border-radius: 6px;
}}

QPushButton:hover {{
    background-color: {p["bg_3"]};
}}

QPushButton#Primary {{
    background-color: {p["accent"]};
    color: #0f1117;
    border: none;
    font-weight: 600;
}}

QPushButton#Primary:hover {{
    background-color: #98b6f9;
}}

QPushButton:disabled {{
    color: {p["fg_2"]};
    background-color: {p["bg_1"]};
}}

QLineEdit, QPlainTextEdit, QTextEdit, QComboBox {{
    background-color: {p["bg_2"]};
    color: {p["fg_0"]};
    border: 1px solid {p["border"]};
    border-radius: 6px;
    padding: 6px 8px;
    selection-background-color: {p["accent"]};
    selection-color: #0f1117;
}}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus, QComboBox:focus {{
    border: 1px solid {p["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QListWidget, QTreeView, QListView {{
    background-color: {p["bg_1"]};
    border: none;
    color: {p["fg_0"]};
    outline: none;
}}

QListWidget::item, QTreeView::item {{
    padding: 6px 10px;
    border-radius: 4px;
}}

QListWidget::item:hover, QTreeView::item:hover {{
    background-color: {p["bg_3"]};
}}

QListWidget::item:selected, QTreeView::item:selected {{
    background-color: {p["accent"]};
    color: #0f1117;
}}

QScrollBar:vertical, QScrollBar:horizontal {{
    background: transparent;
    width: 10px;
    height: 10px;
}}

QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
    background: {p["bg_3"]};
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
    border-radius: 8px;
}}

QFrame#MessageBubbleUser {{
    background-color: {p["bg_2"]};
    border: 1px solid {p["border"]};
    border-radius: 10px;
}}

QFrame#MessageBubbleAssistant {{
    background-color: {p["bg_1"]};
    border: 1px solid {p["border"]};
    border-radius: 10px;
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

QSplitter::handle {{
    background-color: {p["border"]};
}}

QSplitter::handle:horizontal {{
    width: 1px;
}}

QSplitter::handle:vertical {{
    height: 1px;
}}
"""
