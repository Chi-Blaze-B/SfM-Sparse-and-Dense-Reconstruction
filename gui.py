"""视频转 SfM 稀疏/稠密重建 — PySide6 图形界面。"""

import logging
import os
import sys
import threading
import json
from pathlib import Path
from io import BytesIO
from typing import Optional, Dict, Any, List

import cv2
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import psutil

logger = logging.getLogger(__name__)


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def set_affinity_to_all_cores() -> None:
    try:
        p = psutil.Process(os.getpid())
        all_cpus = list(range(psutil.cpu_count()))
        p.cpu_affinity(all_cpus)
        logger.info("CPU 亲和性设置为 %d 个核心", len(all_cpus))
    except Exception as e:
        logger.warning("无法设置 CPU 亲和性: %s", e)


from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QLineEdit, QFileDialog, QProgressBar, QTextEdit,
    QScrollArea, QMessageBox, QStackedWidget, QFormLayout, QListWidget,
    QFrame, QGraphicsDropShadowEffect, QCheckBox,
)
from PySide6.QtCore import Qt, QThread, Signal, QPoint, QPropertyAnimation, QTimer
from PySide6.QtGui import QPixmap, QImage, QFont, QColor, QPainter

from frames import extract_frames
from poses import estimate_poses, CameraPose, FrameStatus
from exporter import (
    intr_from_K,
    write_ply,
    write_cameras_txt,
    write_intrinsics_json,
)
import frame_meta


# ---------------------------------------------------------------------------
# 设计系统
# ---------------------------------------------------------------------------
C = {
    "bg":             "#0f1724",
    "bg_sidebar":     "#141e2c",
    "bg_panel":       "#192436",
    "bg_card":        "#1c2a3e",
    "bg_input":       "#0f1724",
    "border":         "#253346",
    "border_light":   "#1e2a38",
    "border_focus":   "#3498db",
    "text_primary":   "#e6edf3",
    "text_secondary": "#8b949e",
    "text_muted":     "#6e7681",
    "accent":         "#3498db",
    "accent_hover":   "#5dade2",
    "success":        "#2ecc71",
    "success_bg":     "#239b56",
    "danger":         "#e74c3c",
    "danger_bg":      "#c0392b",
    "purple":         "#9b59b6",
    "title":          "#ffffff",
}

RADIUS_CARD = 12
RADIUS_INPUT = 8
RADIUS_BTN = 8


def card_bg():
    return (
        f"background-color: {C['bg_card']}; "
        f"border: 1px solid {C['border']}; "
        f"border-radius: {RADIUS_CARD}px;"
    )


def input_fg():
    return (
        f"background-color: {C['bg_input']}; "
        f"color: {C['text_primary']}; "
        f"border: 1px solid {C['border']}; "
        f"border-radius: {RADIUS_INPUT}px; "
        f"padding: 6px 12px; font-size: 12px;"
    )


# ---------------------------------------------------------------------------
# 自定义控件
# ---------------------------------------------------------------------------

class RoundedCard(QFrame):
    """带圆角与阴影的卡片容器。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(card_bg())
        self.setContentsMargins(0, 0, 0, 0)
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(20)
        shadow.setXOffset(0)
        shadow.setYOffset(4)
        shadow.setColor(QColor(0, 0, 0, 100))
        self.setGraphicsEffect(shadow)


class StyledLabel(QLabel):
    def __init__(self, text="", font_size=11, color=None, bold=False, parent=None):
        super().__init__(text, parent)
        c = color or C["text_secondary"]
        w = "bold;" if bold else ""
        self.setStyleSheet(
            f"color: {c}; font-size: {font_size}pt; font-weight: {w}; "
            f"font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;"
        )


class StyledSpinBox(QWidget):
    """带 +/- 按钮的整数输入框。"""

    valueChanged = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0
        self._min_val = 0
        self._max_val = 9999
        self._single_step = 1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        btn_style = f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']}; border: none;
                border-radius: 6px; font-size: 16px; font-weight: bold; padding: 0; line-height: 1;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """
        self._btn_down = QPushButton("−", self)
        self._btn_down.setFixedSize(24, 28)
        self._btn_down.setCursor(Qt.PointingHandCursor)
        self._btn_down.setStyleSheet(btn_style)
        self._btn_down.clicked.connect(self._decrement)

        self._edit = QLineEdit(self)
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setMaxLength(12)
        self._edit.setText(str(self._value))
        self._edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 2px 4px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{ border: 1px solid {C['border_focus']}; }}
        """)
        self._edit.returnPressed.connect(self._commit_value)

        self._btn_up = QPushButton("+", self)
        self._btn_up.setFixedSize(24, 28)
        self._btn_up.setCursor(Qt.PointingHandCursor)
        self._btn_up.setStyleSheet(btn_style)
        self._btn_up.clicked.connect(self._increment)

        layout.addWidget(self._btn_down)
        layout.addWidget(self._edit)
        layout.addWidget(self._btn_up)

        self._apply_edit_width()

    def _apply_edit_width(self):
        lo_txt = str(self._min_val)
        hi_txt = str(self._max_val)
        widest = lo_txt if len(lo_txt) > len(hi_txt) else hi_txt
        fm = self._edit.fontMetrics()
        text_w = fm.horizontalAdvance(widest)
        self._edit.setFixedWidth(max(int(text_w + 22), 56))

    def _increment(self):
        self._set_value(min(self._value + self._single_step, self._max_val))

    def _decrement(self):
        self._set_value(max(self._value - self._single_step, self._min_val))

    def _set_value(self, val):
        if val != self._value:
            self._value = val
            self._edit.setText(str(val))
            self.valueChanged.emit(val)

    def _commit_value(self):
        try:
            val = int(self._edit.text())
            self._set_value(max(self._min_val, min(self._max_val, val)))
        except ValueError:
            self._edit.setText(str(self._value))

    def setValue(self, val):
        self._set_value(max(self._min_val, min(self._max_val, val)))

    def value(self):
        return self._value

    def setRange(self, lo, hi):
        self._min_val = lo
        self._max_val = hi
        self._apply_edit_width()

    def setSingleStep(self, step):
        self._single_step = step


class StyledDoubleSpinBox(QWidget):
    """带 +/- 按钮的浮点输入框。"""

    valueChanged = Signal(float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._value = 0.0
        self._min_val = 0.0
        self._max_val = 9999.0
        self._single_step = 1.0

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        btn_style = f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']}; border: none;
                border-radius: 6px; font-size: 16px; font-weight: bold; padding: 0; line-height: 1;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """
        self._btn_down = QPushButton("−")
        self._btn_down.setFixedSize(24, 28)
        self._btn_down.setCursor(Qt.PointingHandCursor)
        self._btn_down.setStyleSheet(btn_style)
        self._btn_down.clicked.connect(self._decrement)

        self._edit = QLineEdit()
        self._edit.setAlignment(Qt.AlignCenter)
        self._edit.setMaxLength(12)
        self._edit.setText(str(self._value))
        self._edit.setStyleSheet(f"""
            QLineEdit {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']}; border-radius: 6px;
                padding: 2px 4px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
            QLineEdit:focus {{ border: 1px solid {C['border_focus']}; }}
        """)
        self._edit.returnPressed.connect(self._commit_value)

        self._btn_up = QPushButton("+")
        self._btn_up.setFixedSize(24, 28)
        self._btn_up.setCursor(Qt.PointingHandCursor)
        self._btn_up.setStyleSheet(btn_style)
        self._btn_up.clicked.connect(self._increment)

        layout.addWidget(self._btn_down)
        layout.addWidget(self._edit)
        layout.addWidget(self._btn_up)

        self._apply_edit_width()

    def _apply_edit_width(self):
        def _fmt(v):
            return f"{v:g}"
        lo_txt = _fmt(self._min_val)
        hi_txt = _fmt(self._max_val)
        widest = lo_txt if len(lo_txt) > len(hi_txt) else hi_txt
        fm = self._edit.fontMetrics()
        text_w = fm.horizontalAdvance(widest)
        self._edit.setFixedWidth(max(int(text_w + 22), 56))

    def _increment(self):
        self._set_value(min(self._value + self._single_step, self._max_val))

    def _decrement(self):
        self._set_value(max(self._value - self._single_step, self._min_val))

    def _set_value(self, val):
        if abs(val - self._value) > 1e-10:
            self._value = val
            self._edit.setText(f"{val:g}")
            self.valueChanged.emit(val)

    def _commit_value(self):
        try:
            val = float(self._edit.text())
            self._set_value(max(self._min_val, min(self._max_val, val)))
        except ValueError:
            self._edit.setText(str(self._value))

    def setValue(self, val):
        self._set_value(max(self._min_val, min(self._max_val, val)))

    def value(self):
        return self._value

    def setRange(self, lo, hi):
        self._min_val = lo
        self._max_val = hi
        self._apply_edit_width()

    def setSingleStep(self, step):
        self._single_step = step


class StyledComboBox(QWidget):
    """自绘下拉框，弹出独立 QFrame 列表。"""

    currentTextChanged = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items = []
        self._current_index = -1

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self._display = QLabel("—")
        self._display.setStyleSheet(f"""
            QLabel {{
                background-color: {C['bg_input']}; color: {C['text_primary']};
                border: 1px solid {C['border']};
                border-top-left-radius: 6px; border-bottom-left-radius: 6px;
                padding: 4px 10px; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
            }}
        """)
        self._display.setAlignment(Qt.AlignVCenter)
        layout.addWidget(self._display, stretch=1)

        self._arrow_btn = QPushButton("▼")
        self._arrow_btn.setFixedSize(28, 30)
        self._arrow_btn.setCursor(Qt.PointingHandCursor)
        self._arrow_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent; color: {C['text_secondary']};
                border: 1px solid {C['border']};
                border-top-right-radius: 6px; border-bottom-right-radius: 6px;
                font-size: 12px; padding: 0;
            }}
            QPushButton:hover {{
                background-color: {C['border_light']}; color: {C['accent']};
            }}
        """)
        self._arrow_btn.clicked.connect(self._toggle_popup)
        layout.addWidget(self._arrow_btn)

        self._popup = QFrame(None, Qt.Popup | Qt.FramelessWindowHint)
        self._popup.setWindowFlags(Qt.Popup | Qt.FramelessWindowHint)
        self._popup.setStyleSheet(f"QFrame {{ background-color: {C['bg_card']}; border: 1px solid {C['border']}; border-radius: {RADIUS_CARD}px; }}")
        pl = QVBoxLayout(self._popup)
        pl.setContentsMargins(2, 2, 2, 2)
        pl.setSpacing(0)

        self._list_widget = QListWidget()
        self._list_widget.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._list_widget.setStyleSheet(f"""
            QListWidget {{
                background-color: {C['bg_card']}; color: {C['text_primary']};
                border: none; font-size: 12px;
                font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;
                padding: 2px;
            }}
            QListWidget::item {{ min-height: 28px; padding: 2px 10px; border-radius: 4px; margin: 1px 4px; }}
            QListWidget::item:selected {{ background-color: {C['accent']}; color: #fff; }}
            QListWidget::item:hover:!selected {{ background-color: {C['bg_input']}; color: {C['text_primary']}; }}
        """)
        self._list_widget.currentRowChanged.connect(self._on_select)
        pl.addWidget(self._list_widget)

        self._anim = QPropertyAnimation(self._popup, b"windowOpacity")
        self._anim.setDuration(120)
        self._anim.setStartValue(0.0)
        self._anim.setEndValue(1.0)
        self._is_open = False

    def addItems(self, items):
        self._items = list(items)
        self._list_widget.clear()
        for item in items:
            self._list_widget.addItem(item)
        if items:
            self.setCurrentIndex(0)

    def setCurrentIndex(self, idx):
        if 0 <= idx < len(self._items):
            self._current_index = idx
            self._display.setText(self._items[idx])
            self._list_widget.setCurrentRow(idx)

    def currentIndex(self):
        return self._current_index

    def currentText(self):
        return self._items[self._current_index] if 0 <= self._current_index < len(self._items) else ""

    def _toggle_popup(self):
        if self._is_open:
            self._close_popup()
        else:
            self._open_popup()

    def _open_popup(self):
        scr = QApplication.primaryScreen().geometry()
        pos = self.mapToGlobal(QPoint(0, self.height()))
        self._popup.show()
        self._popup.raise_()
        self._popup.activateWindow()

        ih = self._list_widget.sizeHintForRow(0) + 4
        v = min(len(self._items), 5)
        ph = v * ih + 8
        self._popup.resize(self._display.width() + 28, ph)

        x, y = pos.x(), pos.y()
        if x + self._popup.width() > scr.right():
            x = scr.right() - self._popup.width()
        if y + self._popup.height() > scr.bottom():
            y = pos.y() - self._popup.height()
        if y < scr.top():
            y = pos.y()
        self._popup.move(x, y)

        self._popup.setWindowOpacity(0.0)
        self._anim.start()
        self._is_open = True
        self._list_widget.setFocus(Qt.PopupFocusReason)

    def _close_popup(self):
        self._popup.hide()
        self._is_open = False

    def _on_select(self, row):
        if 0 <= row < len(self._items):
            self._current_index = row
            self._display.setText(self._items[row])
            self.currentTextChanged.emit(self._items[row])
            self._close_popup()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_Escape and self._is_open:
            self._close_popup()
        elif e.key() in (Qt.Key_Down, Qt.Key_Up, Qt.Key_Enter, Qt.Key_Space) and not self._is_open:
            self._open_popup()
        else:
            super().keyPressEvent(e)


class StyledButton(QPushButton):
    def __init__(self, text, bg_color=C["bg_card"], hover_color=None, radius=RADIUS_BTN, font_size=13, parent=None):
        super().__init__(text, parent)
        self.setCursor(Qt.PointingHandCursor)
        hc = hover_color or bg_color
        self.setStyleSheet(f"""
            QPushButton {{
                background-color: {bg_color}; color: {C["text_primary"]}; border: none;
                border-radius: {radius}px; font-size: {font_size}px; font-weight: bold;
                padding: 8px 20px; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif;
                border-bottom: 2px solid rgba(0,0,0,0.3);
            }}
            QPushButton:hover {{ background-color: {hc}; }}
            QPushButton:pressed {{ background-color: {bg_color}; opacity: 0.9; border-bottom: 1px solid rgba(0,0,0,0.3); }}
            QPushButton:disabled {{ background-color: {C["border_light"]}; color: {C["text_muted"]}; border-bottom: 2px solid transparent; }}
        """)


class AccentButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, C["accent"], C["accent_hover"], parent=parent)


class SuccessButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #239b56, stop:1 #2ecc71)", "#27ae60", parent=parent)


class DangerButton(StyledButton):
    def __init__(self, text, parent=None):
        super().__init__(text, "qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #c0392b, stop:1 #e74c3c)", "#e74c3c", parent=parent)


class ThinProgressBar(QProgressBar):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(6)
        self.setTextVisible(False)
        self.setValue(0)
        self.setStyleSheet(f"""
            QProgressBar {{ background-color: {C["border_light"]}; border: none; border-radius: 3px; }}
            QProgressBar::chunk {{ background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {C["accent"]}, stop:1 {C["purple"]}); border-radius: 3px; }}
        """)


class StatusDot(QWidget):
    """状态圆点：idle 灰 / running 绿 / error 红 / done 绿。"""

    def __init__(self, status="idle", parent=None):
        super().__init__(parent)
        self.setFixedSize(10, 10)
        self.status = status
        self._colors = {"idle": C["text_muted"], "running": C["success"], "error": C["danger"], "done": C["success"]}
        self.color = self._colors[status]

    def set_status(self, s):
        self.status = s
        self.color = self._colors.get(s, C["text_muted"])
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(self.color))
        p.drawEllipse(1, 1, 8, 8)


class LogViewer(QTextEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet(f"""
            QTextEdit {{
                background-color: {C["bg"]}; color: {C["text_secondary"]};
                border: 1px solid {C["border_light"]}; border-radius: {RADIUS_INPUT}px;
                padding: 10px 14px; line-height: 1.6;
            }}
        """)

    def append_log(self, msg):
        self.append(msg)
        sb = self.verticalScrollBar()
        sb.setValue(sb.maximum())


class PreviewImage(QLabel):
    """视频预览占位图。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(360, 260)
        self.setMaximumSize(560, 380)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"QLabel {{ background-color: {C['bg']}; border: 2px dashed {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']}; font-size: 13px; }}")
        self.setText("📷  选择视频以预览")

    def set_image(self, img):
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, c = img.shape
        qimg = QImage(img.tobytes(), w, h, w * c, QImage.Format_RGB888)
        self.setPixmap(QPixmap.fromImage(qimg).scaled(self.width() - 4, self.height() - 4, Qt.KeepAspectRatio, Qt.SmoothTransformation))


# ---------------------------------------------------------------------------
# 相机轨迹 / 点云可视化页
# ---------------------------------------------------------------------------

class TrajectoryPage(QWidget):
    """两幅 matplotlib 图：相机轨迹 + 稀疏点云俯视投影。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(14)

        header = QHBoxLayout()
        header.addWidget(StyledLabel("🧭 重建结果", font_size=14, bold=True, color=C["title"]))
        header.addStretch()
        self.stat_label = StyledLabel("等待重建…", font_size=11, color=C["text_muted"])
        header.addWidget(self.stat_label)
        layout.addLayout(header)

        tc = RoundedCard()
        tl = QVBoxLayout(tc)
        tl.setContentsMargins(16, 14, 16, 14)
        tl.setSpacing(10)
        tl.addWidget(StyledLabel("相机轨迹（XZ 平面，俯视）", font_size=12, bold=True, color=C["text_secondary"]))
        self.traj_image = QLabel()
        self.traj_image.setAlignment(Qt.AlignCenter)
        self.traj_image.setMinimumHeight(240)
        self.traj_image.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']};")
        self.traj_image.setText("暂无数据")
        tl.addWidget(self.traj_image, stretch=1)
        layout.addWidget(tc, stretch=1)

        pc = RoundedCard()
        pl = QVBoxLayout(pc)
        pl.setContentsMargins(16, 14, 16, 14)
        pl.setSpacing(10)
        pl.addWidget(StyledLabel("稀疏点云（XZ 平面投影）", font_size=12, bold=True, color=C["text_secondary"]))
        self.pc_image = QLabel()
        self.pc_image.setAlignment(Qt.AlignCenter)
        self.pc_image.setMinimumHeight(240)
        self.pc_image.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: {RADIUS_CARD}px; color: {C['text_muted']};")
        self.pc_image.setText("暂无数据")
        pl.addWidget(self.pc_image, stretch=1)
        layout.addWidget(pc, stretch=1)

        self._mpl_style = {
            "figure.facecolor": C["bg"],
            "axes.facecolor": C["bg"],
            "axes.edgecolor": C["border"],
            "axes.labelcolor": C["text_secondary"],
            "xtick.color": C["text_muted"],
            "ytick.color": C["text_muted"],
            "grid.color": C["border"],
            "grid.alpha": 0.5,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei UI", "Segoe UI", "DejaVu Sans"],
            "font.size": 10,
            "lines.linewidth": 2.2,
            "lines.markersize": 4,
        }

    def update_from_sfm(self, poses: List[Optional[CameraPose]],
                        xyz: Optional[np.ndarray],
                        keyframes: Optional[List[int]] = None,
                        loop_closures: Optional[List] = None,
                        frame_status: Optional[List[str]] = None):
        centers = []
        for p in poses:
            if p is None:
                continue
            centers.append(p.center)
        centers = np.asarray(centers, dtype=np.float64) if centers else np.empty((0, 3))

        n_cam = len(centers)
        n_pts = 0 if xyz is None else int(xyz.shape[0])
        n_kf = len(keyframes) if keyframes else 0
        n_loop = len(loop_closures) if loop_closures else 0
        n_lost = 0
        if frame_status:
            n_lost = sum(1 for s in frame_status
                         if s in (FrameStatus.LOST, FrameStatus.INVALID))
        self.stat_label.setText(
            f"相机 {n_cam} · 关键帧 {n_kf} · 点 {n_pts} · 回环 {n_loop} · 丢失 {n_lost}"
        )

        self._render_trajectory_chart(centers)
        self._render_pointcloud_chart(centers, xyz)

    def _render_trajectory_chart(self, centers: np.ndarray):
        with plt.style.context(self._mpl_style):
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
            fig.patch.set_facecolor(C["bg"])
            ax.set_facecolor(C["bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if centers.shape[0] < 2:
                ax.text(0.5, 0.5, "暂无数据", transform=ax.transAxes,
                        ha="center", va="center", color=C["text_muted"], fontsize=12)
            else:
                xs = centers[:, 0]
                zs = centers[:, 2]
                ax.plot(xs, zs, color="#3498db", linewidth=1.6, alpha=0.85,
                        marker="o", markersize=3, markeredgewidth=0)
                ax.scatter([xs[0]], [zs[0]], c="#2ecc71", s=42, zorder=5,
                           edgecolors="#0f1724", linewidths=1.0, label="起点")
                ax.scatter([xs[-1]], [zs[-1]], c="#e74c3c", s=42, zorder=5,
                           edgecolors="#0f1724", linewidths=1.0, label="终点")
                ax.set_xlabel("X", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_ylabel("Z", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_title(f"相机轨迹 · {centers.shape[0]} 个位姿",
                             color=C["title"], fontsize=13, pad=12)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
                ax.set_aspect("equal", adjustable="datalim")
                ax.legend(facecolor=C["bg_card"], edgecolor=C["border"],
                          labelcolor=C["text_secondary"], fontsize=9, loc="best")

            fig.tight_layout()
            self._set_pixmap(self.traj_image, fig)
            plt.close(fig)

    def _render_pointcloud_chart(self, centers: np.ndarray, xyz: Optional[np.ndarray]):
        with plt.style.context(self._mpl_style):
            fig, ax = plt.subplots(figsize=(8, 3.2), dpi=150)
            fig.patch.set_facecolor(C["bg"])
            ax.set_facecolor(C["bg"])
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            if xyz is None or xyz.shape[0] == 0:
                ax.text(0.5, 0.5, "暂无数据", transform=ax.transAxes,
                        ha="center", va="center", color=C["text_muted"], fontsize=12)
            else:
                pts = np.asarray(xyz, dtype=np.float64)
                # 采样避免过多散点拖慢渲染
                if pts.shape[0] > 20000:
                    idx = np.linspace(0, pts.shape[0] - 1, 20000).astype(int)
                    pts_plot = pts[idx]
                else:
                    pts_plot = pts
                ax.scatter(pts_plot[:, 0], pts_plot[:, 2],
                           s=1.6, c="#9b59b6", alpha=0.55,
                           edgecolors="none", marker=".")
                if centers.shape[0] >= 1:
                    ax.plot(centers[:, 0], centers[:, 2],
                            color="#3498db", linewidth=1.2, alpha=0.85,
                            marker="o", markersize=3, markeredgewidth=0)
                ax.set_xlabel("X", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_ylabel("Z", color=C["text_secondary"], fontsize=11, labelpad=8)
                ax.set_title(f"稀疏点云 · {pts.shape[0]} 点",
                             color=C["title"], fontsize=13, pad=12)
                ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.5)
                ax.set_aspect("equal", adjustable="datalim")

            fig.tight_layout()
            self._set_pixmap(self.pc_image, fig)
            plt.close(fig)

    def _set_pixmap(self, label, fig):
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight",
                    facecolor=fig.get_facecolor(), edgecolor='none')
        buf.seek(0)
        img = QImage()
        if img.loadFromData(buf.read(), "PNG"):
            label.setPixmap(QPixmap.fromImage(img).scaled(
                label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        else:
            label.setText("图片加载失败")
        buf.close()


# ---------------------------------------------------------------------------
# 工作线程
# ---------------------------------------------------------------------------

class PipelineWorker(QThread):
    """后台执行 SfM 流水线（抽帧 → 位姿估计 → 稠密重建 → 导出）。"""

    log_signal = Signal(str)
    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)
    frame_paths_signal = Signal(list)
    sfm_result_signal = Signal(object)

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self._running = True
        self._stop_event = threading.Event()

    def stop(self):
        self._running = False
        self._stop_event.set()

    def _log(self, msg):
        if self._running:
            self.log_signal.emit(msg)

    def _set_progress(self, pct, msg):
        if self._running:
            self.progress_signal.emit(pct, msg)

    def run(self):
        try:
            self._run_pipeline()
        except Exception as e:
            import traceback
            self._log(f"错误: {e}")
            self._log(traceback.format_exc())
            self.finished_signal.emit(False, str(e))

    def _run_pipeline(self):
        c = self.config
        run_mode = c.get("run_mode", "sparse")
        enable_dense = (run_mode == "dense")
        n_steps = 4 if enable_dense else 3

        workdir = Path(c["workdir"])
        workdir.mkdir(parents=True, exist_ok=True)
        frame_dir = workdir / "frames"
        poses_dir = workdir / "poses"
        output_dir = Path(c["output_dir"])
        output_dir.mkdir(parents=True, exist_ok=True)

        has_frames = frame_meta.can_reuse_frames(workdir)
        has_poses = (
            (workdir / "intrinsics.npy").exists()
            and (workdir / "poses.npy").exists()
            and (workdir / "sparse_points.npy").exists()
        )

        # ---- 步骤 1：抽帧 ----
        self._log(f"[1/{n_steps}] 正在提取视频帧...")
        self._set_progress(2, "正在提取视频帧...")

        meta_path = workdir / "frame_meta.json"
        current_meta = {
            "video": os.path.abspath(c["video"]),
            "scale": c["scale"],
            "fps": c["fps"],
            "sampling_mode": c["sampling_mode"],
            "feature_type": c.get("feature_type", "orb"),
            "min_frames": c["min_frames"],
            "max_frames": c["max_frames"],
        }

        if has_frames and meta_path.exists():
            _mismatch = frame_meta.check_meta(meta_path, current_meta)
            if _mismatch:
                self._log(
                    f"  ⚠️  提取参数变化（{'; '.join(_mismatch)}），"
                    f"重新提取帧并作废下游缓存"
                )
                has_frames = False
                has_poses = False
                for _name in frame_meta.invalidate_downstream(workdir):
                    self._log(f"  🗑  已删除过期文件 {_name}")
        elif has_frames:
            self._log("  提示: 复用旧帧缓存（无 frame_meta.json，未校验提取参数）")

        if has_frames:
            frame_paths = [p.strip() for p in (workdir / "frame_paths.txt").read_text().splitlines()]
            self._log(f"  已加载 {len(frame_paths)} 帧（跳过抽帧）")
        else:
            smart_sampling = c["sampling_mode"] != "uniform"
            two_stage = c["sampling_mode"] == "two-stage"

            frame_paths = extract_frames(
                video_path=c["video"],
                output_dir=str(frame_dir),
                fps=c["fps"],
                scale=c["scale"],
                min_frames=c["min_frames"],
                max_frames=c["max_frames"],
                smart_sampling=smart_sampling,
                two_stage=two_stage,
                poses_output_dir=str(poses_dir / "coarse_poses") if two_stage else None,
                optical_flow_method="farneback",
                feature_type=c.get("feature_type", "orb"),
            )
            (workdir / "frame_paths.txt").write_text("\n".join(frame_paths))
            frame_meta.write_meta(meta_path, current_meta)
            self._log(f"  已提取 {len(frame_paths)} 帧")

        self.frame_paths_signal.emit(frame_paths)

        if self._stop_event.is_set():
            self._log("  用户已取消")
            self.finished_signal.emit(False, "已取消")
            return

        if len(frame_paths) < 2:
            self._log("  错误: 至少需要 2 帧。")
            self.finished_signal.emit(False, "帧数不足")
            return

        _img = cv2.imread(frame_paths[0])
        if _img is None:
            self._log("  错误: 无法读取首帧。")
            self.finished_signal.emit(False, "无法读取帧")
            return
        h, w = _img.shape[:2]
        self._log(f"  分辨率: {w}x{h}")

        self._set_progress(15, f"{len(frame_paths)} 帧就绪")

        # ---- 步骤 2：位姿估计 ----
        self._log(f"\n[2/{n_steps}] 正在估算相机位姿...")
        self._set_progress(20, "正在估算相机位姿...")

        sfm_result = None
        K = None
        cached_keyframes = None
        cached_loop_closures = []
        cached_frame_status = None

        sfm_meta_path = workdir / "sfm_meta.json"

        if has_poses:
            try:
                K = np.load(workdir / "intrinsics.npy")
                poses_data = np.load(workdir / "poses.npy")
                sparse_points = np.load(workdir / "sparse_points.npy")
                poses = []
                for p in poses_data:
                    if np.isnan(p).any():
                        poses.append(None)
                    else:
                        poses.append(CameraPose(R=p[:3, :3].copy(), t=p[:3, 3].copy()))
                if len(poses) > len(frame_paths):
                    poses = poses[:len(frame_paths)]
                while len(poses) < len(frame_paths):
                    poses.append(None)

                # 尝试读回 sfm_meta（关键帧 / 回环 / 帧状态）
                if sfm_meta_path.exists():
                    try:
                        _m = json.loads(sfm_meta_path.read_text())
                        cached_keyframes = _m.get("keyframes")
                        cached_loop_closures = [tuple(x)
                                                for x in _m.get("loop_closures", [])]
                        cached_frame_status = _m.get("frame_status")
                    except Exception as e:
                        self._log(f"  ⚠️  读取 sfm_meta.json 失败: {e}")

                n_loaded_valid = sum(1 for p in poses if p is not None)
                self._log(
                    f"  已加载 SfM 快照（intrinsics / poses / sparse_points）："
                    f"{len(poses)} 帧（有效 {n_loaded_valid}），跳过位姿估计"
                )
            except Exception as e:
                self._log(f"  ⚠️  读取 SfM 快照失败: {e}，重新估算")
                has_poses = False

        if not has_poses:
            focal_guess = None
            if c.get("use_focal_guess"):
                focal_guess = float(max(w, h))
                fov_deg = 2.0 * np.degrees(np.arctan(max(w, h) / (2.0 * focal_guess)))
                axis = "水平" if w >= h else "垂直"
                self._log(f"  初始焦距猜测: {focal_guess:.1f}px（约 {fov_deg:.0f}° {axis} FOV）")

            feature_type = c.get("feature_type", "orb")
            label = "SIFT" if feature_type == "sift" else "ORB"
            self._log(f"  使用 {label} + E + EM-BA 进行位姿估算...")
            try:
                sfm_result = estimate_poses(
                    frame_paths,
                    min_inliers=25,
                    feature_type=feature_type,
                    focal_guess=focal_guess,
                    aspect_ratio=1.0,
                    enable_loop=c.get("enable_loop", True),
                    enable_pgo=c.get("enable_pgo", True),
                )
                K = sfm_result.intrinsics.K
                poses = sfm_result.poses
                sparse_points = sfm_result.xyz
            except RuntimeError as e:
                self._log(f"  [ERROR] 位姿估算失败: {e}")
                self._log("  建议更换素材（提高平移、减少纯旋转、增加纹理）。")
                self.finished_signal.emit(False, f"位姿估算失败: {e}")
                return

            if sfm_result.frame_status is not None:
                from collections import Counter as _Counter
                _st = _Counter(sfm_result.frame_status)
                self._log(f"  SfM 帧状态: {dict(_st)}")
                self._log(f"  SfM 关键帧: {len(sfm_result.keyframes)} / {len(frame_paths)}")
                self._log(f"  SfM 回环: {len(sfm_result.loop_closures)} 处")
                _n_lost = _st.get(FrameStatus.LOST, 0) + _st.get(FrameStatus.INVALID, 0)
                if _n_lost > len(frame_paths) * 0.3:
                    self._log(
                        f"  ⚠️  超过 30% 的帧丢失/无效（{_n_lost} / {len(frame_paths)}），"
                        f"建议更换素材。"
                    )
            np.save(workdir / "intrinsics.npy", K)
            poses_arr = np.full((len(poses), 4, 4), np.nan, dtype=np.float32)
            for i, p in enumerate(poses):
                if p is not None:
                    poses_arr[i] = p.RT
            np.save(workdir / "poses.npy", poses_arr)

            if sparse_points is None:
                sparse_points = np.zeros((0, 3), dtype=np.float32)
            np.save(workdir / "sparse_points.npy", sparse_points)

            # sfm_meta：关键帧 / 回环 / 帧状态
            try:
                sfm_meta_path.write_text(json.dumps({
                    "keyframes": list(sfm_result.keyframes),
                    "loop_closures": [[int(a), int(b)]
                                      for a, b in sfm_result.loop_closures],
                    "frame_status": (list(sfm_result.frame_status)
                                     if sfm_result.frame_status else None),
                }, ensure_ascii=False))
            except Exception as e:
                self._log(f"  ⚠️  写入 sfm_meta.json 失败: {e}")

        while len(poses) < len(frame_paths):
            poses.append(None)

        if sfm_result is not None and sfm_result.frame_status is not None:
            _lost_or_invalid = sum(
                1 for _st in sfm_result.frame_status
                if _st in (FrameStatus.LOST, FrameStatus.INVALID)
            )
            valid_count = len(frame_paths) - _lost_or_invalid
        else:
            valid_count = sum(1 for p in poses if p is not None)
        self._log(f"  {valid_count} 个有效位姿 (共 {len(frame_paths)} 帧)")

        if valid_count < 3:
            self._log("  错误: 有效位姿太少，请检查视频质量")
            self.finished_signal.emit(False, "有效位姿过少")
            return

        # 通知 GUI 绘制相机轨迹 / 点云（缓存命中时不再强切 tab）
        self.sfm_result_signal.emit({
            "poses": list(poses),
            "xyz": np.asarray(sparse_points),
            "keyframes": list(sfm_result.keyframes) if sfm_result else [],
            "loop_closures": list(sfm_result.loop_closures) if sfm_result else [],
            "frame_status": list(sfm_result.frame_status) if (sfm_result and sfm_result.frame_status) else None,
            "switch_tab": True,
        })

        if self._stop_event.is_set():
            self._log("  用户已取消")
            self.finished_signal.emit(False, "已取消")
            return

        # ---- 步骤 3（可选）：稠密重建 ----
        dense_result = None
        if enable_dense:
            self._log(f"\n[3/{n_steps}] 正在稠密重建...")
            self._set_progress(55, "正在稠密重建...")

            try:
                from dense import dense_reconstruct, DenseConfig, DenseAbortedError
            except ImportError as e:
                self._log(f"  [WARN] 无法导入稠密模块 dense.py: {e}，跳过稠密重建")
                enable_dense = False
                n_steps = 3
            else:
                if sfm_result is not None and sfm_result.keyframes:
                    _kf = list(sfm_result.keyframes)
                else:
                    _kf = cached_keyframes
                _intr = sfm_result.intrinsics if sfm_result else intr_from_K(K)

                dense_cfg = DenseConfig(
                    downscale=int(c.get("dense_downscale", 2)),
                    max_pairs=int(c.get("dense_max_pairs", 60)),
                    voxel_ratio=float(c.get("dense_voxel_ratio", 0.005)),
                )
                dense_path = str(output_dir / "dense_points.ply")

                def _dense_prog(i, total, msg):
                    if total > 0:
                        pct = 55 + int(i / total * 35)
                        self._set_progress(min(pct, 90), f"稠密 {i}/{total}：{msg}")
                        if i % 5 == 0 or i == total:
                            self._log(f"  稠密 {i}/{total} · {msg}")

                try:
                    dense_result = dense_reconstruct(
                        frame_paths, _intr, poses, dense_path,
                        keyframes=_kf,
                        config=dense_cfg,
                        progress_callback=_dense_prog,
                        cache_dir=str(workdir / "dense_pairs"),
                        stop_event=self._stop_event,
                    )
                    self._log(
                        f"  ✓ 稠密点云: {dense_path}"
                        f"（{dense_result.num_points} 点，"
                        f"{dense_result.num_pairs_used}/{dense_result.num_pairs_total} 对成功，"
                        f"缓存命中 {dense_result.num_pairs_cached}，"
                        f"耗时 {dense_result.elapsed_sec:.1f}s）"
                    )
                except DenseAbortedError:
                    # 稠密中断不丢弃稀疏成果：继续走导出流程
                    self._log("  ⏹ 稠密重建已中断，缓存已保存（下次将跳过已算的帧对）")
                    self._log("  继续导出稀疏结果……")
                    dense_result = None
                except Exception as e:
                    self._log(f"  [WARN] 稠密重建失败: {e}")

        # ---- 步骤 4（或 3）：导出 ----
        self._log(f"\n[{n_steps}/{n_steps}] 正在导出...")
        self._set_progress(92, "正在导出...")

        ply_path = output_dir / "sparse_points.ply"
        cam_path = output_dir / "cameras.txt"
        intr_path = output_dir / "intrinsics.json"
        report_path = output_dir / "sfm_report.json"

        try:
            write_ply(str(ply_path), np.asarray(sparse_points))
            self._log(f"  ✓ 稀疏点云: {ply_path}")
        except Exception as e:
            self._log(f"  [WARN] 稀疏点云导出失败: {e}")

        try:
            write_cameras_txt(str(cam_path), poses)
            self._log(f"  ✓ 相机位姿: {cam_path}")
        except Exception as e:
            self._log(f"  [WARN] 相机导出失败: {e}")

        try:
            write_intrinsics_json(str(intr_path), K, w, h)
            self._log(f"  ✓ 内参: {intr_path}")
        except Exception as e:
            self._log(f"  [WARN] 内参导出失败: {e}")

        try:
            report = {
                "video": os.path.abspath(c["video"]),
                "run_mode": run_mode,
                "num_frames": len(frame_paths),
                "num_valid_poses": int(valid_count),
                "num_sparse_points": int(np.asarray(sparse_points).shape[0]),
                "num_dense_points": int(dense_result.num_points) if dense_result else 0,
                "dense_pairs_used": int(dense_result.num_pairs_used) if dense_result else 0,
                "num_keyframes": (len(sfm_result.keyframes) if sfm_result else 0),
                "num_loop_closures": (len(sfm_result.loop_closures) if sfm_result else 0),
                "feature_type": c.get("feature_type", "orb"),
                "image_width": int(w),
                "image_height": int(h),
            }
            report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
            self._log(f"  ✓ 报告: {report_path}")
        except Exception as e:
            self._log(f"  [WARN] 报告导出失败: {e}")

        self._log(f"\n完成！输出目录: {os.path.abspath(output_dir)}")
        self._set_progress(100, "完成！")
        self.finished_signal.emit(True, "成功")


# ---------------------------------------------------------------------------
# 主窗口
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """左侧参数栏 + 右侧多标签页（帧预览 / 运行日志 / 重建结果）。"""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.setWindowTitle("视频转 SfM 稀疏/稠密重建")
        self.resize(1300, 820)
        self._preview_paths = []
        self._preview_page = 0
        self._preview_per_page = 24
        self._preview_cols = 3
        self._preview_rows = 8
        self._resize_timer = None
        self._build_ui()
        self._apply_theme()

    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 0)

        sidebar = RoundedCard()
        sidebar.setFixedWidth(410)
        sidebar.setStyleSheet(f"background-color: {C['bg_sidebar']}; border-right: 1px solid {C['border']}; border-radius: 0;")
        sidebar.setGraphicsEffect(None)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(0)

        header_widget = QWidget()
        header_layout = QVBoxLayout(header_widget)
        header_layout.setContentsMargins(20, 20, 20, 0)
        header_layout.setSpacing(4)

        self.title_label = StyledLabel("SfM Reconstruction", font_size=18, bold=True, color=C["title"])
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(StyledLabel("SfM 稀疏 / 稠密点云重建", font_size=11, color=C["text_secondary"]))

        sr = QHBoxLayout()
        self.status_dot = StatusDot("idle")
        self.status_text = StyledLabel("就绪", font_size=10, color=C["text_muted"])
        sr.addWidget(self.status_dot)
        sr.addWidget(self.status_text)
        sr.addStretch()
        header_layout.addLayout(sr)

        sep_top = QFrame()
        sep_top.setFixedHeight(1)
        sep_top.setStyleSheet(f"background-color: {C['border']}; margin: 8px 0;")
        header_layout.addWidget(sep_top)
        sidebar_layout.addWidget(header_widget)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setStyleSheet("QScrollArea { background: transparent; border: none; }")
        scroll.verticalScrollBar().setStyleSheet(f"""
            QScrollBar:vertical {{ background: {C['bg_sidebar']}; width: 6px; }}
            QScrollBar::handle:vertical {{ background: {C['border']}; border-radius: 3px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background: {C['text_secondary']}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ height: 0px; }}
        """)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(20, 12, 20, 20)
        scroll_layout.setSpacing(16)

        self._add_input_section(scroll_layout)
        self._add_param_section(scroll_layout)

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background-color: {C['border']}; margin: 4px 0;")
        scroll_layout.addWidget(sep2)

        btn_area = QWidget()
        bl = QVBoxLayout(btn_area)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(10)

        self.start_btn = SuccessButton("▶ 开始重建")
        self.start_btn.setFixedHeight(42)
        self.start_btn.clicked.connect(self._on_start)
        bl.addWidget(self.start_btn)

        self.stop_btn = DangerButton("■ 停止")
        self.stop_btn.setFixedHeight(42)
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self._on_stop)
        bl.addWidget(self.stop_btn)

        scroll_layout.addWidget(btn_area)

        prog_area = QWidget()
        pl = QVBoxLayout(prog_area)
        pl.setContentsMargins(0, 0, 0, 0)
        pl.setSpacing(4)
        self.progress_label = StyledLabel("就绪", font_size=10, color=C["text_secondary"])
        self.progress_bar = ThinProgressBar()
        pl.addWidget(self.progress_label)
        pl.addWidget(self.progress_bar)
        scroll_layout.addWidget(prog_area)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        sidebar_layout.addWidget(scroll)
        main_layout.addWidget(sidebar)

        ma = QWidget()
        rl = QVBoxLayout(ma)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.setSpacing(0)

        tab_bar = QWidget()
        tbl = QHBoxLayout(tab_bar)
        tbl.setContentsMargins(20, 12, 20, 0)

        tab_style = """
            QPushButton#tabBtnActive {
                background-color: transparent; color: #e8ecf1;
                border-bottom: 2px solid #3498db;
                border-top: none; border-left: none; border-right: none;
                border-radius: 0; padding: 6px 16px; font-size: 13px; font-weight: bold;
            }
            QPushButton#tabBtnInactive {
                background-color: transparent; color: #8b949e;
                border-bottom: 2px solid transparent;
                border-top: none; border-left: none; border-right: none;
                border-radius: 0; padding: 6px 16px; font-size: 13px;
            }
            QPushButton#tabBtnInactive:hover { color: #e8ecf1; }
        """

        self.tab_btn_preview = StyledButton("🖼 帧预览", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_preview.setObjectName("tabBtnActive")
        self.tab_btn_preview.setStyleSheet(tab_style)
        self.tab_btn_preview.clicked.connect(lambda: self._switch_tab(0))

        self.tab_btn_logs = StyledButton("📋 运行日志", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_logs.setObjectName("tabBtnInactive")
        self.tab_btn_logs.setStyleSheet(tab_style)
        self.tab_btn_logs.clicked.connect(lambda: self._switch_tab(1))

        self.tab_btn_traj = StyledButton("🧭 重建结果", C["bg_panel"], C["border"], RADIUS_INPUT, 12)
        self.tab_btn_traj.setObjectName("tabBtnInactive")
        self.tab_btn_traj.setStyleSheet(tab_style)
        self.tab_btn_traj.clicked.connect(lambda: self._switch_tab(2))

        tbl.addWidget(self.tab_btn_preview)
        tbl.addWidget(self.tab_btn_logs)
        tbl.addWidget(self.tab_btn_traj)
        tbl.addStretch()
        rl.addWidget(tab_bar)

        ts = QFrame()
        ts.setFixedHeight(1)
        ts.setStyleSheet(f"background-color: {C['border']};")
        rl.addWidget(ts)

        self.stacked = QStackedWidget()
        self.stacked.setStyleSheet("background-color: transparent; border: none;")

        preview_page = QWidget()
        ppl = QVBoxLayout(preview_page)
        ppl.setContentsMargins(20, 16, 20, 16)
        ppl.setSpacing(12)
        self.preview_widget = PreviewImage()
        ppl.addWidget(self.preview_widget, alignment=Qt.AlignCenter)

        self.thumb_scroll = QScrollArea()
        self.thumb_scroll.setWidgetResizable(True)
        self.thumb_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.thumb_scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        self.thumb_inner = QWidget()
        self.thumb_layout = QVBoxLayout(self.thumb_inner)
        self.thumb_layout.setAlignment(Qt.AlignCenter)
        self.thumb_layout.setSpacing(8)
        self.thumb_scroll.setWidget(self.thumb_inner)
        ppl.addWidget(self.thumb_scroll)

        pager = QHBoxLayout()
        pager.setSpacing(12)
        pager.setAlignment(Qt.AlignCenter)

        self.prev_page_btn = StyledButton("◀ 上一页", C["bg_card"], C["border"], RADIUS_INPUT, 11)
        self.prev_page_btn.setFixedWidth(96)
        self.prev_page_btn.clicked.connect(self._preview_prev_page)

        self.page_label = StyledLabel("帧 0-0 / 0", font_size=11, color=C["text_muted"])
        self.page_label.setAlignment(Qt.AlignCenter)

        self.next_page_btn = StyledButton("下一页 ▶", C["bg_card"], C["border"], RADIUS_INPUT, 11)
        self.next_page_btn.setFixedWidth(96)
        self.next_page_btn.clicked.connect(self._preview_next_page)

        pager.addWidget(self.prev_page_btn)
        pager.addWidget(self.page_label)
        pager.addWidget(self.next_page_btn)
        ppl.addLayout(pager)
        self.stacked.addWidget(preview_page)

        log_page = QWidget()
        ll = QVBoxLayout(log_page)
        ll.setContentsMargins(20, 16, 20, 16)
        ll.setSpacing(0)
        self.log_viewer = LogViewer()
        ll.addWidget(self.log_viewer)
        self.stacked.addWidget(log_page)

        self.trajectory_page = TrajectoryPage()
        self.stacked.addWidget(self.trajectory_page)

        rl.addWidget(self.stacked)
        rl.setStretchFactor(self.stacked, 1)
        main_layout.addWidget(ma, stretch=1)

    def _make_path_row(self, name, default="", browse_kind=None):
        edit = QLineEdit(default)
        edit.setStyleSheet(f"{input_fg()} min-height: 32px;")
        btn = AccentButton("浏览")
        btn.setFixedWidth(64)
        btn.clicked.connect(lambda _, e=edit, k=browse_kind or name: self._browse_file(k, e))
        row = QHBoxLayout()
        row.addWidget(edit)
        row.addWidget(btn)
        row.setSpacing(8)
        setattr(self, f"{name}_edit", edit)
        return row

    def _make_step_header(self, title: str) -> QWidget:
        w = QWidget()
        lay = QHBoxLayout(w)
        lay.setContentsMargins(0, 8, 0, 0)
        lay.setSpacing(8)

        lay.addWidget(StyledLabel(title, font_size=11, bold=True, color=C["accent"]))

        line = QFrame()
        line.setFixedHeight(1)
        line.setStyleSheet(f"background-color: {C['border']};")
        lay.addWidget(line, stretch=1)
        return w

    def _make_form(self) -> QFormLayout:
        form = QFormLayout()
        form.setSpacing(10)
        form.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        form.setFormAlignment(Qt.AlignLeft)
        form.setContentsMargins(0, 0, 0, 0)
        return form

    def _add_input_section(self, parent_layout):
        card = RoundedCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        cl.addWidget(StyledLabel("📁 输入 / 输出 · 开始前设置", font_size=12, bold=True, color=C["title"]))

        form = self._make_form()
        form.addRow(StyledLabel("视频文件:", font_size=11, color=C["text_secondary"]),
                    self._make_path_row("video", browse_kind="video"))
        form.addRow(StyledLabel("输出目录:", font_size=11, color=C["text_secondary"]),
                    self._make_path_row("output_dir", default="./sfm_output", browse_kind="dir"))
        form.addRow(StyledLabel("工作目录:", font_size=11, color=C["text_secondary"]),
                    self._make_path_row("workdir", default="./workdir", browse_kind="dir"))

        cl.addLayout(form)

        self.cache_hint = StyledLabel("", font_size=10, color=C["text_muted"])
        self.cache_hint.setWordWrap(True)
        cl.addWidget(self.cache_hint)

        parent_layout.addWidget(card)

        self.workdir_edit.textChanged.connect(self._refresh_cache_hint)
        self._refresh_cache_hint()

    def _refresh_cache_hint(self):
        wd = self.workdir_edit.text().strip() or "./workdir"
        base = "font-size: 10pt; font-family: 'Microsoft YaHei UI', 'Segoe UI', sans-serif;"
        has_poses = (
            Path(wd, "intrinsics.npy").exists()
            and Path(wd, "poses.npy").exists()
            and Path(wd, "sparse_points.npy").exists()
        )
        has_frames = Path(wd, "frame_paths.txt").exists() and Path(wd, "frames").exists()
        if has_poses:
            self.cache_hint.setText("🗂 检测到 SfM 快照（intrinsics / poses / sparse_points）：将跳过位姿估计")
            self.cache_hint.setStyleSheet(f"color: {C['success']}; {base}")
        elif has_frames:
            self.cache_hint.setText("🗂 检测到帧缓存：将跳过抽帧")
            self.cache_hint.setStyleSheet(f"color: {C['accent']}; {base}")
        else:
            self.cache_hint.setText("○ 未检测到缓存：将从头开始")
            self.cache_hint.setStyleSheet(f"color: {C['text_muted']}; {base}")

    def _add_param_section(self, parent_layout):
        card = RoundedCard()
        cl = QVBoxLayout(card)
        cl.setContentsMargins(16, 14, 16, 14)
        cl.setSpacing(10)
        cl.addWidget(StyledLabel("⚙️ 重建参数", font_size=12, bold=True, color=C["title"]))

        # ---- 运行模式 ----
        cl.addWidget(self._make_step_header("运行模式"))

        self.mode_combo = StyledComboBox()
        self.mode_combo.addItems([
            "稀疏重建（只跑 SfM）",
            "稀疏 + 稠密（全程）",
        ])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.setToolTip(
            "稀疏重建：只输出 SfM 位姿 + 稀疏点云。\n"
            "稀疏 + 稠密：在稀疏基础上再跑 SGBM 立体匹配，输出 dense_points.ply。\n"
            "稠密重建耗时较长，但点云密度通常高 1~2 个数量级。"
        )

        f0 = self._make_form()
        f0.addRow(StyledLabel("模式:", font_size=11, color=C["text_secondary"]),
                  self.mode_combo)
        cl.addLayout(f0)

        # ---- 帧提取 ----
        self.sampling_combo = StyledComboBox()
        self.sampling_combo.addItems(["均匀采样", "智能采样", "两阶段采样"])
        self.sampling_combo.setCurrentIndex(0)

        self.fps_spin = StyledDoubleSpinBox()
        self.fps_spin.setRange(1, 60)
        self.fps_spin.setValue(15)
        self.fps_spin.setSingleStep(1)

        self.scale_spin = StyledDoubleSpinBox()
        self.scale_spin.setRange(0.1, 1.0)
        self.scale_spin.setValue(0.5)
        self.scale_spin.setSingleStep(0.05)

        self.min_frames_spin = StyledSpinBox()
        self.min_frames_spin.setRange(10, 500)
        self.min_frames_spin.setValue(30)

        self.max_frames_spin = StyledSpinBox()
        self.max_frames_spin.setRange(10, 500)
        self.max_frames_spin.setValue(200)

        cl.addWidget(self._make_step_header("帧提取"))
        f1 = self._make_form()
        f1.addRow(StyledLabel("采样模式:", font_size=11, color=C["text_secondary"]), self.sampling_combo)
        f1.addRow(StyledLabel("采样帧率:", font_size=11, color=C["text_secondary"]), self.fps_spin)
        f1.addRow(StyledLabel("画面缩放:", font_size=11, color=C["text_secondary"]), self.scale_spin)
        f1.addRow(StyledLabel("最少帧数:", font_size=11, color=C["text_secondary"]), self.min_frames_spin)
        f1.addRow(StyledLabel("最多帧数:", font_size=11, color=C["text_secondary"]), self.max_frames_spin)
        cl.addLayout(f1)

        # ---- 相机位姿估算 ----
        self.feature_type_label = StyledLabel("特征描述子:", font_size=11, color=C["text_secondary"])
        self.feature_type_combo = StyledComboBox()
        self.feature_type_combo.addItems(["ORB（快）", "SIFT（稳）"])
        self.feature_type_combo.setToolTip(
            "ORB：二进制描述子，Hamming 距离匹配，速度快。\n"
            "SIFT：浮点描述子，L2 距离匹配，更稳健但较慢。"
        )

        self.focal_guess_label = StyledLabel("初始焦距:", font_size=11, color=C["text_secondary"])

        self.focal_guess_cb = QCheckBox("使用初始焦距猜测")
        self.focal_guess_cb.setChecked(True)
        self.focal_guess_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")
        self.focal_guess_cb.setToolTip(
            "以 1.0×图像长边作为初始像素焦距（约 53° 长边方向 FOV）。"
        )

        self.loop_cb = QCheckBox("启用回环检测")
        self.loop_cb.setChecked(True)
        self.loop_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")
        self.loop_cb.setToolTip(
            "基于词袋（BoW）+ PnP RANSAC 几何验证检测回环。\n"
            "短序列或纯前向拍摄可关闭以加速。"
        )
        self.loop_label = StyledLabel("回环检测:", font_size=11, color=C["text_secondary"])

        self.pgo_cb = QCheckBox("启用位姿图优化")
        self.pgo_cb.setChecked(True)
        self.pgo_cb.setStyleSheet(f"color: {C['text_primary']}; font-size: 11px; font-weight: 500;")
        self.pgo_cb.setToolTip("在全局 BA 之后对关键帧位姿做位姿图优化（Huber 核）。")
        self.pgo_label = StyledLabel("位姿图优化:", font_size=11, color=C["text_secondary"])

        cl.addWidget(self._make_step_header("相机位姿估算"))
        f2 = self._make_form()
        f2.addRow(self.feature_type_label, self.feature_type_combo)
        f2.addRow(self.focal_guess_label, self.focal_guess_cb)
        f2.addRow(self.loop_label, self.loop_cb)
        f2.addRow(self.pgo_label, self.pgo_cb)
        cl.addLayout(f2)

        # ---- 稠密参数（仅 dense 模式可见） ----
        self.dense_param_widget = QWidget()
        dp = QVBoxLayout(self.dense_param_widget)
        dp.setContentsMargins(0, 0, 0, 0)
        dp.setSpacing(10)

        self.dense_downscale_spin = StyledSpinBox()
        self.dense_downscale_spin.setRange(1, 4)
        self.dense_downscale_spin.setValue(2)
        self.dense_downscale_spin.setToolTip("帧降采样倍数，越大越快，精度越低")

        self.dense_pairs_spin = StyledSpinBox()
        self.dense_pairs_spin.setRange(4, 500)
        self.dense_pairs_spin.setValue(60)
        self.dense_pairs_spin.setToolTip("最多处理的帧对数量")

        self.dense_voxel_spin = StyledDoubleSpinBox()
        self.dense_voxel_spin.setRange(0.0005, 0.05)
        self.dense_voxel_spin.setValue(0.005)
        self.dense_voxel_spin.setSingleStep(0.001)
        self.dense_voxel_spin.setToolTip("体素大小 / 场景尺度（越小越密，也越慢）")

        dp.addWidget(self._make_step_header("稠密参数"))
        f3 = self._make_form()
        f3.addRow(StyledLabel("降采样倍数:", font_size=11, color=C["text_secondary"]),
                  self.dense_downscale_spin)
        f3.addRow(StyledLabel("最多帧对:", font_size=11, color=C["text_secondary"]),
                  self.dense_pairs_spin)
        f3.addRow(StyledLabel("体素比例:", font_size=11, color=C["text_secondary"]),
                  self.dense_voxel_spin)
        dp.addLayout(f3)
        cl.addWidget(self.dense_param_widget)

        parent_layout.addWidget(card)

        self.mode_combo.currentTextChanged.connect(self._on_mode_changed)
        self._on_mode_changed(self.mode_combo.currentText())

    def _on_mode_changed(self, text: str):
        is_dense = "稠密" in text
        self.dense_param_widget.setVisible(is_dense)

    def _apply_theme(self):
        self.setStyleSheet(f"""
            QMainWindow {{ background-color: {C["bg"]}; }}
            QWidget {{ background-color: transparent; color: {C["text_primary"]}; font-family: "Microsoft YaHei UI", "Segoe UI", sans-serif; }}
            QScrollBar:vertical {{ background-color: {C["bg"]}; width: 8px; border: none; }}
            QScrollBar::handle:vertical {{ background-color: {C["border"]}; border-radius: 4px; min-height: 24px; }}
            QScrollBar::handle:vertical:hover {{ background-color: {C["text_secondary"]}; }}
            QScrollBar::add-line, QScrollBar::sub-line {{ border: none; background: none; }}
            QScrollBar::add-page, QScrollBar::sub-page {{ background: none; }}
            QSplitter::handle {{ background-color: {C["border"]}; }}
        """)

    def _switch_tab(self, idx):
        self.stacked.setCurrentIndex(idx)
        btns = [self.tab_btn_preview, self.tab_btn_logs, self.tab_btn_traj]
        for i, btn in enumerate(btns):
            btn.setObjectName("tabBtnActive" if i == idx else "tabBtnInactive")
            btn.style().unpolish(btn)
            btn.style().polish(btn)

    def _browse_file(self, kind, target_edit):
        if kind == "video":
            path, _ = QFileDialog.getOpenFileName(self, "选择视频", "", "视频文件 (*.mp4 *.avi *.mov *.mkv);;所有文件 (*)")
        elif kind == "dir":
            path = QFileDialog.getExistingDirectory(self, "选择目录")
        else:
            path, _ = QFileDialog.getSaveFileName(self, "保存文件", "")
        if path:
            target_edit.setText(path)
            if kind == "video":
                self._show_single_preview(path)

    def _show_single_preview(self, vp):
        """读取视频中间帧作为预览；失败时回退到第 0 帧。"""
        cap = None
        try:
            cap = cv2.VideoCapture(vp)
            if not cap.isOpened():
                return
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total > 1:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total // 2)
            ok, frame = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = cap.read()
            if ok:
                self.preview_widget.set_image(frame)
        except Exception:
            pass
        finally:
            if cap is not None:
                cap.release()

    def _on_start(self):
        video = self.video_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip() or "./sfm_output"
        workdir = self.workdir_edit.text().strip() or "./workdir"

        if not video or not os.path.isfile(video):
            QMessageBox.warning(self, "提示", "请选择有效的视频文件。")
            return

        mode_map = {"均匀采样": "uniform", "智能采样": "smart", "两阶段采样": "two-stage"}
        sampling_mode = mode_map.get(self.sampling_combo.currentText(), "uniform")

        feature_type = "sift" if self.feature_type_combo.currentIndex() == 1 else "orb"
        run_mode = "dense" if "稠密" in self.mode_combo.currentText() else "sparse"

        config = {
            "video": video,
            "output_dir": output_dir,
            "workdir": workdir,
            "run_mode": run_mode,
            "sampling_mode": sampling_mode,
            "fps": self.fps_spin.value(),
            "scale": self.scale_spin.value(),
            "min_frames": self.min_frames_spin.value(),
            "max_frames": self.max_frames_spin.value(),
            "use_focal_guess": self.focal_guess_cb.isChecked(),
            "enable_loop": self.loop_cb.isChecked(),
            "enable_pgo": self.pgo_cb.isChecked(),
            "feature_type": feature_type,
            "dense_downscale": self.dense_downscale_spin.value(),
            "dense_max_pairs": self.dense_pairs_spin.value(),
            "dense_voxel_ratio": self.dense_voxel_spin.value(),
        }

        self.start_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.status_dot.set_status("running")
        self.status_text.setText("运行中…")
        self.status_text.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
        self.progress_bar.setValue(0)
        self.progress_label.setText("准备启动…")
        self.progress_label.setStyleSheet(f"color: {C['text_secondary']}; font-size: 10pt;")
        self.log_viewer.clear()

        self._log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log("  SfM 重建 启动")
        self._log("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        self._log(f"📹  视频:      {config['video']}")
        self._log(f"💾  输出目录:  {config['output_dir']}")
        self._log(f"📂  工作目录:  {config['workdir']}")
        self._log(f"🧩  运行模式:  {'稀疏 + 稠密' if run_mode == 'dense' else '稀疏重建'}")
        self._log(f"🎯  采样模式:  {config['sampling_mode']}")
        self._log(f"⚙️   帧率:      {config['fps']} FPS")
        self._log(f"📐  缩放比例:  {config['scale']:.2f}")
        self._log(f"🎞️   帧数范围:  {config['min_frames']}–{config['max_frames']}")
        self._log(f"🔎  特征描述子: {config['feature_type'].upper()}")
        self._log(f"🔍  初始焦距猜测: {'开（1.0×长边）' if config['use_focal_guess'] else '关'}")
        self._log(f"🔁  回环检测:  {'开' if config['enable_loop'] else '关'}")
        self._log(f"🧭  位姿图优化: {'开' if config['enable_pgo'] else '关'}")
        if run_mode == "dense":
            self._log(
                f"🧊  稠密参数:  降采样 ×{config['dense_downscale']}，"
                f"最多 {config['dense_max_pairs']} 对，"
                f"体素 {config['dense_voxel_ratio']:.4f}"
            )

        self.worker = PipelineWorker(config)
        self.worker.log_signal.connect(self._append_log)
        self.worker.progress_signal.connect(self._update_progress)
        self.worker.finished_signal.connect(self._on_finished)
        self.worker.frame_paths_signal.connect(self._show_preview_frames)
        self.worker.sfm_result_signal.connect(self._on_sfm_result)
        self.worker.start()

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self._log("\n⏹ 正在停止…")

    def _on_finished(self, success, message):
        self.start_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)

        canceled = (not success) and message == "已取消"

        if success:
            self.status_dot.set_status("done")
            self.status_text.setText("完成 ✓")
            self.status_text.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
            self.progress_bar.setValue(100)
            self.progress_label.setText("重建完成！")
            self.progress_label.setStyleSheet(f"color: {C['success']}; font-size: 10pt;")
            self._log(f"✅ {message}")
            self._log("🎉 SfM 重建完成！请在输出目录查看结果。")
        elif canceled:
            self.status_dot.set_status("idle")
            self.status_text.setText("已取消")
            self.status_text.setStyleSheet(f"color: {C['text_secondary']}; font-size: 10pt;")
            self.progress_label.setText("已取消")
            self.progress_label.setStyleSheet(f"color: {C['text_secondary']}; font-size: 10pt;")
            self._log(f"⏹ {message}")
        else:
            self.status_dot.set_status("error")
            self.status_text.setText("失败 ✗")
            self.status_text.setStyleSheet(f"color: {C['danger']}; font-size: 10pt;")
            self.progress_label.setText("重建失败")
            self.progress_label.setStyleSheet(f"color: {C['danger']}; font-size: 10pt;")
            self._log(f"❌ {message}")

        self._refresh_cache_hint()

    def _append_log(self, msg):
        self.log_viewer.append_log(msg)

    def _log(self, msg):
        self.log_viewer.append_log(msg)

    def _update_progress(self, pct, text):
        self.progress_bar.setValue(pct)
        self.progress_label.setText(text)

    def _on_sfm_result(self, payload):
        try:
            self.trajectory_page.update_from_sfm(
                payload.get("poses", []),
                payload.get("xyz"),
                payload.get("keyframes"),
                payload.get("loop_closures"),
                payload.get("frame_status"),
            )
            # 只有真正跑完/最终结果才切到重建结果页
            if payload.get("switch_tab", True):
                self._switch_tab(2)
        except Exception as e:
            logger.warning("绘制重建结果失败: %s", e)

    # ---- 帧预览 ----

    def _show_preview_frames(self, frame_paths):
        self._preview_paths = list(frame_paths)
        self._preview_page = 0
        self._render_preview_page()

    def _preview_available_width(self) -> int:
        return max(0, self.thumb_scroll.viewport().width())

    def _preview_available_height(self) -> int:
        return max(0, self.thumb_scroll.viewport().height())

    def _compute_preview_cols(self) -> int:
        avail = self._preview_available_width()
        card_w = 192
        cols = max(1, (avail + 8) // card_w)
        return min(cols, 10)

    def _compute_preview_rows(self) -> int:
        avail = self._preview_available_height()
        card_h = 166
        rows = max(1, (avail + 8) // card_h)
        return min(rows, 10)

    def _clear_thumbnails(self):
        while self.thumb_layout.count():
            item = self.thumb_layout.takeAt(0)
            layout = item.layout() if item else None
            if layout is not None:
                while layout.count():
                    li = layout.takeAt(0)
                    w = li.widget() if li else None
                    if w is not None:
                        w.hide()
                        w.setParent(None)
                        w.deleteLater()

    def _preview_total_pages(self) -> int:
        n = len(self._preview_paths)
        return max(1, (n + self._preview_per_page - 1) // self._preview_per_page)

    def _render_preview_page(self):
        self._preview_cols = self._compute_preview_cols()
        self._preview_rows = self._compute_preview_rows()
        self._preview_per_page = self._preview_cols * self._preview_rows

        scroll_pos = self.thumb_scroll.verticalScrollBar().value()
        self._clear_thumbnails()

        n = len(self._preview_paths)
        total_pages = self._preview_total_pages()
        self._preview_page = max(0, min(self._preview_page, total_pages - 1))
        start = self._preview_page * self._preview_per_page
        end = min(start + self._preview_per_page, n)
        page_paths = self._preview_paths[start:end]

        for i, fp in enumerate(page_paths):
            global_idx = start + i
            if i % self._preview_cols == 0:
                row = QHBoxLayout()
                row.setSpacing(8)
                row.setAlignment(Qt.AlignCenter)
                self.thumb_layout.addLayout(row)

            try:
                img = cv2.imread(fp)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pix = QPixmap.fromImage(QImage(img.tobytes(), img.shape[1], img.shape[0], img.shape[1] * 3, QImage.Format_RGB888))
                pix = pix.scaled(180, 130, Qt.KeepAspectRatio, Qt.SmoothTransformation)

                lbl = QLabel()
                lbl.setPixmap(pix)
                lbl.setAlignment(Qt.AlignCenter)
                lbl.setStyleSheet(f"background-color: {C['bg']}; border: 1px solid {C['border_light']}; border-radius: 6px;")

                card = QWidget()
                cl = QVBoxLayout(card)
                cl.setContentsMargins(2, 2, 2, 2)
                cl.setSpacing(2)
                cl.addWidget(lbl)

                fl = StyledLabel(f"帧 {global_idx}", font_size=9, color=C["text_muted"])
                fl.setAlignment(Qt.AlignCenter)
                cl.addWidget(fl)

                card.setStyleSheet(f"background-color: {C['bg_card']}; border: 1px solid {C['border_light']}; border-radius: 8px;")
                row.addWidget(card)
            except Exception:
                pass

        self.prev_page_btn.setEnabled(self._preview_page > 0)
        self.next_page_btn.setEnabled(self._preview_page < total_pages - 1)
        if n > 0:
            self.page_label.setText(f"帧 {start}–{end - 1} / {n}  （第 {self._preview_page + 1}/{total_pages} 页，每页 {self._preview_per_page}）")
        else:
            self.page_label.setText("无帧可预览")
        self.thumb_scroll.verticalScrollBar().setValue(min(scroll_pos, self.thumb_scroll.verticalScrollBar().maximum()))

    def _preview_prev_page(self):
        if self._preview_page > 0:
            self._preview_page -= 1
            self._render_preview_page()

    def _preview_next_page(self):
        total_pages = self._preview_total_pages()
        if self._preview_page < total_pages - 1:
            self._preview_page += 1
            self._render_preview_page()

    def _schedule_preview_relayout(self):
        if not self._preview_paths:
            return
        if self._resize_timer is None:
            self._resize_timer = QTimer(self)
            self._resize_timer.setSingleShot(True)
            self._resize_timer.timeout.connect(self._on_preview_resize_timeout)
        self._resize_timer.start(150)

    def _on_preview_resize_timeout(self):
        new_cols = self._compute_preview_cols()
        new_rows = self._compute_preview_rows()
        if new_cols != self._preview_cols or new_rows != self._preview_rows:
            self._render_preview_page()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._schedule_preview_relayout()

    def closeEvent(self, event):
        if self.worker and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
        event.accept()


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------

def main():
    setup_logging()
    set_affinity_to_all_cores()
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("SfM Reconstruction")
    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()