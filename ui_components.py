"""
UI Components for the Medical MPR Viewer.
Includes custom widgets like CollapsibleBox and SliceView.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider, QSizePolicy, QToolButton, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from skimage import measure
import matplotlib.patches as patches

from config import OUTLINE_COLORS


class CollapsibleBox(QWidget):
    """A collapsible group box with smooth animation."""

    def __init__(self, title="", parent=None, collapsed=False):
        super(CollapsibleBox, self).__init__(parent)

        self.toggle_button = QToolButton()
        self.toggle_button.setText(title)
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(not collapsed)
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                background: transparent;
                font-weight: bold;
                font-size: 11px;
                padding: 5px;
                text-align: left;
            }
            QToolButton:hover {
                background-color: #e0e0e0;
            }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(5)
        self.content_layout.setContentsMargins(10, 5, 10, 10)
        self.content_area.setLayout(self.content_layout)
        self.content_area.setVisible(not collapsed)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 5)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

        # Style the whole box
        self.setStyleSheet("""
            CollapsibleBox {
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
        """)

    def toggle(self):
        """Toggle the collapsed/expanded state."""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        self.content_area.setVisible(checked)

    def addWidget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)


class SliceView(QWidget):
    canvas_clicked = pyqtSignal(int, int, str)  # x, y in display coords, plane name
    def __init__(self, plane_name, get_slice_func, get_mask_func=None, get_color_func=None, label_top=""):
        super().__init__()
        self.plane = plane_name
        self.get_slice = get_slice_func
        self.get_mask = get_mask_func
        self.get_color = get_color_func
        self.top_label = label_top
        self.roi_zoom_enabled = False
        self.manual_roi_enabled = False
        self.manual_roi_rect = None  # (xmin, ymin, xmax, ymax) in display coords
        self._rect_selector = None
        self.crosshair_enabled = True
        self.crosshair_pos = None
        self.volume_shape = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(5, 5, 5, 5)
        layout.setSpacing(3)

        self.title_label = QLabel(f"{self.top_label}{self.plane}")
        self.title_label.setMaximumHeight(25)
        layout.addWidget(self.title_label, 0)

        self.canvas = FigureCanvas(plt.Figure())
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.canvas.setMinimumSize(200, 200)
        layout.addWidget(self.canvas, 1)  # Stretch factor 1
        self.ax = self.canvas.figure.subplots()

        self.slice_label = QLabel("Slice: 0")
        self.slice_label.setMaximumHeight(20)
        layout.addWidget(self.slice_label, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumHeight(20)
        self.slider.setMaximumHeight(30)
        layout.addWidget(self.slider, 0)

        # Mouse click hookup
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)

    def _on_canvas_click(self, event):
        if event is None or event.inaxes is None or getattr(event, 'button', None) != 1:
            return
        try:
            x = int(event.xdata)
            y = int(event.ydata)
        except Exception:
            return
        self.canvas_clicked.emit(x, y, self.plane)

    def set_manual_roi_enabled(self, enabled: bool):
        self.manual_roi_enabled = enabled
        # Initialize or remove rectangle selector
        try:
            if enabled:
                if self._rect_selector is None:
                    self._rect_selector = RectangleSelector(
                        self.ax,
                        onselect=self._on_rect_selected,
                        drawtype='box',
                        useblit=False,
                        button=[1],
                        interactive=True,
                        minspanx=2,
                        minspany=2,
                        spancoords='data'
                    )
                self._rect_selector.set_active(True)
            else:
                if self._rect_selector is not None:
                    self._rect_selector.set_active(False)
        except Exception as e:
            print(f"Manual ROI error: {e}")
            pass

    def clear_manual_roi(self):
        self.manual_roi_rect = None
        try:
            if self._rect_selector is not None:
                self._rect_selector.set_active(False)
                # Clear any existing rectangle
                for artist in self.ax.collections + self.ax.patches:
                    if hasattr(artist, '_manual_roi'):
                        artist.remove()
        except Exception:
            pass
        self.canvas.draw_idle()

    def _on_rect_selected(self, eclick, erelease):
        # Store rectangle in display coordinates
        try:
            xmin = min(eclick.xdata, erelease.xdata)
            xmax = max(eclick.xdata, erelease.xdata)
            ymin = min(eclick.ydata, erelease.ydata)
            ymax = max(eclick.ydata, erelease.ydata)
            self.manual_roi_rect = (xmin, ymin, xmax, ymax)
            print(f"Manual ROI selected: {self.manual_roi_rect}")
            self.canvas.draw_idle()
        except Exception as e:
            print(f"ROI selection error: {e}")
            pass

    def set_dark_mode(self, enabled: bool):
        """Apply dark or light styling to the view canvas and labels."""
        bg = '#111111' if enabled else 'white'
        fg = '#e0e0e0' if enabled else 'black'
        grid = '#333333' if enabled else '#444444'

        self.ax.set_facecolor(bg)
        if self.canvas.figure is not None:
            self.canvas.figure.set_facecolor(bg)
        self.title_label.setStyleSheet(f"color: {fg};")
        self.slice_label.setStyleSheet(f"color: {fg};")
        try:
            for spine in self.ax.spines.values():
                spine.set_color(grid)
        except Exception:
            pass
        self.canvas.setStyleSheet(f"background-color: {bg};")
        self.canvas.draw_idle()

    def set_roi_zoom(self, enabled):
        self.roi_zoom_enabled = enabled

    def set_crosshair_position(self, axial_idx, coronal_idx, sagittal_idx, volume_shape):
        self.crosshair_pos = (axial_idx, coronal_idx, sagittal_idx)
        self.volume_shape = volume_shape

    def draw_crosshairs(self, img_shape):
        if not self.crosshair_enabled or self.crosshair_pos is None or self.volume_shape is None:
            return

        axial_idx, coronal_idx, sagittal_idx = self.crosshair_pos
        h_rot, w_rot = img_shape
        vol_axial, vol_coronal, vol_sagittal = self.volume_shape

        if self.plane == "Axial":
            v_pos = vol_sagittal - 1 - sagittal_idx
            self.ax.axhline(y=v_pos, color='yellow', linewidth=1.5, linestyle='-', alpha=0.8)
            h_pos = coronal_idx
            self.ax.axvline(x=h_pos, color='cyan', linewidth=1.5, linestyle='-', alpha=0.8)

        elif self.plane == "Coronal":
            v_pos = vol_sagittal - 1 - sagittal_idx
            self.ax.axhline(y=v_pos, color='yellow', linewidth=1.5, linestyle='-', alpha=0.8)
            h_pos = axial_idx
            self.ax.axvline(x=h_pos, color='red', linewidth=1.5, linestyle='-', alpha=0.8)

        elif self.plane == "Sagittal":
            v_pos = vol_coronal - 1 - coronal_idx
            self.ax.axhline(y=v_pos, color='cyan', linewidth=1.5, linestyle='-', alpha=0.8)
            h_pos = axial_idx
            self.ax.axvline(x=h_pos, color='red', linewidth=1.5, linestyle='-', alpha=0.8)

    def update_slice(self, idx, show_mask=False):
        img = self.get_slice(idx)
        self.slice_label.setText(f"Slice: {idx}")
        self.ax.clear()
        self.ax.imshow(np.rot90(img), cmap="gray")
        self.ax.axis("off")

        mask = self.get_mask(idx) if self.get_mask else None
        bbox = None
        if mask is not None and np.any(mask):
            bbox = self.get_slice_bounding_box(mask)

        if show_mask and mask is not None and np.any(mask):
            try:
                color_mask = self.get_color(idx) if self.get_color else None
                for c in measure.find_contours(np.rot90(mask), 0.5):
                    self.ax.plot(c[:, 1], c[:, 0], color="white", linewidth=1.2)
                if isinstance(color_mask, dict):
                    for name, arr in color_mask.items():
                        if arr is not None and np.any(arr):
                            col = OUTLINE_COLORS.get(name, "red")
                            for c in measure.find_contours(np.rot90(arr), 0.5):
                                self.ax.plot(c[:, 1], c[:, 0], color=col, linewidth=1.4)
            except Exception as e:
                print(f"[Mask display error {self.plane}] {e}")

        # Manual ROI zoom takes precedence when enabled
        if self.manual_roi_enabled and self.manual_roi_rect is not None and self.roi_zoom_enabled:
            xmin, ymin, xmax, ymax = self.manual_roi_rect
            img_h, img_w = np.rot90(img).shape
            xmin = int(np.clip(xmin, 0, img_w - 1))
            xmax = int(np.clip(xmax, 0, img_w - 1))
            ymin = int(np.clip(ymin, 0, img_h - 1))
            ymax = int(np.clip(ymax, 0, img_h - 1))
            if xmax > xmin and ymax > ymin:
                self.ax.set_xlim(xmin, xmax)
                self.ax.set_ylim(ymax, ymin)
        elif bbox and self.roi_zoom_enabled:
            (xmin, ymin), (xmax, ymax) = bbox
            box_w = xmax - xmin
            box_h = ymax - ymin
            margin_x = max(int(0.08 * box_w), 4)
            margin_y = max(int(0.08 * box_h), 4)
            draw_xmin = max(0, xmin - margin_x)
            draw_xmax = min(mask.shape[1], xmax + margin_x)
            draw_ymin = max(0, ymin - margin_y)
            draw_ymax = min(mask.shape[0], ymax + margin_y)
            rect = patches.Rectangle(
                (draw_xmin, draw_ymin),
                draw_xmax - draw_xmin,
                draw_ymax - draw_ymin,
                edgecolor='yellow', facecolor='none', linewidth=2
            )
            self.ax.add_patch(rect)
            img_h, img_w = mask.shape[0], mask.shape[1]
            scale = 0.8
            zoom_width = (draw_xmax - draw_xmin) / scale
            zoom_height = (draw_ymax - draw_ymin) / scale
            x_center = (draw_xmin + draw_xmax) // 2
            y_center = (draw_ymin + draw_ymax) // 2
            xlim_low = max(0, int(x_center - zoom_width // 2))
            xlim_high = min(img_w, int(x_center + zoom_width // 2))
            ylim_low = max(0, int(y_center - zoom_height // 2))
            ylim_high = min(img_h, int(y_center + zoom_height // 2))
            self.ax.set_xlim(xlim_low, xlim_high)
            self.ax.set_ylim(ylim_high, ylim_low)

        self.draw_crosshairs(np.rot90(img).shape)
        self.canvas.draw()

    @staticmethod
    def get_slice_bounding_box(mask_slice):
        mask_slice = np.rot90(mask_slice)
        rows = np.any(mask_slice, axis=1)
        cols = np.any(mask_slice, axis=0)
        if not np.any(rows) or not np.any(cols):
            return None
        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]
        return (cmin, rmin), (cmax, rmax)
