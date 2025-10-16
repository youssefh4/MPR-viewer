"""
Slice view component for displaying medical image slices.
"""

import numpy as np
from skimage import measure
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QSlider, QSizePolicy
from PyQt5.QtCore import Qt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

OUTLINE_COLORS = {
    "lungs": "green",
    "heart_main": "red",
    "heart_vessels": "blue",
    "brain": "purple",
    "kidneys": "magenta",
    "liver": "orange",
    "spleen": "cyan",
    "spine": "yellow"
}

class SliceView(QWidget):
    def __init__(self, plane_name: str, get_slice_func, get_mask_func=None, get_color_func=None, label_top=""):
        super().__init__()
        self.plane = plane_name
        self.get_slice = get_slice_func
        self.get_mask = get_mask_func
        self.get_color = get_color_func
        self.top_label = label_top
        self.roi_zoom_enabled = False
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

    def set_roi_zoom(self, enabled: bool):
        self.roi_zoom_enabled = enabled

    def set_crosshair_position(self, axial_idx: int, coronal_idx: int, sagittal_idx: int, volume_shape):
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

    def update_slice(self, idx: int, show_mask: bool = False):
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

        if bbox and self.roi_zoom_enabled:
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
