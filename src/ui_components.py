"""
UI Components for the Medical MPR Viewer.
Includes custom widgets like CollapsibleBox and SliceView.
"""

import numpy as np
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QSizePolicy, QToolButton, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, QRect
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
        
        # Enhanced typography
        font = QFont()
        font.setFamily("Segoe UI")
        font.setPointSize(10)
        font.setWeight(QFont.Bold)
        self.toggle_button.setFont(font)
        
        self.toggle_button.setStyleSheet("""
            QToolButton {
                border: none;
                padding: 8px;
                text-align: left;
                font-weight: bold;
                background-color: #f8f9fa;
                border-radius: 4px;
            }
            QToolButton:hover {
                background-color: #e9ecef;
            }
            QToolButton:pressed {
                background-color: #dee2e6;
            }
        """)
        self.toggle_button.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle_button.setArrowType(Qt.DownArrow if not collapsed else Qt.RightArrow)
        self.toggle_button.clicked.connect(self.toggle)

        self.content_area = QWidget()
        self.content_layout = QVBoxLayout()
        self.content_layout.setSpacing(4)
        self.content_layout.setContentsMargins(8, 6, 8, 8)
        self.content_area.setLayout(self.content_layout)
        
        # Animation setup
        self.animation = QPropertyAnimation(self.content_area, b"maximumHeight")
        self.animation.setDuration(200)
        self.animation.setEasingCurve(QEasingCurve.OutQuad)
        
        # Set initial state
        if collapsed:
            self.content_area.setMaximumHeight(0)
            self.content_area.setVisible(False)
        else:
            self.content_area.setVisible(True)

        # Main layout
        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(0)
        main_layout.setContentsMargins(0, 0, 0, 5)
        main_layout.addWidget(self.toggle_button)
        main_layout.addWidget(self.content_area)

    def toggle(self):
        """Toggle the collapsed/expanded state with smooth animation."""
        checked = self.toggle_button.isChecked()
        self.toggle_button.setArrowType(Qt.DownArrow if checked else Qt.RightArrow)
        
        # Simple toggle without animation to avoid conflicts with dropdowns
        self.content_area.setVisible(checked)

    def addWidget(self, widget):
        """Add a widget to the content area."""
        self.content_layout.addWidget(widget)

    def addLayout(self, layout):
        """Add a layout to the content area."""
        self.content_layout.addLayout(layout)
        
    def set_dark_mode(self, enabled):
        """Apply dark mode styling to the collapsible box."""
        if enabled:
            self.toggle_button.setStyleSheet("""
                QToolButton {
                    border: none;
                    padding: 6px;
                    text-align: left;
                    font-weight: bold;
                    font-size: 11px;
                    background-color: #2d2d2d;
                    border-radius: 6px;
                    color: #ffffff;
                }
                QToolButton:hover {
                    background-color: #3a3a3a;
                }
                QToolButton:pressed {
                    background-color: #1a1a1a;
                }
            """)
            self.setStyleSheet("""
                CollapsibleBox {
                    background-color: #1e1e1e;
                    border: 1px solid #404040;
                    border-radius: 8px;
                    margin: 4px;
                }
            """)
        else:
            self.toggle_button.setStyleSheet("""
                QToolButton {
                    border: none;
                    padding: 6px;
                    text-align: left;
                    font-weight: bold;
                    font-size: 11px;
                    background-color: #f8f9fa;
                    border-radius: 6px;
                }
                QToolButton:hover {
                    background-color: #e9ecef;
                }
                QToolButton:pressed {
                    background-color: #dee2e6;
                }
            """)
            self.setStyleSheet("""
                CollapsibleBox {
                    background-color: #f8f9fa;
                    border: 1px solid #d0d7de;
                    border-radius: 8px;
                    margin: 4px;
                }
            """)


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
        self.click_mode = "crosshair"  # "crosshair" or "roi"
        self.roi_start_point = None  # Store the starting point for rectangular ROI
        self.roi_drawing = False  # Track if we're currently drawing ROI
        self.is_dark_mode = False  # Track dark mode state

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
        
        # Set initial background colors
        self.canvas.figure.patch.set_facecolor('white')
        self.ax.set_facecolor('white')

        self.slice_label = QLabel("Slice: 0")
        self.slice_label.setMaximumHeight(20)
        layout.addWidget(self.slice_label, 0)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimumHeight(20)
        self.slider.setMaximumHeight(30)
        layout.addWidget(self.slider, 0)

        # Mouse event hookups
        self.canvas.mpl_connect('button_press_event', self._on_canvas_click)
        self.canvas.mpl_connect('motion_notify_event', self._on_mouse_motion)
        self.canvas.mpl_connect('button_release_event', self._on_mouse_release)

    def _on_canvas_click(self, event):
        if event is None or event.inaxes is None or getattr(event, 'button', None) != 1:
            return
        
        # Handle click based on current mode
        if self.click_mode == "roi":
            # Start rectangular ROI drawing
            try:
                x = int(event.xdata)
                y = int(event.ydata)
                self.roi_start_point = (x, y)
                self.roi_drawing = True
                print(f"ROI drawing started at: {self.roi_start_point}")
            except Exception as e:
                print(f"ROI start error: {e}")
        else:
            # Crosshair mode - emit click event
            try:
                x = int(event.xdata)
                y = int(event.ydata)
            except Exception:
                return
            self.canvas_clicked.emit(x, y, self.plane)

    def _on_mouse_motion(self, event):
        """Handle mouse motion during ROI drawing."""
        if not self.roi_drawing or event is None or event.inaxes is None:
            return
        
        if self.click_mode == "roi" and self.roi_start_point is not None:
            try:
                x = int(event.xdata)
                y = int(event.ydata)
                # Draw preview rectangle
                self._draw_roi_preview(self.roi_start_point, (x, y))
            except Exception as e:
                print(f"ROI motion error: {e}")

    def _on_mouse_release(self, event):
        """Handle mouse release to complete ROI drawing."""
        if not self.roi_drawing or event is None or event.inaxes is None:
            return
        
        if self.click_mode == "roi" and self.roi_start_point is not None:
            try:
                x = int(event.xdata)
                y = int(event.ydata)
                
                # Create final ROI rectangle
                xmin = min(self.roi_start_point[0], x)
                xmax = max(self.roi_start_point[0], x)
                ymin = min(self.roi_start_point[1], y)
                ymax = max(self.roi_start_point[1], y)
                
                # Ensure minimum size
                if xmax - xmin < 5 or ymax - ymin < 5:
                    print("ROI too small, ignoring")
                    self.roi_drawing = False
                    self.roi_start_point = None
                    return
                
                self.manual_roi_rect = (xmin, ymin, xmax, ymax)
                print(f"ROI completed: {self.manual_roi_rect}")
                
                # Clear preview and draw final ROI
                self._clear_roi_preview()
                self.canvas.draw_idle()
                
            except Exception as e:
                print(f"ROI completion error: {e}")
            finally:
                self.roi_drawing = False
                self.roi_start_point = None

    def _draw_roi_preview(self, start_point, current_point):
        """Draw a preview rectangle during ROI drawing."""
        try:
            # Clear previous preview
            self._clear_roi_preview()
            
            # Draw new preview rectangle
            xmin = min(start_point[0], current_point[0])
            xmax = max(start_point[0], current_point[0])
            ymin = min(start_point[1], current_point[1])
            ymax = max(start_point[1], current_point[1])
            
            from matplotlib.patches import Rectangle
            preview_rect = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                  linewidth=2, edgecolor='red', facecolor='none', linestyle='--')
            self.ax.add_patch(preview_rect)
            self.canvas.draw_idle()
        except Exception as e:
            print(f"Preview drawing error: {e}")

    def _clear_roi_preview(self):
        """Clear ROI preview rectangles."""
        try:
            # Remove all preview rectangles (patches with dashed lines)
            patches_to_remove = []
            for patch in self.ax.patches:
                if hasattr(patch, 'get_linestyle') and patch.get_linestyle() == '--':
                    patches_to_remove.append(patch)
            
            for patch in patches_to_remove:
                patch.remove()
        except Exception as e:
            print(f"Preview clear error: {e}")

    def set_click_mode(self, mode: str):
        """Set the click mode: 'crosshair' or 'roi'."""
        self.click_mode = mode
        if mode == "roi":
            self.title_label.setText(f"{self.top_label}{self.plane} (Click & Drag for ROI)")
        else:
            self.title_label.setText(f"{self.top_label}{self.plane}")

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
                # Update title to show manual ROI is active
                self.title_label.setText(f"{self.top_label}{self.plane} (Manual ROI: Click & Drag)")
            else:
                if self._rect_selector is not None:
                    self._rect_selector.set_active(False)
                # Restore normal title
                self.title_label.setText(f"{self.top_label}{self.plane}")
        except Exception as e:
            print(f"Manual ROI error: {e}")
            pass

    def clear_manual_roi(self):
        self.manual_roi_rect = None
        self.roi_drawing = False
        self.roi_start_point = None
        try:
            if self._rect_selector is not None:
                self._rect_selector.set_active(False)
                # Clear any existing rectangle
                for artist in self.ax.collections + self.ax.patches:
                    if hasattr(artist, '_manual_roi'):
                        artist.remove()
            
            # Clear preview rectangles
            self._clear_roi_preview()
            
            # Clear any existing ROI patches
            patches_to_remove = []
            for patch in self.ax.patches:
                if hasattr(patch, 'get_edgecolor') and patch.get_edgecolor() == 'red':
                    patches_to_remove.append(patch)
            
            for patch in patches_to_remove:
                patch.remove()
                
        except Exception as e:
            print(f"Clear ROI error: {e}")
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

    def set_dark_mode(self, enabled):
        """Apply dark or light styling to the view canvas and labels."""
        self.is_dark_mode = enabled
        bg = '#1e1e1e' if enabled else 'white'
        fg = '#ffffff' if enabled else '#2c3e50'
        
        # Update matplotlib canvas background
        self.ax.set_facecolor(bg)
        if self.canvas.figure is not None:
            self.canvas.figure.set_facecolor(bg)
            
        # Update label styling
        self.title_label.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                font-weight: bold;
                font-size: 12px;
                padding: 2px;
            }}
        """)
        self.slice_label.setStyleSheet(f"""
            QLabel {{
                color: {fg};
                font-size: 11px;
                padding: 2px;
            }}
        """)
        
        # Update slider styling
        self.slider.setStyleSheet(f"""
            QSlider::groove:horizontal {{
                border: 1px solid {'#404040' if enabled else '#bdc3c7'};
                height: 6px;
                background: {'#2d2d2d' if enabled else '#ecf0f1'};
                border-radius: 3px;
            }}
            QSlider::handle:horizontal {{
                background: {'#0d7377' if enabled else '#3498db'};
                border: 2px solid #ffffff;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }}
        """)
        
        # Update canvas styling
        self.canvas.setStyleSheet(f"background-color: {bg}; border: 1px solid {'#404040' if enabled else '#cccccc'}; border-radius: 6px;")
        
        # Force redraw
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
        try:
            img = self.get_slice(idx)
            self.slice_label.setText(f"Slice: {idx}")
            
            # Store ROI rectangle before clearing
            roi_rect = None
            if self.manual_roi_rect is not None:
                xmin, ymin, xmax, ymax = self.manual_roi_rect
                roi_rect = (xmin, ymin, xmax, ymax)
            
            self.ax.clear()
            # Set the axis background color based on dark mode
            bg_color = '#1e1e1e' if self.is_dark_mode else 'white'
            self.ax.set_facecolor(bg_color)
            
            # Use appropriate colormap
            cmap = "gray"
            self.ax.imshow(np.rot90(img), cmap=cmap)
            self.ax.axis("off")
        except (IndexError, ValueError) as e:
            print(f"Error updating slice {idx} in {self.plane}: {e}")
            # Try to get a valid slice by clamping the index
            try:
                # This is a fallback - the main fix should be in mpr_viewer.py
                if hasattr(self, 'slider') and self.slider:
                    max_idx = self.slider.maximum()
                    clamped_idx = max(0, min(idx, max_idx))
                    img = self.get_slice(clamped_idx)
                    self.slice_label.setText(f"Slice: {clamped_idx} (clamped from {idx})")
                    self.ax.clear()
                    # Set the axis background color
                    bg_color = '#1e1e1e' if self.is_dark_mode else 'white'
                    self.ax.set_facecolor(bg_color)
                    self.ax.imshow(np.rot90(img), cmap="gray")
                    self.ax.axis("off")
                else:
                    return
            except Exception:
                print(f"Failed to recover from slice error in {self.plane}")
                return

        # Restore ROI rectangle if it exists
        if roi_rect is not None:
            from matplotlib.patches import Rectangle
            xmin, ymin, xmax, ymax = roi_rect
            roi_patch = Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                linewidth=2, edgecolor='red', facecolor='none')
            self.ax.add_patch(roi_patch)

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
        if self.click_mode == "roi" and self.manual_roi_rect is not None and self.roi_zoom_enabled:
            xmin, ymin, xmax, ymax = self.manual_roi_rect
            img_h, img_w = np.rot90(img).shape
            xmin = int(np.clip(xmin, 0, img_w - 1))
            xmax = int(np.clip(xmax, 0, img_w - 1))
            ymin = int(np.clip(ymin, 0, img_h - 1))
            ymax = int(np.clip(ymax, 0, img_h - 1))
            if xmax > xmin and ymax > ymin:
                # Add margin like automatic ROI
                box_w = xmax - xmin
                box_h = ymax - ymin
                margin_x = max(int(0.08 * box_w), 4)
                margin_y = max(int(0.08 * box_h), 4)
                draw_xmin = max(0, xmin - margin_x)
                draw_xmax = min(img_w, xmax + margin_x)
                draw_ymin = max(0, ymin - margin_y)
                draw_ymax = min(img_h, ymax + margin_y)
                self.ax.set_xlim(draw_xmin, draw_xmax)
                self.ax.set_ylim(draw_ymax, draw_ymin)
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
