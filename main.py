"""
Medical MPR Viewer - DICOM & NIfTI Multi-Planar Reconstruction Viewer

Features:
- Load DICOM series and NIfTI files
- Multi-planar reconstruction (Axial, Sagittal, Coronal views)
- Oblique slicing with 3D rotations
- TotalSegmentator integration for automatic organ segmentation
- External mask loading support
- Crosshair navigation across views
- ROI zoom functionality
- Automatic playback with play/pause, speed control, and direction control
- Collapsible sidebar sections for organized, clutter-free interface
- Professional GUI with modern controls
"""

import sys
import os
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from skimage import measure
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QMessageBox, QComboBox, QHBoxLayout,
    QCheckBox, QGroupBox, QSizePolicy, QScrollArea, QFrame, QSpinBox,
    QToolButton, QDialog
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QFont
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.pyplot as plt
from totalsegmentator.python_api import totalsegmentator
import matplotlib.patches as patches

# Import our modular components
from ai.orientation import detect_orientation_ai
from utils.image_utils import match_shape, detect_main_plane_dicom
from utils.geometry import create_rotation_matrix, get_oblique_slice
from ui.slice_view import SliceView
from ui.collapsible_box import CollapsibleBox
from ui.dialogs import SliceRangeDialog
from data_io.export import export_slices
from data_io.dicom_loader import load_dicom_series
from data_io.nifti_loader import load_nifti_file

# Organ groups for TotalSegmentator - used in both detection and masking
ORGAN_GROUPS_SIMPLE = {
    "Lungs": ["lung_upper_lobe_left", "lung_upper_lobe_right", "lung_middle_lobe_right",
              "lung_lower_lobe_left", "lung_lower_lobe_right", "trachea", "airways"],
    "Heart": ["heart", "aorta", "pulmonary_artery", "pulmonary_vein"],
    "Brain": ["brain", "cerebellum", "brainstem"],
    "Kidneys": ["kidney_left", "kidney_right"],
    "Liver": ["liver"],
    "Spleen": ["spleen"],
    "Spine": ["spinal_cord"] + [f"vertebrae_{v}" for v in [
        "C1","C2","C3","C4","C5","C6","C7",
        "T1","T2","T3","T4","T5","T6","T7","T8","T9","T10","T11","T12",
        "L1","L2","L3","L4","L5","S1"
    ]] + ["sacrum"]
}

# Heart organ with color separation (for prepare_masks)
HEART_COLOR_GROUPS = {"main": ["heart"], "vessels": ["aorta", "pulmonary_artery", "pulmonary_vein"]}

# Organ keywords for external mask detection
ORGAN_KEYWORDS = {
    "Lungs": ["lung", "trachea", "airway"],
    "Heart": ["heart", "aorta", "pulmonary"],
    "Brain": ["brain", "cerebellum", "brainstem"],
    "Kidneys": ["kidney"],
    "Liver": ["liver"],
    "Spleen": ["spleen"],
    "Spine": ["vertebra", "spinal", "sacrum"]
}

class DICOM_MPR_Viewer(QWidget):
    def __init__(self):
        """Initialize the MPR viewer with sidebar controls and image grid."""
        super().__init__()
        self._init_window_settings()
        self._init_variables()
        self._create_ui()

    def _init_window_settings(self):
        """Set up window title, size, and geometry."""
        self.setWindowTitle("Medical MPR Viewer - DICOM & NIfTI with TotalSegmentator")
        self.setGeometry(60, 60, 1800, 1200)
        self.setMinimumSize(1000, 700)

    def _init_variables(self):
        """Initialize all instance variables for data and state management."""
        # Volume and mask data
        self.volume = None
        self.original_volume = None  # Store original before reorientation
        self.output_dir = "totalsegmentator_output"
        self.mask_volume = None
        self.color_masks = None
        
        # Plane and viewing settings
        self.main_plane = "Axial"
        self.detected_plane = "Axial"  # Store AI-detected plane
        self.overlay_on = False
        self.body_part_examined = "Unknown"
        self.segmentation_plane = "Axial"
        
        # External masks
        self.external_mask_volume = None
        self.external_color_masks = None
        self.using_external_masks = False
        self.external_masks_dir = None

        # Oblique view settings
        self.oblique_rotation_x = 0
        self.oblique_rotation_y = 0
        self.oblique_rotation_z = 0
        self.oblique_reference_plane = "Axial"
        self.fourth_view_mode = "Segmentation"

        # View state
        self.slice_indices = {"Axial": 0, "Coronal": 0, "Sagittal": 0, "Oblique": 0, "Segmentation": 0}
        self.roi_zoom_on = False
        self.views = []
        self.views_dict = {}
        self.oblique_view = None
        self.export_btn = None
        
        # Playback controls
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_slice)
        self.is_playing = False
        self.playback_speed = 100  # milliseconds between frames
        self.playback_view = "Axial"  # Which view to animate
        self.playback_direction = 1  # 1 for forward, -1 for backward
        self.loop_playback = True  # Whether to loop at the end

    def _create_ui(self):
        """Create the main user interface layout."""
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Create and add sidebar
        sidebar = self._create_sidebar()
        main_layout.addWidget(sidebar)
        
        # Create and add image grid area
        right_container = self._create_image_grid_container()
        main_layout.addWidget(right_container, 1)

    def _create_sidebar(self):
        """Create the left sidebar with all control panels."""
        sidebar = QScrollArea()
        sidebar.setWidgetResizable(True)
        sidebar.setMaximumWidth(280)
        sidebar.setMinimumWidth(280)
        sidebar.setStyleSheet("""
            QScrollArea {
                background-color: #f5f5f5;
                border-right: 1px solid #cccccc;
            }
        """)
        
        sidebar_content = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_content)
        sidebar_layout.setSpacing(10)
        sidebar_layout.setContentsMargins(10, 10, 10, 10)
        
        # Add all control groups
        sidebar_layout.addWidget(self._create_load_data_group())
        sidebar_layout.addWidget(self._create_segmentation_group())
        sidebar_layout.addWidget(self._create_display_options_group())
        sidebar_layout.addWidget(self._create_playback_controls_group())
        sidebar_layout.addWidget(self._create_oblique_controls_group())
        sidebar_layout.addWidget(self._create_export_group())
        sidebar_layout.addStretch()
        
        sidebar.setWidget(sidebar_content)
        return sidebar

    def _create_load_data_group(self):
        """Create the Load Data control group."""
        load_group = CollapsibleBox("📁 Load Data & Information")
        
        btn_load_dicom = QPushButton("Load DICOM Series")
        btn_load_dicom.clicked.connect(self.load_dicom)
        load_group.addWidget(btn_load_dicom)
        
        btn_load_nifti = QPushButton("Load NIfTI File")
        btn_load_nifti.clicked.connect(self.load_nifti)
        load_group.addWidget(btn_load_nifti)
        
        btn_load_masks = QPushButton("Load Segmentation Masks")
        btn_load_masks.clicked.connect(self.load_external_masks)
        load_group.addWidget(btn_load_masks)
        
        # Add separator line
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("QFrame { color: #cccccc; }")
        load_group.addWidget(separator)
        
        # Add information labels
        self.info_label = QLabel("No data loaded.")
        self.info_label.setWordWrap(True)
        self.info_label.setStyleSheet("QLabel { font-size: 12px; color: #555; font-weight: 500; }")
        load_group.addWidget(self.info_label)
        
        self.bodypart_label = QLabel("Type: Unknown")
        self.bodypart_label.setWordWrap(True)
        self.bodypart_label.setStyleSheet("QLabel { font-size: 12px; color: #555; font-weight: 500; }")
        load_group.addWidget(self.bodypart_label)
        
        return load_group

    def _create_segmentation_group(self):
        """Create the Segmentation control group."""
        seg_group = CollapsibleBox("🔬 Segmentation")
        
        btn_run_seg = QPushButton("Run TotalSegmentator")
        btn_run_seg.clicked.connect(self.run_segmentation)
        btn_run_seg.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        seg_group.addWidget(btn_run_seg)
        
        seg_group.addWidget(QLabel("Organ:"))
        self.organ_dropdown = QComboBox()
        self.organ_dropdown.addItems(["None", "Lungs", "Heart", "Brain", "Kidneys", "Liver", "Spleen", "Spine"])
        self.organ_dropdown.currentIndexChanged.connect(self.prepare_masks)
        seg_group.addWidget(self.organ_dropdown)
        
        seg_group.addWidget(QLabel("View Plane:"))
        self.segplane_dropdown = QComboBox()
        self.segplane_dropdown.addItems(["Axial", "Coronal", "Sagittal"])
        self.segplane_dropdown.currentIndexChanged.connect(self.change_segmentation_plane)
        seg_group.addWidget(self.segplane_dropdown)
        
        return seg_group

    def _create_display_options_group(self):
        """Create the Display Options control group."""
        display_group = CollapsibleBox("👁 Display Options")
        
        self.overlay_btn = QPushButton("Toggle Overlay: OFF")
        self.overlay_btn.clicked.connect(self.toggle_overlay)
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setStyleSheet("QPushButton:checked { background-color: #2196F3; color: white; }")
        display_group.addWidget(self.overlay_btn)
        
        self.roi_btn = QPushButton("Toggle ROI Zoom: OFF")
        self.roi_btn.clicked.connect(self.toggle_roi_zoom)
        self.roi_btn.setCheckable(True)
        self.roi_btn.setStyleSheet("QPushButton:checked { background-color: #2196F3; color: white; }")
        display_group.addWidget(self.roi_btn)
        
        # Fourth view selector
        display_group.addWidget(QLabel("4th View:"))
        self.fourth_view_dropdown = QComboBox()
        self.fourth_view_dropdown.addItems(["Segmentation", "Oblique"])
        self.fourth_view_dropdown.currentIndexChanged.connect(self.switch_fourth_view)
        display_group.addWidget(self.fourth_view_dropdown)
        
        return display_group

    def _create_playback_controls_group(self):
        """Create the Playback Controls group."""
        playback_group = CollapsibleBox("▶ Playback Controls")
        
        # Play/Pause button
        self.play_pause_btn = QPushButton("▶ Play")
        self.play_pause_btn.clicked.connect(self.toggle_playback)
        self.play_pause_btn.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
        playback_group.addWidget(self.play_pause_btn)
        
        # View selection
        playback_group.addWidget(QLabel("Animate View:"))
        self.playback_view_dropdown = QComboBox()
        self.playback_view_dropdown.addItems(["Axial", "Coronal", "Sagittal", "Segmentation", "Oblique"])
        self.playback_view_dropdown.currentTextChanged.connect(self.change_playback_view)
        playback_group.addWidget(self.playback_view_dropdown)
        
        # Speed control
        speed_layout = QVBoxLayout()
        speed_layout.setSpacing(2)
        self.speed_label = QLabel("Speed (FPS): 10")
        speed_layout.addWidget(self.speed_label)
        self.speed_slider = QSlider(Qt.Horizontal)
        self.speed_slider.setMinimum(1)
        self.speed_slider.setMaximum(60)
        self.speed_slider.setValue(10)
        self.speed_slider.valueChanged.connect(self.update_playback_speed)
        speed_layout.addWidget(self.speed_slider)
        playback_group.addLayout(speed_layout)
        
        # Direction and loop controls
        controls_layout = QHBoxLayout()
        self.direction_btn = QPushButton("Forward ▶")
        self.direction_btn.clicked.connect(self.toggle_direction)
        self.direction_btn.setCheckable(True)
        controls_layout.addWidget(self.direction_btn)
        
        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.setChecked(True)
        self.loop_checkbox.stateChanged.connect(self.toggle_loop)
        controls_layout.addWidget(self.loop_checkbox)
        playback_group.addLayout(controls_layout)
        
        return playback_group

    def _create_oblique_controls_group(self):
        """Create the Oblique View Controls group."""
        oblique_group = CollapsibleBox("🔄 Oblique View Controls", collapsed=True)
        
        oblique_group.addWidget(QLabel("Reference Plane:"))
        self.oblique_ref_dropdown = QComboBox()
        self.oblique_ref_dropdown.addItems(["Axial", "Coronal", "Sagittal"])
        self.oblique_ref_dropdown.currentIndexChanged.connect(self.change_oblique_reference)
        oblique_group.addWidget(self.oblique_ref_dropdown)
        
        # Rotation sliders
        rot_x_layout = QVBoxLayout()
        rot_x_layout.setSpacing(2)
        self.rot_x_label = QLabel("Rotation X: 0°")
        rot_x_layout.addWidget(self.rot_x_label)
        self.rot_x_slider = QSlider(Qt.Horizontal)
        self.rot_x_slider.setMinimum(-180)
        self.rot_x_slider.setMaximum(180)
        self.rot_x_slider.setValue(0)
        self.rot_x_slider.valueChanged.connect(self.update_rotation_x)
        rot_x_layout.addWidget(self.rot_x_slider)
        oblique_group.addLayout(rot_x_layout)
        
        rot_y_layout = QVBoxLayout()
        rot_y_layout.setSpacing(2)
        self.rot_y_label = QLabel("Rotation Y: 0°")
        rot_y_layout.addWidget(self.rot_y_label)
        self.rot_y_slider = QSlider(Qt.Horizontal)
        self.rot_y_slider.setMinimum(-180)
        self.rot_y_slider.setMaximum(180)
        self.rot_y_slider.setValue(0)
        self.rot_y_slider.valueChanged.connect(self.update_rotation_y)
        rot_y_layout.addWidget(self.rot_y_slider)
        oblique_group.addLayout(rot_y_layout)
        
        rot_z_layout = QVBoxLayout()
        rot_z_layout.setSpacing(2)
        self.rot_z_label = QLabel("Rotation Z: 0°")
        rot_z_layout.addWidget(self.rot_z_label)
        self.rot_z_slider = QSlider(Qt.Horizontal)
        self.rot_z_slider.setMinimum(-180)
        self.rot_z_slider.setMaximum(180)
        self.rot_z_slider.setValue(0)
        self.rot_z_slider.valueChanged.connect(self.update_rotation_z)
        rot_z_layout.addWidget(self.rot_z_slider)
        oblique_group.addLayout(rot_z_layout)
        
        reset_btn = QPushButton("Reset Rotations")
        reset_btn.clicked.connect(self.reset_oblique_rotations)
        oblique_group.addWidget(reset_btn)
        
        return oblique_group

    def _create_export_group(self):
        """Create the Export control group."""
        export_group = CollapsibleBox("🗄 Export", collapsed=True)
        
        self.export_btn = QPushButton("Export Organ Slices")
        self.export_btn.clicked.connect(self.export_organ_slices)
        export_group.addWidget(self.export_btn)
        
        return export_group

    def _create_image_grid_container(self):
        """Create the right-side container for the image grid."""
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(5, 5, 5, 5)
        
        self.grid = QGridLayout()
        self.grid.setSpacing(8)
        self.grid.setContentsMargins(0, 0, 0, 0)
        right_layout.addLayout(self.grid, 1)
        
        return right_container

    def load_dicom(self):
        """Load DICOM series from folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
            
        try:
            print(f"Loading DICOM from folder: {folder}")
            self.volume, detected_plane, body_part = load_dicom_series(folder)
            self.original_volume = self.volume.copy()
            self.detected_plane = detected_plane
            self.body_part_examined = body_part
            
            print(f"Successfully loaded DICOM: shape={self.volume.shape}, plane={detected_plane}, body_part={body_part}")
            print(f"Volume orientation: Z={self.volume.shape[0]}, Y={self.volume.shape[1]}, X={self.volume.shape[2]}")
            
            # Update UI
            self.info_label.setText(f"Loaded: {self.volume.shape[0]} slices")
            self.bodypart_label.setText(f"Type: {body_part}")
            
            # Create views
            self.create_views()
            self.prepare_masks()
            
        except Exception as e:
            error_msg = f"Failed to load DICOM: {str(e)}"
            print(f"DICOM loading error: {error_msg}")
            QMessageBox.critical(self, "Error", error_msg)

    def load_nifti(self):
        """Load NIfTI file."""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select NIfTI File", "", "NIfTI Files (*.nii *.nii.gz)"
        )
        if not file_path:
            return
            
        try:
            self.volume, detected_plane, body_part = load_nifti_file(file_path)
            self.original_volume = self.volume.copy()
            self.detected_plane = detected_plane
            self.body_part_examined = body_part
            
            # Update UI
            self.info_label.setText(f"Loaded: {self.volume.shape[0]} slices")
            self.bodypart_label.setText(f"Type: {body_part}")
            
            # Create views
            self.create_views()
            self.prepare_masks()
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load NIfTI: {str(e)}")

    def load_external_masks(self):
        """Load external segmentation masks."""
        folder = QFileDialog.getExistingDirectory(self, "Select Masks Folder")
        if not folder:
            return
            
        try:
            self.external_masks_dir = folder
            self.using_external_masks = True
            
            # Load masks based on organ keywords
            self.load_external_masks_by_keywords()
            
            QMessageBox.information(self, "Success", "External masks loaded successfully!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load masks: {str(e)}")

    def load_external_masks_by_keywords(self):
        """Load external masks by matching filenames to organ keywords."""
        if not self.external_masks_dir or not self.volume is not None:
            return
            
        import os
        import glob
        
        self.external_mask_volume = np.zeros_like(self.volume, dtype=np.uint8)
        self.external_color_masks = {}
        
        for organ_group, keywords in ORGAN_KEYWORDS.items():
            mask_files = []
            for keyword in keywords:
                pattern = os.path.join(self.external_masks_dir, f"*{keyword}*")
                mask_files.extend(glob.glob(pattern))
            
            if mask_files:
                # Load first matching file
                try:
                    mask_data = nib.load(mask_files[0]).get_fdata()
                    mask_data = match_shape(mask_data, self.volume)
                    
                    # Add to combined mask
                    self.external_mask_volume[mask_data > 0] = 1
                    
                    # Store individual mask
                    self.external_color_masks[organ_group.lower()] = mask_data
                    
                except Exception as e:
                    print(f"Error loading mask for {organ_group}: {e}")

    def run_segmentation(self):
        """Run TotalSegmentator segmentation."""
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
            
        try:
            QMessageBox.information(self, "Info", "Starting TotalSegmentator... This may take several minutes.")
            
            # Run TotalSegmentator
            totalsegmentator(
                self.volume,
                self.output_dir,
                fast=True,
                device="cpu"  # Use CPU for compatibility
            )
            
            # Load the generated masks
            self.load_totalsegmentator_masks()
            
            QMessageBox.information(self, "Success", "Segmentation completed!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Segmentation failed: {str(e)}")

    def load_totalsegmentator_masks(self):
        """Load masks generated by TotalSegmentator."""
        if not os.path.exists(self.output_dir):
            return
            
        import glob
        
        self.mask_volume = np.zeros_like(self.volume, dtype=np.uint8)
        self.color_masks = {}
        
        # Load organ masks
        for organ_group, organs in ORGAN_GROUPS_SIMPLE.items():
            combined_mask = np.zeros_like(self.volume, dtype=np.uint8)
            
            for organ in organs:
                pattern = os.path.join(self.output_dir, f"{organ}.nii.gz")
                files = glob.glob(pattern)
                
                if files:
                    try:
                        mask_data = nib.load(files[0]).get_fdata()
                        mask_data = match_shape(mask_data, self.volume)
                        combined_mask[mask_data > 0] = 1
                    except Exception as e:
                        print(f"Error loading {organ}: {e}")
            
            if np.any(combined_mask):
                self.mask_volume[combined_mask > 0] = 1
                self.color_masks[organ_group.lower()] = combined_mask

    def prepare_masks(self):
        """Prepare masks for display based on selected organ."""
        if not self.color_masks and not self.external_color_masks:
            return
            
        organ = self.organ_dropdown.currentText()
        if organ == "None":
            return
            
        # Use external masks if available, otherwise use TotalSegmentator masks
        masks_to_use = self.external_color_masks if self.using_external_masks else self.color_masks
        
        if not masks_to_use:
            return
            
        # Prepare masks for the selected organ
        organ_key = organ.lower()
        if organ_key in masks_to_use:
            # Update all views with the selected organ mask
            self.update_all_views()

    def change_segmentation_plane(self):
        """Change the segmentation view plane."""
        self.segmentation_plane = self.segplane_dropdown.currentText()
        self.update_all_views()

    def toggle_overlay(self):
        """Toggle mask overlay display."""
        self.overlay_on = self.overlay_btn.isChecked()
        self.overlay_btn.setText(f"Toggle Overlay: {'ON' if self.overlay_on else 'OFF'}")
        self.update_all_views()

    def toggle_roi_zoom(self):
        """Toggle ROI zoom functionality."""
        self.roi_zoom_on = self.roi_btn.isChecked()
        self.roi_btn.setText(f"Toggle ROI Zoom: {'ON' if self.roi_zoom_on else 'OFF'}")
        
        # Update all views
        for view in self.views:
            view.set_roi_zoom(self.roi_zoom_on)
        self.update_all_views()

    def switch_fourth_view(self):
        """Switch between Segmentation and Oblique views."""
        self.fourth_view_mode = self.fourth_view_dropdown.currentText()
        self.create_views()

    def toggle_playback(self):
        """Toggle automatic playback."""
        if self.is_playing:
            self.playback_timer.stop()
            self.is_playing = False
            self.play_pause_btn.setText("▶ Play")
        else:
            self.playback_timer.start(self.playback_speed)
            self.is_playing = True
            self.play_pause_btn.setText("⏸ Pause")

    def change_playback_view(self, view_name):
        """Change which view to animate during playback."""
        self.playback_view = view_name

    def update_playback_speed(self, value):
        """Update playback speed."""
        self.playback_speed = max(50, 1000 - value * 15)  # Convert FPS to milliseconds
        self.speed_label.setText(f"Speed (FPS): {value}")
        
        if self.is_playing:
            self.playback_timer.setInterval(self.playback_speed)

    def toggle_direction(self):
        """Toggle playback direction."""
        self.playback_direction = -1 if self.direction_btn.isChecked() else 1
        self.direction_btn.setText("Backward ◀" if self.playback_direction == -1 else "Forward ▶")

    def toggle_loop(self, state):
        """Toggle loop playback."""
        self.loop_playback = state == Qt.Checked

    def advance_slice(self):
        """Advance to next slice during playback."""
        if self.playback_view not in self.slice_indices:
            return
            
        current_idx = self.slice_indices[self.playback_view]
        max_slices = self.get_max_slices_for_view(self.playback_view)
        
        # Calculate next index
        next_idx = current_idx + self.playback_direction
        
        # Handle looping
        if next_idx >= max_slices:
            if self.loop_playback:
                next_idx = 0
            else:
                self.toggle_playback()  # Stop playback
                return
        elif next_idx < 0:
            if self.loop_playback:
                next_idx = max_slices - 1
            else:
                self.toggle_playback()  # Stop playback
                return
        
        # Update slice index
        self.slice_indices[self.playback_view] = next_idx
        
        # Update the specific view
        if self.playback_view in self.views_dict:
            view = self.views_dict[self.playback_view]
            view.update_slice(next_idx, self.overlay_on)

    def get_max_slices_for_view(self, view_name):
        """Get maximum number of slices for a view."""
        if self.volume is None:
            return 0
            
        if view_name == "Axial":
            return self.volume.shape[0]
        elif view_name == "Coronal":
            return self.volume.shape[1]
        elif view_name == "Sagittal":
            return self.volume.shape[2]
        elif view_name == "Segmentation":
            return self.volume.shape[0]  # Default to axial
        elif view_name == "Oblique":
            return self.volume.shape[0]  # Default to axial
        return 0

    def change_oblique_reference(self):
        """Change oblique reference plane."""
        self.oblique_reference_plane = self.oblique_ref_dropdown.currentText()
        if self.oblique_view:
            self.update_oblique_view()

    def update_rotation_x(self, value):
        """Update X rotation for oblique view."""
        self.oblique_rotation_x = value
        self.rot_x_label.setText(f"Rotation X: {value}°")
        if self.oblique_view:
            self.update_oblique_view()

    def update_rotation_y(self, value):
        """Update Y rotation for oblique view."""
        self.oblique_rotation_y = value
        self.rot_y_label.setText(f"Rotation Y: {value}°")
        if self.oblique_view:
            self.update_oblique_view()

    def update_rotation_z(self, value):
        """Update Z rotation for oblique view."""
        self.oblique_rotation_z = value
        self.rot_z_label.setText(f"Rotation Z: {value}°")
        if self.oblique_view:
            self.update_oblique_view()

    def reset_oblique_rotations(self):
        """Reset all oblique rotations to zero."""
        self.oblique_rotation_x = 0
        self.oblique_rotation_y = 0
        self.oblique_rotation_z = 0
        
        self.rot_x_slider.setValue(0)
        self.rot_y_slider.setValue(0)
        self.rot_z_slider.setValue(0)
        
        if self.oblique_view:
            self.update_oblique_view()

    def export_organ_slices(self):
        """Export organ slices using dialog."""
        if self.volume is None:
            QMessageBox.warning(self, "Warning", "Please load data first!")
            return
            
        # Show slice range dialog
        dialog = SliceRangeDialog(self, self.volume.shape[0])
        if dialog.exec_() == QDialog.Accepted:
            start, end = dialog.get_range()
            
            # Get output filename
            filename, _ = QFileDialog.getSaveFileName(
                self, "Save Organ Slices", "organ_slices.nii.gz", 
                "NIfTI Files (*.nii *.nii.gz)"
            )
            
            if filename:
                try:
                    # Export slices
                    mask_volume = self.mask_volume if hasattr(self, 'mask_volume') else None
                    export_slices(self.volume, start, end, filename, mask_volume)
                    
                    QMessageBox.information(self, "Success", f"Exported {end-start} slices successfully!")
                    
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")

    def create_views(self):
        """Create all image views."""
        # Clear existing views
        for i in reversed(range(self.grid.count())):
            self.grid.itemAt(i).widget().setParent(None)
        
        self.views = []
        self.views_dict = {}
        
        if self.volume is None:
            return
            
        # Create standard views
        views_to_create = ["Axial", "Coronal", "Sagittal"]
        
        # Add fourth view based on mode
        if self.fourth_view_mode == "Segmentation":
            views_to_create.append("Segmentation")
        else:
            views_to_create.append("Oblique")
        
        # Create views
        for i, view_name in enumerate(views_to_create):
            try:
                if view_name == "Segmentation":
                    view = self.create_segmentation_view()
                elif view_name == "Oblique":
                    view = self.create_oblique_view()
                else:
                    view = self.create_standard_view(view_name)
                
                self.views.append(view)
                self.views_dict[view_name] = view
                
                # Add to grid (2x2 layout)
                row = i // 2
                col = i % 2
                self.grid.addWidget(view, row, col)
                
            except Exception as e:
                print(f"Error creating {view_name} view: {e}")
                raise
        
        # Update all views
        self.update_all_views()

    def create_standard_view(self, plane_name):
        """Create a standard orthogonal view."""
        def get_slice_func(idx):
            # Standard medical imaging orientation:
            # volume[slice, height, width] where:
            # - slice dimension (axis 0): superior-inferior (Z-axis)
            # - height dimension (axis 1): anterior-posterior (Y-axis)  
            # - width dimension (axis 2): left-right (X-axis)
            
            if plane_name == "Axial":
                # Axial view: horizontal slices (superior-inferior)
                # Extract slices along the slice dimension (axis 0)
                slice_data = self.volume[idx, :, :]
                return slice_data
            elif plane_name == "Coronal":
                # Coronal view: vertical slices (anterior-posterior)
                # Extract slices along the height dimension (axis 1)
                slice_data = self.volume[:, idx, :]
                return slice_data
            else:  # Sagittal
                # Sagittal view: vertical slices (left-right)
                # Extract slices along the width dimension (axis 2)
                slice_data = self.volume[:, :, idx]
                return slice_data
        
        def get_mask_func(idx):
            if not self.overlay_on:
                return None
            if self.using_external_masks and self.external_mask_volume is not None:
                if plane_name == "Axial":
                    return self.external_mask_volume[idx, :, :]
                elif plane_name == "Coronal":
                    return self.external_mask_volume[:, idx, :]
                else:  # Sagittal
                    return self.external_mask_volume[:, :, idx]
            elif self.mask_volume is not None:
                if plane_name == "Axial":
                    return self.mask_volume[idx, :, :]
                elif plane_name == "Coronal":
                    return self.mask_volume[:, idx, :]
                else:  # Sagittal
                    return self.mask_volume[:, :, idx]
            return None
        
        def get_color_func(idx):
            if not self.overlay_on:
                return None
            masks_to_use = self.external_color_masks if self.using_external_masks else self.color_masks
            if not masks_to_use:
                return None
            
            color_masks = {}
            for organ_name, mask_vol in masks_to_use.items():
                if plane_name == "Axial":
                    color_masks[organ_name] = mask_vol[idx, :, :]
                elif plane_name == "Coronal":
                    color_masks[organ_name] = mask_vol[:, idx, :]
                else:  # Sagittal
                    color_masks[organ_name] = mask_vol[:, :, idx]
            return color_masks
        
        view = SliceView(plane_name, get_slice_func, get_mask_func, get_color_func)
        
        # Connect slider
        view.slider.valueChanged.connect(lambda val: self.update_view_slice(plane_name, val))
        
        # Set initial slice count
        max_slices = self.get_max_slices_for_view(plane_name)
        view.slider.setMaximum(max_slices - 1)
        view.slider.setValue(self.slice_indices[plane_name])
        
        return view

    def create_segmentation_view(self):
        """Create segmentation view."""
        def get_slice_func(idx):
            try:
                return self.volume[idx, :, :]
            except Exception as e:
                print(f"Error getting slice {idx}: {e}")
                raise
        
        def get_mask_func(idx):
            if not self.overlay_on:
                return None
            if self.using_external_masks and self.external_mask_volume is not None:
                return self.external_mask_volume[idx, :, :]
            elif self.mask_volume is not None:
                return self.mask_volume[idx, :, :]
            return None
        
        def get_color_func(idx):
            if not self.overlay_on:
                return None
            masks_to_use = self.external_color_masks if self.using_external_masks else self.color_masks
            if not masks_to_use:
                return None
            
            color_masks = {}
            for organ_name, mask_vol in masks_to_use.items():
                color_masks[organ_name] = mask_vol[idx, :, :]
            return color_masks
        
        view = SliceView("Segmentation", get_slice_func, get_mask_func, get_color_func, "Segmentation - ")
        
        # Connect slider
        view.slider.valueChanged.connect(lambda val: self.update_view_slice("Segmentation", val))
        
        # Set initial slice count
        max_slices = self.volume.shape[0]
        view.slider.setMaximum(max_slices - 1)
        view.slider.setValue(self.slice_indices["Segmentation"])
        
        return view

    def create_oblique_view(self):
        """Create oblique view."""
        def get_slice_func(idx):
            return get_oblique_slice(
                self.volume, idx, 
                self.oblique_rotation_x, 
                self.oblique_rotation_y, 
                self.oblique_rotation_z,
                self.oblique_reference_plane
            )
        
        def get_mask_func(idx):
            if not self.overlay_on or self.mask_volume is None:
                return None
            return get_oblique_slice(
                self.mask_volume, idx,
                self.oblique_rotation_x,
                self.oblique_rotation_y, 
                self.oblique_rotation_z,
                self.oblique_reference_plane
            )
        
        def get_color_func(idx):
            if not self.overlay_on:
                return None
            masks_to_use = self.external_color_masks if self.using_external_masks else self.color_masks
            if not masks_to_use:
                return None
            
            color_masks = {}
            for organ_name, mask_vol in masks_to_use.items():
                color_masks[organ_name] = get_oblique_slice(
                    mask_vol, idx,
                    self.oblique_rotation_x,
                    self.oblique_rotation_y,
                    self.oblique_rotation_z,
                    self.oblique_reference_plane
                )
            return color_masks
        
        view = SliceView("Oblique", get_slice_func, get_mask_func, get_color_func, "Oblique - ")
        
        # Connect slider
        view.slider.valueChanged.connect(lambda val: self.update_view_slice("Oblique", val))
        
        # Set initial slice count
        max_slices = self.get_max_slices_for_view("Oblique")
        view.slider.setMaximum(max_slices - 1)
        view.slider.setValue(self.slice_indices["Oblique"])
        
        self.oblique_view = view
        return view

    def update_view_slice(self, view_name, slice_idx):
        """Update a specific view's slice."""
        self.slice_indices[view_name] = slice_idx
        
        if view_name in self.views_dict:
            view = self.views_dict[view_name]
            view.update_slice(slice_idx, self.overlay_on)
            
            # Update crosshairs for all views
            self.update_crosshairs()

    def update_oblique_view(self):
        """Update the oblique view with new rotations."""
        if self.oblique_view:
            current_idx = self.slice_indices["Oblique"]
            self.oblique_view.update_slice(current_idx, self.overlay_on)

    def update_all_views(self):
        """Update all views with current settings."""
        for view_name, view in self.views_dict.items():
            try:
                current_idx = self.slice_indices[view_name]
                view.update_slice(current_idx, self.overlay_on)
            except Exception as e:
                print(f"Error updating {view_name} view: {e}")
                raise
        
        self.update_crosshairs()

    def update_crosshairs(self):
        """Update crosshair positions across all views."""
        if self.volume is None:
            return
            
        axial_idx = self.slice_indices["Axial"]
        coronal_idx = self.slice_indices["Coronal"] 
        sagittal_idx = self.slice_indices["Sagittal"]
        
        volume_shape = self.volume.shape
        
        # Update crosshairs for all views
        for view_name, view in self.views_dict.items():
            view.set_crosshair_position(axial_idx, coronal_idx, sagittal_idx, volume_shape)
            current_idx = self.slice_indices[view_name]
            view.update_slice(current_idx, self.overlay_on)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = DICOM_MPR_Viewer()
    viewer.show()
    sys.exit(app.exec_())
