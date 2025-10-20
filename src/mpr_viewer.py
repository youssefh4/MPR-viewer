"""
Main MPR Viewer class for the Medical MPR Viewer.
Contains the main application logic and UI management.
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QGridLayout, QPushButton,
    QFileDialog, QLabel, QSlider, QMessageBox, QComboBox, QHBoxLayout,
    QCheckBox, QGroupBox, QSizePolicy, QScrollArea, QFrame, QSpinBox,
    QToolButton
)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtSignal
from PyQt5.QtGui import QFont

# 3D visualization imports
try:
    import pyvista as pv
    from skimage import measure
    PYTISTA_AVAILABLE = True
except ImportError:
    PYTISTA_AVAILABLE = False

from ui_components import CollapsibleBox, SliceView
from data_loader import DataLoader
from segmentation import SegmentationManager
from utils import get_oblique_slice
from config import DEFAULT_OUTPUT_DIR, DEFAULT_PLAYBACK_SPEED, DEFAULT_WINDOW_SIZE, DEFAULT_MIN_WINDOW_SIZE


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
        self.setGeometry(60, 60, *DEFAULT_WINDOW_SIZE)
        self.setMinimumSize(*DEFAULT_MIN_WINDOW_SIZE)

    def _init_variables(self):
        """Initialize all instance variables for data and state management."""
        # Initialize managers
        self.data_loader = DataLoader()
        self.segmentation_manager = SegmentationManager(DEFAULT_OUTPUT_DIR)
        
        # Volume and mask data
        self.volume = None
        self.original_volume = None
        self.mask_volume = None
        self.color_masks = None

        # Plane and viewing settings
        self.main_plane = "Axial"
        self.detected_plane = "Axial"
        self.overlay_on = False
        self.body_part_examined = "Unknown"
        self.segmentation_plane = "Axial"

        # Oblique view settings
        self.oblique_rotation_x = 0
        self.oblique_rotation_y = 0
        self.oblique_rotation_z = 0
        self.oblique_reference_plane = "Axial"
        self.fourth_view_mode = "Segmentation"

        # View state
        self.slice_indices = {"Axial": 0, "Coronal": 0, "Sagittal": 0, "Oblique": 0}
        self.roi_zoom_on = False
        self.click_mode = "crosshair"  # "crosshair" or "roi"
        self.views = []
        self.views_dict = {}
        self.oblique_view = None
        self.seg_view = None
        self.export_btn = None

        # Playback controls
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self.advance_slice)
        self.is_playing = False
        self.playback_speed = DEFAULT_PLAYBACK_SPEED
        self.playback_view = "Axial"
        self.playback_direction = 1
        self.loop_playback = True

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

        self.crosshair_btn = QPushButton("Toggle Reference Lines: ON")
        self.crosshair_btn.setCheckable(True)
        self.crosshair_btn.setChecked(True)
        self.crosshair_btn.clicked.connect(self.toggle_crosshairs)
        display_group.addWidget(self.crosshair_btn)

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

        # Click mode toggle
        self.click_mode_btn = QPushButton("Click Mode: Crosshair")
        self.click_mode_btn.clicked.connect(self.toggle_click_mode)
        self.click_mode_btn.setCheckable(True)
        self.click_mode_btn.setStyleSheet("QPushButton:checked { background-color: #4CAF50; color: white; }")
        display_group.addWidget(self.click_mode_btn)

        self.clear_roi_btn = QPushButton("Clear ROI")
        self.clear_roi_btn.clicked.connect(self.clear_manual_roi)
        display_group.addWidget(self.clear_roi_btn)

        # Dark mode toggle
        self.darkmode_btn = QPushButton("Dark Mode: OFF")
        self.darkmode_btn.setCheckable(True)
        self.darkmode_btn.clicked.connect(self.toggle_dark_mode)
        self.darkmode_btn.setStyleSheet("QPushButton:checked { background-color: #333333; color: white; }")
        display_group.addWidget(self.darkmode_btn)

        # Fourth view selector
        display_group.addWidget(QLabel("4th View:"))
        self.fourth_view_dropdown = QComboBox()
        self.fourth_view_dropdown.addItems(["Segmentation", "Oblique", "3D Surface"])
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

        # Manual export controls
        manual_btn = QPushButton("Manual Export: Choose Slices…")
        manual_btn.clicked.connect(self.export_manual_slices)
        export_group.addWidget(manual_btn)

        # Export all views
        export_all_btn = QPushButton("Export All Views")
        export_all_btn.clicked.connect(self.export_all_views)
        export_group.addWidget(export_all_btn)

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

    # Data loading methods
    def load_dicom(self):
        """Load a DICOM series from a folder and set up the viewer."""
        folder = QFileDialog.getExistingDirectory(self, "Select DICOM Folder")
        if not folder:
            return
        
        success, error_msg, volume_data, metadata = self.data_loader.load_dicom(folder)
        
        if not success:
            QMessageBox.critical(self, "DICOM Error", error_msg)
            return
        
        # Update instance variables
        self.volume = volume_data
        self.original_volume = self.data_loader.original_volume
        self.main_plane = metadata['main_plane']
        self.detected_plane = metadata['main_plane']
        self.body_part_examined = metadata['body_part']
        
        # Update UI
        self.info_label.setText(f"Loaded {metadata['num_slices']} DICOM slices. Shape: {metadata['shape']}  Main: {self.main_plane}")
        self.bodypart_label.setText(f"Dicom Type: {self.body_part_examined}")
        
        # Reset masks and organ dropdown
        self.data_loader.reset_masks()
        self.organ_dropdown.clear()
        self.organ_dropdown.addItems(["None", "Lungs", "Heart", "Brain", "Kidneys", "Liver", "Spleen", "Spine"])
        self.organ_dropdown.setCurrentIndex(0)
        
        self.setup_views()

    def load_nifti(self):
        """Load a NIfTI file (.nii or .nii.gz) and set up the viewer."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select NIfTI File",
            "",
            "NIfTI Files (.nii *.nii.gz);;All Files ()"
        )
        if not file_path:
            return
        
        success, error_msg, volume_data, metadata = self.data_loader.load_nifti(file_path)
        
        if not success:
            QMessageBox.critical(self, "NIfTI Load Error", f"Failed to load NIfTI file:\n{error_msg}")
            return
        
        # Update instance variables
        self.volume = volume_data
        self.original_volume = self.data_loader.original_volume
        self.main_plane = metadata['main_plane']
        self.detected_plane = metadata['main_plane']
        self.body_part_examined = metadata['body_part']
        
        # Update UI
        ai_method = metadata.get('ai_method', 'Heuristic')
        confidence = metadata.get('confidence', 0.0)
        if confidence > 0:
            self.info_label.setText(f"Loaded NIfTI file. Shape: {metadata['shape']}  Main: {self.main_plane} ({ai_method} AI, {confidence:.1f}% confidence)")
        else:
            self.info_label.setText(f"Loaded NIfTI file. Shape: {metadata['shape']}  Main: {self.main_plane} ({ai_method} AI)")
        self.bodypart_label.setText(f"File Type: NIfTI - {self.body_part_examined}")
        
        # Reset masks and organ dropdown
        self.data_loader.reset_masks()
        self.organ_dropdown.clear()
        self.organ_dropdown.addItems(["None", "Lungs", "Heart", "Brain", "Kidneys", "Liver", "Spleen", "Spine"])
        self.organ_dropdown.setCurrentIndex(0)
        
        self.setup_views()
        
        QMessageBox.information(self, "Success",
                                f"NIfTI file loaded successfully!\nShape: {metadata['shape']}\nOrientation: {self.main_plane}\nAI Model: {ai_method}\nConfidence: {confidence:.1f}%" if confidence > 0 else f"NIfTI file loaded successfully!\nShape: {metadata['shape']}\nOrientation: {self.main_plane}\nAI Model: {ai_method}")

    def load_external_masks(self):
        """Load pre-existing segmentation masks from folder or individual NIfTI files."""
        if self.volume is None:
            QMessageBox.warning(self, "No Input",
                                "First load a DICOM series or NIfTI file, then load your segmentation masks.")
            return

        # Ask user: folder or files
        reply = QMessageBox.question(
            self,
            "Load Masks",
            "Load masks from:\n\n• Folder (all .nii/.nii.gz files) - Click 'Yes'\n• Select individual NIfTI file(s) - Click 'No'",
            QMessageBox.Yes | QMessageBox.No | QMessageBox.Cancel
        )

        if reply == QMessageBox.Cancel:
            return

        mask_files_to_load = []

        if reply == QMessageBox.Yes:
            # Load from folder
            folder = QFileDialog.getExistingDirectory(self, "Select Mask Folder")
            if not folder:
                return
            self.data_loader.external_masks_dir = folder
            mask_files = [f for f in os.listdir(folder) if f.endswith('.nii') or f.endswith('.nii.gz')]
            mask_files_to_load = [(os.path.join(folder, f), f.replace('.nii.gz', '').replace('.nii', '')) for f in
                                  mask_files]
        else:
            # Load individual files
            file_paths, _ = QFileDialog.getOpenFileNames(
                self,
                "Select NIfTI Mask File(s)",
                "",
                "NIfTI Files (.nii *.nii.gz);;All Files ()"
            )
            if not file_paths:
                return
            self.data_loader.external_masks_dir = os.path.dirname(file_paths[0]) if file_paths else ""
            mask_files_to_load = [(f, os.path.basename(f).replace('.nii.gz', '').replace('.nii', '')) for f in
                                  file_paths]

        success, error_msg, mask_data, metadata = self.data_loader.load_external_masks(mask_files_to_load)
        
        if not success:
            QMessageBox.warning(self, "No Masks", error_msg)
            return

        # Update UI
        mask_names = metadata['mask_names']
        main_organ = metadata['main_organ']
        
        self.organ_dropdown.clear()
        self.organ_dropdown.addItems(mask_names)
        if mask_names:
            self.organ_dropdown.setCurrentIndex(0)

        # Update body part label with detected organ
        current_type = self.body_part_examined if hasattr(self, 'body_part_examined') else "Unknown"
        self.bodypart_label.setText(f"Dicom Type: {current_type}\nPrimary Organ (from masks): {main_organ}")

        QMessageBox.information(self, "Mask Loader",
                                f"Loaded {len(mask_names)} mask(s).\nDetected Primary Organ: {main_organ}")
        self.refresh_all_views()

    # Segmentation methods
    def run_segmentation(self):
        """Run TotalSegmentator on the loaded DICOM/NIfTI file and detect main organ."""
        if not hasattr(self.data_loader, "temp_nifti") or not os.path.exists(self.data_loader.temp_nifti):
            QMessageBox.warning(self, "No Input", "Load a DICOM series or NIfTI file first.")
            return

        QMessageBox.information(self, "Segmentation", "Running TotalSegmentator. This may take several minutes...")
        
        success, error_msg, main_organ = self.segmentation_manager.run_totalsegmentator(self.data_loader.temp_nifti)
        
        if not success:
            QMessageBox.critical(self, "Segmentation Error", error_msg)
            return
        
        # Update UI
        self.data_loader.using_external_masks = False
        self.organ_dropdown.clear()
        self.organ_dropdown.addItems(["None", "Lungs", "Heart", "Brain", "Kidneys", "Liver", "Spleen", "Spine"])
        self.organ_dropdown.setCurrentIndex(0)
        self.bodypart_label.setText(
            f"Dicom Type: {self.body_part_examined}\nPrimary Organ (Segmentation): {main_organ}")
        QMessageBox.information(self, "Done", f"Segmentation complete.\nPrimary Organ: {main_organ}")
        self.prepare_masks()

    def prepare_masks(self):
        """Prepare organ masks for display overlay from TotalSegmentator output."""
        organ = self.organ_dropdown.currentText()
        
        self.mask_volume, self.color_masks = self.segmentation_manager.prepare_masks(
            self.volume, organ, 
            self.data_loader.using_external_masks, 
            self.data_loader.external_color_masks
        )
        
        self.refresh_all_views()

    # View management methods
    def setup_views(self):
        """
        Set up the 4-panel MPR view grid.
        Creates Axial, Sagittal, Coronal views, plus either Segmentation or Oblique
        view based on user selection (fourth_view_mode).
        Applies automatic reorientation based on detected main plane.
        """
        self.clear_views()
        if self.volume is None:
            return
        shape = self.volume.shape

        # Reorient volume based on detected main plane
        if hasattr(self.data_loader, 'detected_plane') and self.data_loader.detected_plane != "Axial":
            if self.data_loader.detected_plane == "Coronal":
                print(f"[Info] Detected as Coronal orientation - adjusting view functions")
                print(f"  Shape: {self.original_volume.shape}")
                # Don't transpose data, just adjust view functions
                get_ax = lambda k: self.volume[:, int(np.clip(k, 0, shape[1] - 1)), :]  # Coronal slices
                get_co = lambda j: self.volume[int(np.clip(j, 0, shape[0] - 1)), :, :]  # Axial slices  
                get_sa = lambda i: self.volume[:, :, int(np.clip(i, 0, shape[2] - 1))]   # Sagittal slices
            elif self.data_loader.detected_plane == "Sagittal":
                print(f"[Info] Detected as Sagittal orientation - adjusting view functions")
                print(f"  Shape: {self.original_volume.shape}")
                # Don't transpose data, just adjust view functions
                get_ax = lambda k: self.volume[:, :, int(np.clip(k, 0, shape[2] - 1))]   # Sagittal slices
                get_co = lambda j: self.volume[:, int(np.clip(j, 0, shape[1] - 1)), :]  # Coronal slices
                get_sa = lambda i: self.volume[int(np.clip(i, 0, shape[0] - 1)), :, :]  # Axial slices
            else:
                # Default axial orientation
                get_ax = lambda k: self.volume[int(np.clip(k, 0, shape[0] - 1)), :, :]
                get_co = lambda j: self.volume[:, int(np.clip(j, 0, shape[1] - 1)), :]
                get_sa = lambda i: self.volume[:, :, int(np.clip(i, 0, shape[2] - 1))]
        else:
            # Default axial orientation
            get_ax = lambda k: self.volume[int(np.clip(k, 0, shape[0] - 1)), :, :]
            get_co = lambda j: self.volume[:, int(np.clip(j, 0, shape[1] - 1)), :]
            get_sa = lambda i: self.volume[:, :, int(np.clip(i, 0, shape[2] - 1))]

        def get_oblique(k):
            return get_oblique_slice(
                self.volume,
                k,
                self.oblique_rotation_x,
                self.oblique_rotation_y,
                self.oblique_rotation_z,
                self.oblique_reference_plane
            )

        def seg_mask_func(k):
            if self.mask_volume is None:
                return None
            return self.segmentation_manager.get_mask_slice(k, self.segmentation_plane, self.volume.shape)

        def seg_color_func(k):
            if not self.color_masks:
                return None
            return self.segmentation_manager.get_color_mask_slice(k, self.segmentation_plane, self.volume.shape)

        axial = SliceView("Axial", get_ax)
        sagittal = SliceView("Sagittal", get_sa)
        coronal = SliceView("Coronal", get_co)
        seg_view = SliceView("Segmentation", get_ax, seg_mask_func, seg_color_func)

        # Create oblique view
        if self.oblique_reference_plane == "Axial":
            max_val = shape[0] - 1
            self.slice_indices["Oblique"] = shape[0] // 2
        elif self.oblique_reference_plane == "Coronal":
            max_val = shape[1] - 1
            self.slice_indices["Oblique"] = shape[1] // 2
        else:
            max_val = shape[2] - 1
            self.slice_indices["Oblique"] = shape[2] // 2

        oblique_view = SliceView("Oblique", get_oblique)
        oblique_view.slider.setMaximum(max_val)
        oblique_view.slider.setValue(self.slice_indices["Oblique"])
        oblique_view.slider.valueChanged.connect(self.oblique_slider_changed)

        self.views_dict = {"Axial": axial, "Sagittal": sagittal, "Coronal": coronal}
        self.seg_view = seg_view
        self.oblique_view = oblique_view

        # Click-to-sync hookup
        axial.canvas_clicked.connect(self.on_view_click)
        sagittal.canvas_clicked.connect(self.on_view_click)
        coronal.canvas_clicked.connect(self.on_view_click)

        # Always show first 3 views
        self.views = [axial, sagittal, coronal]

        # Add 4th view based on selection
        if self.fourth_view_mode == "Oblique":
            self.views.append(oblique_view)
            fourth_view = oblique_view
        elif self.fourth_view_mode == "3D Surface":
            # Create a placeholder view for 3D surface
            fourth_view = self.create_3d_surface_view()
            self.views.append(fourth_view)
        else:
            self.views.append(seg_view)
            fourth_view = seg_view

        # Set slider maximums and slice indices based on detected orientation
        if hasattr(self.data_loader, 'detected_plane') and self.data_loader.detected_plane != "Axial":
            if self.data_loader.detected_plane == "Coronal":
                # For Coronal data: Axial=height, Coronal=depth, Sagittal=width
                axial_max = shape[1] - 1
                coronal_max = shape[0] - 1
                sagittal_max = shape[2] - 1
                axial_default = shape[1] // 2
                coronal_default = shape[0] // 2
                sagittal_default = shape[2] // 2
            elif self.data_loader.detected_plane == "Sagittal":
                # For Sagittal data: Axial=width, Coronal=height, Sagittal=depth
                axial_max = shape[2] - 1
                coronal_max = shape[1] - 1
                sagittal_max = shape[0] - 1
                axial_default = shape[2] // 2
                coronal_default = shape[1] // 2
                sagittal_default = shape[0] // 2
            else:
                # Default axial orientation
                axial_max = shape[0] - 1
                coronal_max = shape[1] - 1
                sagittal_max = shape[2] - 1
                axial_default = shape[0] // 2
                coronal_default = shape[1] // 2
                sagittal_default = shape[2] // 2
        else:
            # Default axial orientation
            axial_max = shape[0] - 1
            coronal_max = shape[1] - 1
            sagittal_max = shape[2] - 1
            axial_default = shape[0] // 2
            coronal_default = shape[1] // 2
            sagittal_default = shape[2] // 2

        # Set slider maximums first
        axial.slider.setMaximum(axial_max)
        coronal.slider.setMaximum(coronal_max)
        sagittal.slider.setMaximum(sagittal_max)

        # Ensure slice indices are within bounds after reorientation
        self.slice_indices = {
            "Axial": max(0, min(self.slice_indices.get("Axial", axial_default), axial_max)),
            "Coronal": max(0, min(self.slice_indices.get("Coronal", coronal_default), coronal_max)),
            "Sagittal": max(0, min(self.slice_indices.get("Sagittal", sagittal_default), sagittal_max)),
            "Oblique": max(0, min(self.slice_indices.get("Oblique", axial_default), axial_max)),
        }

        axial.slider.setValue(self.slice_indices["Axial"])
        coronal.slider.setValue(self.slice_indices["Coronal"])
        sagittal.slider.setValue(self.slice_indices["Sagittal"])
        axial.slider.valueChanged.connect(lambda v: self.main_view_slider_changed("Axial", v))
        coronal.slider.valueChanged.connect(lambda v: self.main_view_slider_changed("Coronal", v))
        sagittal.slider.valueChanged.connect(lambda v: self.main_view_slider_changed("Sagittal", v))
        seg_view.slider.valueChanged.connect(self.seg_view_slider_changed)

        # Standard 2x2 grid layout
        self.grid.addWidget(axial, 0, 0)
        self.grid.addWidget(sagittal, 0, 1)
        self.grid.addWidget(coronal, 1, 0)
        self.grid.addWidget(fourth_view, 1, 1)

        # Set equal stretching for all rows and columns
        for i in range(2):
            self.grid.setRowStretch(i, 1)
            self.grid.setColumnStretch(i, 1)

        for view in self.views:
            if hasattr(view, 'set_roi_zoom'):
                view.set_roi_zoom(self.roi_zoom_on)

        # Standard 2x2 grid layout
        self.grid.addWidget(axial, 0, 0)
        self.grid.addWidget(sagittal, 0, 1)
        self.grid.addWidget(coronal, 1, 0)
        self.grid.addWidget(fourth_view, 1, 1)

        # Set equal stretching for all rows and columns
        for i in range(2):
            self.grid.setRowStretch(i, 1)
            self.grid.setColumnStretch(i, 1)

        for view in self.views:
            if hasattr(view, 'set_roi_zoom'):
                view.set_roi_zoom(self.roi_zoom_on)
        self.update_seg_view_slice_func()
        self.update_all_crosshairs()
        self.refresh_all_views()

    def clear_views(self):
        """Clear all views from the grid."""
        for v in self.views:
            self.grid.removeWidget(v)
            v.deleteLater()
        # Also clean up seg_view and oblique_view if they exist but aren't in views list
        if hasattr(self, 'seg_view') and self.seg_view is not None and self.seg_view not in self.views:
            self.grid.removeWidget(self.seg_view)
            self.seg_view.deleteLater()
        if hasattr(self, 'oblique_view') and self.oblique_view is not None and self.oblique_view not in self.views:
            self.grid.removeWidget(self.oblique_view)
            self.oblique_view.deleteLater()
        self.views = []
        self.views_dict = {}
        self.oblique_view = None
        self.seg_view = None

    # Event handlers and utility methods
    def toggle_crosshairs(self):
        """Toggle crosshair display on all views."""
        new_state = not any([view.crosshair_enabled for view in self.views])
        for view in self.views:
            view.crosshair_enabled = new_state
            view.update_slice(view.slider.value(), self.overlay_on)
        self.crosshair_btn.setText(
            "Toggle Reference Lines: ON" if new_state else "Toggle Reference Lines: OFF"
        )
        self.crosshair_btn.setChecked(new_state)

    def toggle_overlay(self):
        """Toggle mask overlay display."""
        self.overlay_on = not self.overlay_on
        self.overlay_btn.setText(f"Toggle Overlay: {'ON' if self.overlay_on else 'OFF'}")
        self.overlay_btn.setChecked(self.overlay_on)
        self.refresh_all_views()

    def toggle_click_mode(self):
        """Toggle between crosshair and ROI click modes."""
        if self.click_mode == "crosshair":
            self.click_mode = "roi"
            self.click_mode_btn.setText("Click Mode: ROI")
            self.click_mode_btn.setChecked(True)
        else:
            self.click_mode = "crosshair"
            self.click_mode_btn.setText("Click Mode: Crosshair")
            self.click_mode_btn.setChecked(False)
        
        # Update all views with the new click mode
        for view in self.views:
            if hasattr(view, 'set_click_mode'):
                view.set_click_mode(self.click_mode)

    def toggle_roi_zoom(self):
        """Toggle ROI zoom functionality."""
        self.roi_zoom_on = not self.roi_zoom_on
        self.roi_btn.setText(f"Toggle ROI Zoom: {'ON' if self.roi_zoom_on else 'OFF'}")
        self.roi_btn.setChecked(self.roi_zoom_on)
        for view in self.views:
            if hasattr(view, 'set_roi_zoom'):
                view.set_roi_zoom(self.roi_zoom_on)
        self.refresh_all_views()

    def switch_fourth_view(self):
        """Switch between Segmentation, Oblique, and 3D Surface view in the 4th panel."""
        self.fourth_view_mode = self.fourth_view_dropdown.currentText()
        if self.volume is not None:
            self.setup_views()

    def change_segmentation_plane(self):
        """Change the segmentation view plane."""
        self.segmentation_plane = self.segplane_dropdown.currentText()
        self.update_seg_view_slice_func()
        self.refresh_all_views()

    def update_seg_view_slice_func(self):
        """Update the segmentation view slice function based on selected plane."""
        if not hasattr(self, 'seg_view') or self.volume is None:
            return
        shape = self.volume.shape
        if self.segmentation_plane == "Axial":
            self.seg_view.get_slice = lambda k: self.volume[int(np.clip(k, 0, shape[0] - 1)), :, :]
            self.seg_view.slider.setMaximum(shape[0] - 1)
            idx = self.slice_indices["Axial"]
        elif self.segmentation_plane == "Coronal":
            self.seg_view.get_slice = lambda k: self.volume[:, int(np.clip(k, 0, shape[1] - 1)), :]
            self.seg_view.slider.setMaximum(shape[1] - 1)
            idx = self.slice_indices["Coronal"]
        else:
            self.seg_view.get_slice = lambda k: self.volume[:, :, int(np.clip(k, 0, shape[2] - 1))]
            self.seg_view.slider.setMaximum(shape[2] - 1)
            idx = self.slice_indices["Sagittal"]
        self.seg_view.slider.blockSignals(True)
        self.seg_view.slider.setValue(idx)
        self.seg_view.update_slice(idx, self.overlay_on)
        self.seg_view.slider.blockSignals(False)

    def _validate_slice_indices(self):
        """Validate and clamp slice indices to prevent IndexError."""
        if self.volume is None:
            return
        
        shape = self.volume.shape
        
        # Determine max indices based on detected orientation
        if hasattr(self.data_loader, 'detected_plane') and self.data_loader.detected_plane != "Axial":
            if self.data_loader.detected_plane == "Coronal":
                # For Coronal data: Axial=height, Coronal=depth, Sagittal=width
                axial_max = shape[1] - 1
                coronal_max = shape[0] - 1
                sagittal_max = shape[2] - 1
            elif self.data_loader.detected_plane == "Sagittal":
                # For Sagittal data: Axial=width, Coronal=height, Sagittal=depth
                axial_max = shape[2] - 1
                coronal_max = shape[1] - 1
                sagittal_max = shape[0] - 1
            else:
                # Default axial orientation
                axial_max = shape[0] - 1
                coronal_max = shape[1] - 1
                sagittal_max = shape[2] - 1
        else:
            # Default axial orientation
            axial_max = shape[0] - 1
            coronal_max = shape[1] - 1
            sagittal_max = shape[2] - 1
        
        # Clamp all slice indices to valid ranges
        self.slice_indices["Axial"] = max(0, min(self.slice_indices.get("Axial", 0), axial_max))
        self.slice_indices["Coronal"] = max(0, min(self.slice_indices.get("Coronal", 0), coronal_max))
        self.slice_indices["Sagittal"] = max(0, min(self.slice_indices.get("Sagittal", 0), sagittal_max))
        self.slice_indices["Oblique"] = max(0, min(self.slice_indices.get("Oblique", 0), axial_max))

    def update_all_crosshairs(self):
        """Update crosshair positions across all views."""
        if self.volume is None:
            return
        axial_idx = self.slice_indices["Axial"]
        coronal_idx = self.slice_indices["Coronal"]
        sagittal_idx = self.slice_indices["Sagittal"]
        volume_shape = self.volume.shape
        for view in self.views:
            if view.plane in ["Axial", "Coronal", "Sagittal"]:
                view.set_crosshair_position(axial_idx, coronal_idx, sagittal_idx, volume_shape)

    def refresh_all_views(self):
        """Refresh all views with current settings."""
        if not self.views or self.volume is None:
            return
        
        # Validate and clamp slice indices to prevent IndexError
        self._validate_slice_indices()
        
        self.update_all_crosshairs()
        self.views_dict["Axial"].update_slice(self.slice_indices["Axial"], self.overlay_on)
        self.views_dict["Coronal"].update_slice(self.slice_indices["Coronal"], self.overlay_on)
        self.views_dict["Sagittal"].update_slice(self.slice_indices["Sagittal"], self.overlay_on)

        # Update 4th view based on mode
        if self.fourth_view_mode == "Oblique":
            if self.oblique_view:
                self.oblique_view.update_slice(self.slice_indices["Oblique"], self.overlay_on)
        else:  # Segmentation
            if self.seg_view and self.segmentation_plane in self.slice_indices:
                seg_slice_idx = self.slice_indices[self.segmentation_plane]
                self.seg_view.slider.blockSignals(True)
                self.seg_view.slider.setValue(seg_slice_idx)
                self.seg_view.update_slice(seg_slice_idx, self.overlay_on)
                self.seg_view.slider.blockSignals(False)

    def apply_dark_mode(self, enabled: bool):
        """Apply dark theme to app widgets and slice views."""
        try:
            from PyQt5.QtWidgets import QApplication
            app = QApplication.instance()
            if app is not None:
                if enabled:
                    app.setStyleSheet(
                        """
                        QWidget { background-color: #121212; color: #e0e0e0; }
                        QScrollArea { background-color: #121212; }
                        QFrame { border-color: #333333; }
                        QPushButton { background-color: #2a2a2a; color: #e0e0e0; border: 1px solid #3a3a3a; }
                        QPushButton:hover { background-color: #333333; }
                        QSlider::groove:horizontal { background: #2a2a2a; height: 6px; }
                        QSlider::handle:horizontal { background: #4a90e2; width: 12px; margin: -4px 0; border-radius: 6px; }
                        QLabel { color: #e0e0e0; }
                        """
                    )
                else:
                    app.setStyleSheet("")
        except Exception:
            pass

        # Apply to each view canvas/labels
        for view in [self.views_dict.get("Axial"), self.views_dict.get("Coronal"), self.views_dict.get("Sagittal"), self.seg_view, self.oblique_view]:
            if view is not None and hasattr(view, 'set_dark_mode'):
                view.set_dark_mode(enabled)

    def toggle_dark_mode(self):
        is_enabled = not self.darkmode_btn.isChecked()  # will flip after setChecked
        # Reflect current checked state text after toggle
        enabled = not (self.darkmode_btn.text().endswith("OFF")) if False else self.darkmode_btn.isChecked()
        enabled = self.darkmode_btn.isChecked()
        self.darkmode_btn.setText(f"Dark Mode: {'ON' if enabled else 'OFF'}")
        self.apply_dark_mode(enabled)

    # Slider event handlers
    def main_view_slider_changed(self, plane, value):
        """Handle slider changes for main views (Axial, Coronal, Sagittal)."""
        # Validate slice indices before updating
        self._validate_slice_indices()
        
        self.slice_indices[plane] = value
        self.update_all_crosshairs()
        self.views_dict[plane].update_slice(value, self.overlay_on)
        for other_plane in ["Axial", "Coronal", "Sagittal"]:
            if other_plane != plane:
                self.views_dict[other_plane].update_slice(self.slice_indices[other_plane], self.overlay_on)
        if self.segmentation_plane == plane and hasattr(self, 'seg_view'):
            self.seg_view.slider.blockSignals(True)
            self.seg_view.slider.setValue(value)
            self.seg_view.update_slice(value, self.overlay_on)
            self.seg_view.slider.blockSignals(False)

    def seg_view_slider_changed(self, value):
        """Handle slider changes for segmentation view."""
        # Validate slice indices before updating
        self._validate_slice_indices()
        
        main_plane = self.segmentation_plane
        self.slice_indices[main_plane] = value
        self.update_all_crosshairs()
        self.views_dict[main_plane].slider.blockSignals(True)
        self.views_dict[main_plane].slider.setValue(value)
        self.views_dict[main_plane].update_slice(value, self.overlay_on)
        self.views_dict[main_plane].slider.blockSignals(False)
        if hasattr(self, 'seg_view'):
            self.seg_view.update_slice(value, self.overlay_on)
        for other_plane in ["Axial", "Coronal", "Sagittal"]:
            if other_plane != main_plane:
                self.views_dict[other_plane].update_slice(self.slice_indices[other_plane], self.overlay_on)

    def oblique_slider_changed(self, value):
        """Handle slider changes for oblique view."""
        # Validate slice indices before updating
        self._validate_slice_indices()
        
        self.slice_indices["Oblique"] = value
        if hasattr(self, 'oblique_view') and self.oblique_view:
            self.oblique_view.update_slice(value, self.overlay_on)

    # Mouse click sync across views (mirrors viewport_manager.py behavior)
    def on_view_click(self, x, y, plane):
        if self.volume is None:
            return
        # Map click to other views' indices; note our displayed images are np.rot90
        Z, Y, X = self.volume.shape
        # Adjust mapping to match visual click position (accounting for rotation and axis flips)
        try:
            if plane == "Axial":
                # Original axial slice shape (Y, X)
                orig_row = int(np.clip(x, 0, Y - 1))            # maps to Coronal (Y) - flip X
                orig_col = int(np.clip((X - 1) - y, 0, X - 1))  # maps to Sagittal (X) - flip Y
                self.slice_indices["Sagittal"] = orig_col
                self.slice_indices["Coronal"] = orig_row
                self.views_dict["Sagittal"].slider.setValue(self.slice_indices["Sagittal"])
                self.views_dict["Coronal"].slider.setValue(self.slice_indices["Coronal"])
            elif plane == "Sagittal":
                # Original sagittal slice shape (Z, Y)
                orig_row = int(np.clip(x, 0, Z - 1))            # maps to Axial (Z) - flip X
                orig_col = int(np.clip((Y - 1) - y, 0, Y - 1))  # maps to Coronal (Y) - flip Y
                self.slice_indices["Coronal"] = orig_col
                self.slice_indices["Axial"] = orig_row
                self.views_dict["Coronal"].slider.setValue(self.slice_indices["Coronal"])
                self.views_dict["Axial"].slider.setValue(self.slice_indices["Axial"])
            elif plane == "Coronal":
                # Original coronal slice shape (Z, X)
                orig_row = int(np.clip(x, 0, Z - 1))            # maps to Axial (Z) - flip X
                orig_col = int(np.clip((X - 1) - y, 0, X - 1))  # maps to Sagittal (X) - flip Y
                self.slice_indices["Sagittal"] = orig_col
                self.slice_indices["Axial"] = orig_row
                self.views_dict["Sagittal"].slider.setValue(self.slice_indices["Sagittal"])
                self.views_dict["Axial"].slider.setValue(self.slice_indices["Axial"])
            
            # Validate slice indices before refreshing views
            self._validate_slice_indices()
            
            # Refresh all views and crosshairs
            self.refresh_all_views()
        except Exception:
            pass

    # Oblique view controls
    def change_oblique_reference(self):
        """Change the oblique view reference plane."""
        new_ref = self.oblique_ref_dropdown.currentText()
        if self.volume is not None:
            shape = self.volume.shape
            if new_ref == "Axial":
                max_val = shape[0] - 1
                self.slice_indices["Oblique"] = shape[0] // 2
            elif new_ref == "Coronal":
                max_val = shape[1] - 1
                self.slice_indices["Oblique"] = shape[1] // 2
            else:
                max_val = shape[2] - 1
                self.slice_indices["Oblique"] = shape[2] // 2
            if hasattr(self, 'oblique_view') and self.oblique_view:
                self.oblique_view.slider.blockSignals(True)
                self.oblique_view.slider.setMaximum(max_val)
                self.oblique_view.slider.setValue(self.slice_indices["Oblique"])
                self.oblique_view.slider.blockSignals(False)
        self.oblique_reference_plane = new_ref
        if self.fourth_view_mode == "Oblique" and self.volume is not None:
            self.update_oblique_view()

    def update_rotation_x(self, value):
        """Update X rotation for oblique view."""
        self.oblique_rotation_x = value
        self.rot_x_label.setText(f"Rotation X: {value}°")
        if self.fourth_view_mode == "Oblique" and hasattr(self, 'oblique_view') and self.oblique_view:
            self.update_oblique_view()

    def update_rotation_y(self, value):
        """Update Y rotation for oblique view."""
        self.oblique_rotation_y = value
        self.rot_y_label.setText(f"Rotation Y: {value}°")
        if self.fourth_view_mode == "Oblique" and hasattr(self, 'oblique_view') and self.oblique_view:
            self.update_oblique_view()

    def update_rotation_z(self, value):
        """Update Z rotation for oblique view."""
        self.oblique_rotation_z = value
        self.rot_z_label.setText(f"Rotation Z: {value}°")
        if self.fourth_view_mode == "Oblique" and hasattr(self, 'oblique_view') and self.oblique_view:
            self.update_oblique_view()

    def reset_oblique_rotations(self):
        """Reset all oblique rotations to zero."""
        self.rot_x_slider.setValue(0)
        self.rot_y_slider.setValue(0)
        self.rot_z_slider.setValue(0)

    def update_oblique_view(self):
        """Update the oblique view display."""
        if hasattr(self, 'oblique_view') and self.oblique_view and self.volume is not None:
            idx = self.slice_indices.get("Oblique", 0)
            self.oblique_view.update_slice(idx, self.overlay_on)

    # Manual ROI support
    def clear_manual_roi(self):
        for v in [self.views_dict.get("Axial"), self.views_dict.get("Coronal"), self.views_dict.get("Sagittal"), self.seg_view, self.oblique_view]:
            if v is not None and hasattr(v, 'clear_manual_roi'):
                v.clear_manual_roi()

    # Manual export flow
    def export_manual_slices(self):
        if self.volume is None:
            QMessageBox.warning(self, "No Data", "Load a scan first.")
            return
        
        # Simple dialog for slice range selection
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        
        dialog = QDialog(self)
        dialog.setWindowTitle("Manual Export - Select Range")
        dialog.setModal(True)
        layout = QVBoxLayout(dialog)
        
        # Plane selection
        layout.addWidget(QLabel("Export Plane:"))
        plane_combo = QComboBox()
        plane_combo.addItems(["Axial", "Coronal", "Sagittal"])
        plane_combo.setCurrentText(self.segmentation_plane if self.segmentation_plane in ["Axial", "Coronal", "Sagittal"] else "Axial")
        layout.addWidget(plane_combo)
        
        # Current slice info
        current_idx = self.slice_indices.get(plane_combo.currentText(), 0)
        layout.addWidget(QLabel(f"Current slice: {current_idx}"))
        
        # Range selection
        layout.addWidget(QLabel("Start slice:"))
        start_spin = QSpinBox()
        start_spin.setRange(0, 9999)
        start_spin.setValue(max(0, current_idx - 5))
        layout.addWidget(start_spin)
        
        layout.addWidget(QLabel("End slice:"))
        end_spin = QSpinBox()
        end_spin.setRange(0, 9999)
        end_spin.setValue(min(9999, current_idx + 5))
        layout.addWidget(end_spin)
        
        # Update max values when plane changes
        def update_max_values():
            plane = plane_combo.currentText()
            if plane == "Axial":
                max_val = self.volume.shape[0] - 1
            elif plane == "Coronal":
                max_val = self.volume.shape[1] - 1
            else:  # Sagittal
                max_val = self.volume.shape[2] - 1
            
            start_spin.setMaximum(max_val)
            end_spin.setMaximum(max_val)
            start_spin.setValue(max(0, min(start_spin.value(), max_val)))
            end_spin.setValue(max(0, min(end_spin.value(), max_val)))
        
        plane_combo.currentTextChanged.connect(update_max_values)
        update_max_values()  # Initial setup
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        
        if dialog.exec_() != QDialog.Accepted:
            return
        
        plane = plane_combo.currentText()
        start = min(start_spin.value(), end_spin.value())
        end = max(start_spin.value(), end_spin.value())
        
        # Export the selected range
        fname, _ = QFileDialog.getSaveFileName(
            self, 
            "Export Manual Selection", 
            f"manual_{plane.lower()}_{start}-{end}.nii.gz", 
            "NIfTI (*.nii *.nii.gz)"
        )
        if not fname:
            return
        
        try:
            import nibabel as nib
            
            # Extract the slice range
            if plane == "Axial":
                sub_volume = self.volume[start:end+1, :, :]
            elif plane == "Coronal":
                sub_volume = self.volume[:, start:end+1, :]
            else:  # Sagittal
                sub_volume = self.volume[:, :, start:end+1]
            
            # Create NIfTI image
            img_nifti = nib.Nifti1Image(sub_volume, affine=np.eye(4))
            nib.save(img_nifti, fname)
            
            QMessageBox.information(
                self, 
                "Export Complete", 
                f"Exported {plane} slices {start}-{end} to:\n{os.path.basename(fname)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export: {str(e)}")

    def export_all_views(self):
        """Export all three main views (Axial, Coronal, Sagittal) as separate files."""
        if self.volume is None:
            QMessageBox.warning(self, "No Data", "Load a scan first.")
            return
        
        # Ask for output directory
        output_dir = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if not output_dir:
            return
        
        try:
            import nibabel as nib
            
            # Export each view
            views_exported = []
            
            # Axial view (all slices)
            axial_data = self.volume
            axial_nifti = nib.Nifti1Image(axial_data, affine=np.eye(4))
            axial_path = os.path.join(output_dir, "axial_view.nii.gz")
            nib.save(axial_nifti, axial_path)
            views_exported.append("Axial")
            
            # Coronal view (transposed)
            coronal_data = np.transpose(self.volume, (1, 0, 2))
            coronal_nifti = nib.Nifti1Image(coronal_data, affine=np.eye(4))
            coronal_path = os.path.join(output_dir, "coronal_view.nii.gz")
            nib.save(coronal_nifti, coronal_path)
            views_exported.append("Coronal")
            
            # Sagittal view (transposed)
            sagittal_data = np.transpose(self.volume, (2, 1, 0))
            sagittal_nifti = nib.Nifti1Image(sagittal_data, affine=np.eye(4))
            sagittal_path = os.path.join(output_dir, "sagittal_view.nii.gz")
            nib.save(sagittal_nifti, sagittal_path)
            views_exported.append("Sagittal")
            
            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported all views to:\n{output_dir}\n\nFiles:\n" + 
                "\n".join([f"• {view.lower()}_view.nii.gz" for view in views_exported])
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export all views: {str(e)}")

    # Playback controls
    def toggle_playback(self):
        """Start or stop the automatic playback."""
        if self.volume is None:
            QMessageBox.warning(self, "No Data", "Please load a scan first.")
            return

        self.is_playing = not self.is_playing

        if self.is_playing:
            self.play_pause_btn.setText("⏸ Pause")
            self.play_pause_btn.setStyleSheet(
                "QPushButton { background-color: #FF9800; color: white; font-weight: bold; }")
            self.playback_timer.start(self.playback_speed)
        else:
            self.play_pause_btn.setText("▶ Play")
            self.play_pause_btn.setStyleSheet(
                "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; }")
            self.playback_timer.stop()

    def change_playback_view(self, view_name):
        """Change which view is being animated."""
        self.playback_view = view_name

    def update_playback_speed(self, fps):
        """Update the playback speed based on frames per second."""
        self.playback_speed = int(1000 / fps)  # Convert FPS to milliseconds
        self.speed_label.setText(f"Speed (FPS): {fps}")

        # Update timer interval if playing
        if self.is_playing:
            self.playback_timer.setInterval(self.playback_speed)

    def toggle_direction(self):
        """Toggle playback direction between forward and backward."""
        self.playback_direction *= -1
        if self.playback_direction == 1:
            self.direction_btn.setText("Forward ▶")
            self.direction_btn.setChecked(False)
        else:
            self.direction_btn.setText("◀ Backward")
            self.direction_btn.setChecked(True)

    def toggle_loop(self, state):
        """Enable or disable looping at the end of playback."""
        self.loop_playback = (state == Qt.Checked)

    def advance_slice(self):
        """Advance to the next slice during playback."""
        if self.volume is None:
            self.playback_timer.stop()
            return

        # Validate slice indices before advancing
        self._validate_slice_indices()

        # Determine which view to animate and get its slider
        target_view = None
        max_value = 0
        current_value = 0

        if self.playback_view == "Segmentation":
            if hasattr(self, 'seg_view') and self.seg_view:
                target_view = self.seg_view
                max_value = target_view.slider.maximum()
                current_value = target_view.slider.value()
        elif self.playback_view == "Oblique":
            if hasattr(self, 'oblique_view') and self.oblique_view:
                target_view = self.oblique_view
                max_value = target_view.slider.maximum()
                current_value = target_view.slider.value()
        elif self.playback_view in self.views_dict:
            target_view = self.views_dict[self.playback_view]
            max_value = target_view.slider.maximum()
            current_value = target_view.slider.value()

        if target_view is None:
            return

        # Calculate next slice index
        next_value = current_value + self.playback_direction

        # Handle boundaries
        if next_value > max_value:
            if self.loop_playback:
                next_value = 0
            else:
                # Stop at the end
                self.toggle_playback()
                return
        elif next_value < 0:
            if self.loop_playback:
                next_value = max_value
            else:
                # Stop at the beginning
                self.toggle_playback()
                return

        # Update the slider (this will trigger the appropriate update callbacks)
        target_view.slider.setValue(next_value)

    # Export functionality
    def export_organ_slices(self):
        """Export organ slices to NIfTI file (image + mask sub-volumes)."""
        organ = self.organ_dropdown.currentText()
        if self.volume is None or self.mask_volume is None:
            QMessageBox.warning(self, "No Data", "Please load a scan and masks, then select an organ.")
            return

        # Find slice indices containing organ
        slices_with_organ = []
        for i in range(self.volume.shape[0]):  # axial slices
            if np.any(self.mask_volume[i, :, :] > 0):
                slices_with_organ.append(i)

        if not slices_with_organ:
            QMessageBox.warning(self, "No Slices", f"No slices found for organ '{organ}'.")
            return

        # Suggest range with margin
        margin = 2
        start = max(min(slices_with_organ) - margin, 0)
        end = min(max(slices_with_organ) + margin + 1, self.volume.shape[0])

        # Prompt user for export location
        fname, _ = QFileDialog.getSaveFileName(
            self,
            "Export Slices",
            f"{organ.lower()}_slices.nii.gz",
            "NIfTI (*.nii *.nii.gz)"
        )
        if not fname:
            return

        # Save image and mask sub-volumes
        try:
            import nibabel as nib
            
            img_affine = None
            img_header = None

            # Try to get affine/header from NIfTI source if available
            if hasattr(self.data_loader, 'temp_nifti') and os.path.exists(self.data_loader.temp_nifti):
                src_img = nib.load(self.data_loader.temp_nifti)
                img_affine = src_img.affine
                img_header = src_img.header

            img_data = self.volume[start:end, :, :]
            mask_data = self.mask_volume[start:end, :, :]

            img_nifti = nib.Nifti1Image(
                img_data,
                affine=img_affine if img_affine is not None else np.eye(4),
                header=img_header
            )
            mask_nifti = nib.Nifti1Image(
                mask_data,
                affine=img_affine if img_affine is not None else np.eye(4),
                header=img_header
            )

            # Remove extension and add suffix
            if fname.endswith('.nii.gz'):
                base = fname[:-7]  # Remove .nii.gz
                img_filename = base + '_img.nii.gz'
                mask_filename = base + '_mask.nii.gz'
            elif fname.endswith('.nii'):
                base = fname[:-4]  # Remove .nii
                img_filename = base + '_img.nii'
                mask_filename = base + '_mask.nii'
            else:
                img_filename = fname + '_img.nii.gz'
                mask_filename = fname + '_mask.nii.gz'

            nib.save(img_nifti, img_filename)
            nib.save(mask_nifti, mask_filename)

            QMessageBox.information(
                self,
                "Export Complete",
                f"Exported slices {start}-{end - 1} for organ '{organ}'.\n\nFiles:\n• {os.path.basename(img_filename)}\n• {os.path.basename(mask_filename)}"
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", str(e))

    def create_3d_surface_view(self):
        """Create an enhanced view for 3D surface visualization with controls."""
        view_widget = QWidget()
        view_widget.plane = "3D Surface"  # Add plane attribute for compatibility
        layout = QVBoxLayout()
        
        # Add a button to show 3D surface
        show_3d_btn = QPushButton("Show 3D Surface")
        show_3d_btn.clicked.connect(self.show_3d_surface)
        show_3d_btn.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-weight: bold; }")
        layout.addWidget(show_3d_btn)
        
        # Add threshold control
        threshold_layout = QVBoxLayout()
        threshold_layout.setSpacing(2)
        
        self.threshold_label = QLabel("Surface Threshold: 70%")
        threshold_layout.addWidget(self.threshold_label)
        
        self.threshold_slider = QSlider(Qt.Horizontal)
        self.threshold_slider.setMinimum(10)
        self.threshold_slider.setMaximum(95)
        self.threshold_slider.setValue(70)
        self.threshold_slider.valueChanged.connect(self.update_3d_threshold)
        threshold_layout.addWidget(self.threshold_slider)
        
        layout.addLayout(threshold_layout)
        
        # Add surface options
        options_layout = QVBoxLayout()
        options_layout.setSpacing(5)
        
        self.smooth_surface_cb = QCheckBox("Smooth Surface")
        self.smooth_surface_cb.setChecked(True)
        options_layout.addWidget(self.smooth_surface_cb)
        
        self.show_axes_cb = QCheckBox("Show Axes")
        self.show_axes_cb.setChecked(True)
        options_layout.addWidget(self.show_axes_cb)
        
        layout.addLayout(options_layout)
        
        # Add info label
        info_label = QLabel("3D Surface Controls\n\n• Adjust threshold to change surface detail\n• Use mouse to zoom, pan, and rotate\n• Enable/disable surface smoothing")
        info_label.setAlignment(Qt.AlignCenter)
        info_label.setWordWrap(True)
        info_label.setStyleSheet("QLabel { font-size: 10px; color: #666; }")
        layout.addWidget(info_label)
        
        view_widget.setLayout(layout)
        return view_widget

    def update_3d_threshold(self, value):
        """Update the 3D surface threshold value."""
        self.threshold_label.setText(f"Surface Threshold: {value}%")
    
    def show_3d_surface(self):
        """Show 3D surface visualization using PyVista with interactive zoom controls."""
        if not PYTISTA_AVAILABLE:
            QMessageBox.warning(self, "3D Visualization", 
                              "PyVista and scikit-image are required for 3D visualization.\n"
                              "Install with: pip install pyvista scikit-image")
            return
        
        if self.volume is None:
            QMessageBox.warning(self, "3D Visualization", "No volume data loaded.")
            return
        
        try:
            # Ensure volume is 3D and has reasonable size
            if len(self.volume.shape) != 3:
                QMessageBox.warning(self, "3D Visualization", "Volume must be 3D for surface visualization.")
                return
            
            # Get threshold from slider
            threshold_percent = self.threshold_slider.value()
            
            # Use threshold and ensure data is properly scaled
            volume_normalized = (self.volume - self.volume.min()) / (self.volume.max() - self.volume.min())
            threshold = np.percentile(volume_normalized, threshold_percent)
            mask_3d = volume_normalized > threshold
            
            # Check if mask has any True values
            if not np.any(mask_3d):
                QMessageBox.warning(self, "3D Visualization", f"No surface found with {threshold_percent}% threshold. Try adjusting the threshold.")
                return
            
            # Marching cubes with proper spacing
            verts, faces, _, _ = measure.marching_cubes(mask_3d, level=0.5, spacing=(1.0, 1.0, 1.0))
            
            if len(verts) == 0 or len(faces) == 0:
                QMessageBox.warning(self, "3D Visualization", "No surface geometry generated.")
                return
            
            faces_flat = np.hstack([np.full((faces.shape[0], 1), 3), faces]).astype(np.int32)
            surface = pv.PolyData(verts, faces_flat)
            
            # Create interactive plotter with zoom controls
            plotter = pv.Plotter()
            
            # Get settings from checkboxes
            smooth_shading = self.smooth_surface_cb.isChecked()
            show_axes = self.show_axes_cb.isChecked()
            
            plotter.add_mesh(surface, smooth_shading=smooth_shading, color='lightblue', opacity=0.8)
            
            # Add helpful text
            plotter.add_text("3D Surface Visualization\n\nMouse Controls:\n• Left Click + Drag: Rotate\n• Right Click + Drag: Pan\n• Scroll Wheel: Zoom In/Out\n• Middle Click + Drag: Zoom\n\nKeyboard:\n• 'r': Reset view\n• 'q': Quit", 
                           position='upper_left', font_size=10)
            
            # Set up the plotter with better initial view
            plotter.set_background('white')
            if show_axes:
                plotter.show_axes()
            plotter.enable_depth_peeling()  # Better transparency rendering
            
            # Show the interactive window
            plotter.show()
            
        except Exception as e:
            QMessageBox.critical(self, "3D Visualization Error", f"Error creating 3D surface: {str(e)}")
    


def main():
    """Main entry point for the MPR Viewer application."""
    app = QApplication(sys.argv)
    
    # Create and show the main window
    viewer = DICOM_MPR_Viewer()
    viewer.show()
    
    # Start the application event loop
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()