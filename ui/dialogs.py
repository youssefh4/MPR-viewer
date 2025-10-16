"""
Dialog components for the medical viewer.
"""

from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSpinBox, QDialogButtonBox

class SliceRangeDialog(QDialog):
    """Dialog for selecting slice range for export."""
    
    def __init__(self, parent, max_slices):
        super().__init__(parent)
        self.setWindowTitle("Select Slice Range")
        self.setModal(True)
        self.resize(400, 200)
        
        layout = QVBoxLayout(self)
        
        # Instructions
        info_label = QLabel(f"Select slice range to export (Total slices: {max_slices})")
        info_label.setStyleSheet("font-weight: bold; margin-bottom: 10px;")
        layout.addWidget(info_label)
        
        # Start slice
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start slice:"))
        self.start_spinbox = QSpinBox()
        self.start_spinbox.setMinimum(0)
        self.start_spinbox.setMaximum(max_slices - 1)
        self.start_spinbox.setValue(0)
        start_layout.addWidget(self.start_spinbox)
        start_layout.addStretch()
        layout.addLayout(start_layout)
        
        # End slice
        end_layout = QHBoxLayout()
        end_layout.addWidget(QLabel("End slice:"))
        self.end_spinbox = QSpinBox()
        self.end_spinbox.setMinimum(1)
        self.end_spinbox.setMaximum(max_slices)
        self.end_spinbox.setValue(max_slices)
        end_layout.addWidget(self.end_spinbox)
        end_layout.addStretch()
        layout.addLayout(end_layout)
        
        # Preview
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("color: #666; font-style: italic;")
        layout.addWidget(self.preview_label)
        
        # Connect signals for live preview
        self.start_spinbox.valueChanged.connect(self.update_preview)
        self.end_spinbox.valueChanged.connect(self.update_preview)
        
        # Buttons
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        
        # Initial preview
        self.update_preview()
    
    def update_preview(self):
        """Update the preview text."""
        start = self.start_spinbox.value()
        end = self.end_spinbox.value()
        count = end - start
        self.preview_label.setText(f"Will export {count} slices (from {start} to {end-1})")
    
    def get_range(self):
        """Get the selected range."""
        return self.start_spinbox.value(), self.end_spinbox.value()
