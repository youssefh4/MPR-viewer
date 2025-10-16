"""
Collapsible box UI component.
"""

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QToolButton
from PyQt5.QtCore import Qt

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
