#!/usr/bin/env python3
"""
Launcher script for the Medical MPR Viewer.
Run this file to start the application.
"""

import sys
from PyQt5.QtWidgets import QApplication
from src.mpr_viewer import DICOM_MPR_Viewer

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
