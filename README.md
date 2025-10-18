# Medical MPR Viewer

A medical image viewer for DICOM and NIfTI files with multi-planar reconstruction (MPR) capabilities.

## Features

- DICOM and NIfTI file support
- Multi-planar reconstruction (Axial, Sagittal, Coronal views)
- TotalSegmentator integration for organ segmentation
- Crosshair navigation
- Playback functionality

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python run_viewer.py
```

## Project Structure

```
MPR-viewer/
├── src/                    # Source code
│   ├── mpr_viewer.py       # Main application
│   ├── data_loader.py      # Data loading utilities
│   ├── segmentation.py     # Segmentation management
│   ├── ui_components.py    # UI components
│   ├── utils.py           # Utility functions
│   ├── config.py          # Configuration
│   └── __init__.py        # Package initialization
├── models/                 # Model files
│   └── resnet18_orientation_finetuned.pth
├── data/                   # Data directory
│   └── totalsegmentator_output/
├── run_viewer.py          # Main launcher script
├── requirements.txt       # Dependencies
└── README.md             # This file
```

## Requirements

- Python 3.7+
- PyQt5
- NumPy
- Nibabel
- SimpleITK
- TotalSegmentator
