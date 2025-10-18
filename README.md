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

## Requirements

- Python 3.7+
- PyQt5
- NumPy
- Nibabel
- SimpleITK
- TotalSegmentator
