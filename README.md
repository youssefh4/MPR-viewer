# Medical MPR Viewer

<div align="center">

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![PyQt5](https://img.shields.io/badge/PyQt5-GUI-green.svg)
![License](https://img.shields.io/badge/License-Educational-orange.svg)
![Status](https://img.shields.io/badge/Status-Stable-brightgreen.svg)

**A comprehensive medical image viewer with advanced visualization capabilities for DICOM and NIfTI files - Designed for Educational and Research Purposes**

[Features](#features) • [Installation](#installation) • [Usage](#usage) • [Documentation](#documentation) • [Contributing](#contributing)

</div>

---

## 🚀 Features

### Core Functionality
- **Multi-Planar Reconstruction (MPR)** - View medical images in axial, sagittal, and coronal planes
- **AI Orientation Detection** - Automatic detection of primary anatomical plane using ResNet18 AI model
- **Dynamic View Arrangement** - Views automatically arrange based on detected orientation
- **Oblique Slicing** - Advanced 3D rotation and oblique plane visualization
- **Crosshair Navigation** - Synchronized navigation across all views
- **ROI Zoom** - Region of interest zoom functionality
- **Automatic Playback** - Play/pause with speed and direction control

### File Format Support
- **DICOM Series** - Full support for DICOM medical imaging files
- **NIfTI Files** - Support for NIfTI-1 and NIfTI-2 formats
- **External Masks** - Load custom segmentation masks

### AI-Powered Features
- **Smart Orientation Detection** - ResNet18 AI model automatically detects primary anatomical plane (Axial, Coronal, Sagittal)
- **Confidence Scoring** - AI provides confidence percentages for orientation detection
- **Fallback Detection** - DICOM metadata fallback when AI detection fails
- **Top 3 Organ Analysis** - AI identifies and ranks organs by volume for comprehensive analysis
- **Intelligent View Layout** - Main view automatically positioned based on detected orientation

### Advanced Segmentation
- **TotalSegmentator Integration** - Automatic organ segmentation using AI
- **Organ Detection** - Automatic detection of lungs, heart, brain, kidneys, liver, spleen, spine, and ribcage
- **Top 3 Organ Detection** - AI identifies and displays the top 3 organs by volume
- **Color-coded Visualization** - Different colors for different organ types
- **Mask Management** - Load, save, and manipulate segmentation masks

### User Interface
- **Modern GUI** - Professional PyQt5-based interface
- **Collapsible Sidebar** - Organized, clutter-free interface
- **Responsive Design** - Adapts to different screen sizes
- **Intuitive Controls** - Easy-to-use medical imaging tools

#### Appearance Modes

**Dark Mode**
![Medical MPR Viewer - Dark Mode]
*Dark mode example: multi-planar views with sidebar in dark theme*

<img width="1920" height="1020" alt="Screenshot 2025-10-21 201045" src="https://github.com/user-attachments/assets/daffd774-53aa-4484-bfab-0d9f2523bd0a" />

**Light Mode**

![Medical MPR Viewer - Light Mode]
*Light mode example: multi-planar views with sidebar in light theme*

<img width="1920" height="1020" alt="Screenshot 2025-10-21 202331" src="https://github.com/user-attachments/assets/d6e38f6c-cb9d-4c34-9273-5af791d99ed4" />


## 🎬 Feature Demo Videos

See the following demonstrations showcasing key features of Medical MPR Viewer:

### 1. Oblique View


https://github.com/user-attachments/assets/0e0c6c7a-71f0-41d1-afbe-4df1b094cc70


<!-- Example: [Watch the Oblique View Demo](link-to-your-video) -->
<!-- Or embed: <video src="link-to-your-video.mp4" controls></video> -->

### 2. Playback Option



https://github.com/user-attachments/assets/62d11e0b-2d01-43ec-87f4-fc25026748ae


<!-- Example: [Watch the Playback Option Demo](link-to-your-video) -->
<!-- Or embed: <video src="link-to-your-video.mp4" controls></video> -->

### 3. ROI (Region of Interest) Option



https://github.com/user-attachments/assets/b1055eac-6ff2-4055-ae3e-e193bc16b96b


<!-- Place your ROI option demo video here -->
<!-- Example: [Watch the ROI Option Demo](link-to-your-video) -->
<!-- Or embed: <video src="link-to-your-video.mp4" controls></video> -->


## 📋 Requirements

### System Requirements
- **Python**: 3.7 or higher
- **Operating System**: Windows, macOS, or Linux
- **Memory**: Minimum 4GB RAM (8GB recommended for large datasets)
- **GPU**: Recommended for TotalSegmentator (optional)

### Dependencies
- PyQt5 >= 5.15.0
- NumPy >= 1.21.0
- Nibabel >= 3.2.0
- SimpleITK >= 2.1.0
- TotalSegmentator >= 1.5.0
- PyTorch >= 1.9.0 (required for AI orientation detection)
- ResNet18 Model (included in models/ directory)

## 🛠️ Installation

### Quick Start
```bash
# Clone the repository
git clone https://github.com/youssefh4/MPR-viewer.git
cd MPR-viewer

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_viewer.py
```

### Alternative Installation
```bash
# Install specific dependencies
pip install PyQt5 numpy nibabel SimpleITK totalsegmentator

# Optional: Install PyTorch for enhanced features
pip install torch torchvision
```

## 🚀 Usage

### Basic Usage
1. **Launch the Application**:
   ```bash
   python run_viewer.py
   ```

2. **Load Medical Images**:
   - Click "Load DICOM Series" or "Load NIfTI File"
   - Navigate to your medical imaging data
   - Select the folder/file to load
   - AI will automatically detect orientation and arrange views accordingly

3. **Navigate Images**:
   - Use mouse wheel to scroll through slices
   - Click and drag to pan
   - Use crosshairs to navigate between views

4. **Run Segmentation** (Optional):
   - Click "Run TotalSegmentator" in the sidebar
   - Wait for AI processing to complete
   - View segmented organs with color coding
   - AI will display the top 3 detected organs by volume

### Advanced Features
- **Oblique Slicing**: Use rotation controls for 3D visualization
- **Playback**: Enable automatic slice playback with speed control
- **ROI Zoom**: Select regions of interest for detailed examination
- **External Masks**: Load custom segmentation results

## 📁 Project Structure

```
MPR-viewer/
├── src/                           # Source code
│   ├── mpr_viewer.py              # Main application
│   ├── data_loader.py             # Data loading utilities
│   ├── segmentation.py            # Segmentation management
│   ├── ui_components.py           # UI components
│   ├── utils.py                   # Utility functions
│   ├── config.py                  # Configuration settings
│   └── __init__.py                # Package initialization
├── models/                         # Model files
│   └── resnet18_orientation_finetuned.pth
├── data/                           # Data directory
│   └── totalsegmentator_output/   # Segmentation outputs
├── run_viewer.py                  # Main launcher script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

## 🔧 Configuration

The application can be configured through `src/config.py`:

- **Organ Colors**: Customize colors for different organ types
- **Organ Groups**: Define organ groupings for segmentation (including ribcage support)
- **Default Settings**: Window size, playback speed, output directories
- **AI Model**: ResNet18 orientation detection model path and settings

## 📖 Documentation

### API Reference
- `DICOM_MPR_Viewer`: Main application class
- `DataLoader`: Handles medical image loading
- `SegmentationManager`: Manages AI segmentation
- `CollapsibleBox`: UI component for organized interface

### Examples
```python
# Basic usage
from src.mpr_viewer import DICOM_MPR_Viewer
from PyQt5.QtWidgets import QApplication

app = QApplication([])
viewer = DICOM_MPR_Viewer()
viewer.show()
app.exec_()
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Clone the repository
git clone https://github.com/youssefh4/MPR-viewer.git
cd MPR-viewer

# Install development dependencies
pip install -r requirements.txt

# Run the application
python run_viewer.py
```

## 🎓 Educational Use

This software is specifically designed for educational and research purposes in medical imaging. It provides an excellent platform for:

### Learning Objectives
- **Medical Image Processing** - Understanding DICOM and NIfTI formats
- **Multi-Planar Reconstruction** - Learning anatomical plane visualization
- **AI in Medical Imaging** - Exploring orientation detection and organ segmentation
- **Medical Visualization** - Hands-on experience with medical image viewers
- **Research Applications** - Tool for academic research in medical imaging

### Educational Features
- **Interactive Learning** - Hands-on exploration of medical images
- **AI Demonstration** - Real-world AI applications in medical imaging
- **Comprehensive Documentation** - Detailed explanations of medical imaging concepts
- **Open Source** - Full access to source code for learning purposes

### Academic Use Cases
- **Medical School Curriculum** - Teaching medical imaging concepts
- **Biomedical Engineering** - Image processing and visualization courses
- **Research Projects** - Academic research in medical imaging
- **Student Assignments** - Practical exercises in medical image analysis

**Note**: This software is NOT intended for clinical use or commercial applications without proper licensing.

## 📝 License

This project is licensed under the Educational License - see the [LICENSE](LICENSE) file for details.

**Educational Use Only**: This software is intended for educational and research purposes. Commercial use requires separate licensing.

## 🙏 Acknowledgments

- **TotalSegmentator** - For providing excellent organ segmentation capabilities
- **PyQt5** - For the robust GUI framework
- **SimpleITK** - For medical image processing
- **Nibabel** - For NIfTI file support

## 📞 Support

- **Issues**: Report bugs and request features on [GitHub Issues](https://github.com/youssefh4/MPR-viewer/issues)
- **Discussions**: Join community discussions on [GitHub Discussions](https://github.com/youssefh4/MPR-viewer/discussions)

## 🔮 Roadmap

- [ ] Web-based viewer using Streamlit
- [ ] Additional segmentation models
- [ ] 3D volume rendering
- [ ] Plugin system for extensions
- [ ] Cloud deployment support
- [ ] Mobile app companion

---

<div align="center">

**Made with ❤️ for the medical imaging community**

[⭐ Star this repo](https://github.com/youssefh4/MPR-viewer) • [🐛 Report Bug](https://github.com/youssefh4/MPR-viewer/issues) • [💡 Request Feature](https://github.com/youssefh4/MPR-viewer/i[...]

</div>
