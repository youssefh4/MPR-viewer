"""
Configuration constants and settings for the Medical MPR Viewer.
"""

# Outline colors for different organ types
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
        "C1", "C2", "C3", "C4", "C5", "C6", "C7",
        "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9", "T10", "T11", "T12",
        "L1", "L2", "L3", "L4", "L5", "S1"
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

# Default settings
DEFAULT_OUTPUT_DIR = "../data/totalsegmentator_output"
DEFAULT_PLAYBACK_SPEED = 100  # milliseconds between frames
DEFAULT_PLAYBACK_FPS = 10
DEFAULT_WINDOW_SIZE = (1800, 1200)
DEFAULT_MIN_WINDOW_SIZE = (1000, 700)
