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
    "spine": "yellow",
    "ribcage": "brown",
    "thyroid": "pink",
    "trachea": "lightgreen",
    "adrenal": "darkblue",
    "gallbladder": "gold",
    "pancreas": "darkorange",
    "prostate": "darkmagenta",
    "esophagus": "teal",
    "stomach": "salmon",
    "duodenum": "olive",
    "small bowel": "darkcyan",
    "colon": "darkred",
    "urinary bladder": "deepskyblue",
    "cv_heart": "crimson",
    "cv_arteries": "tomato",
    "cv_veins": "royalblue"
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
    ]] + ["sacrum"],
    "Ribcage": [f"rib_left_{i}" for i in range(1, 13)] + [f"rib_right_{i}" for i in range(1, 13)] + 
              ["sternum", "costal_cartilages"],
    "Thyroid": ["thyroid_gland"],
    "Trachea": ["trachea"],
    "Adrenal": ["adrenal_gland_left", "adrenal_gland_right"],
    "Gallbladder": ["gallbladder"],
    "Pancreas": ["pancreas"],
    "Prostate": ["prostate"],
    "Esophagus": ["esophagus"],
    "Stomach": ["stomach"],
    "Duodenum": ["duodenum"],
    "Small Bowel": ["small_bowel"],
    "Colon": ["colon"],
    "Urinary Bladder": ["urinary_bladder"],
    "Cardiovascular": [
        "heart",
        "aorta",
        "pulmonary_artery",
        "pulmonary_vein",
        "vena_cava_inferior",
        "vena_cava_superior",
        "portal_vein"
    ]
}

# Heart organ with color separation (for prepare_masks)
HEART_COLOR_GROUPS = {"main": ["heart"], "vessels": ["aorta", "pulmonary_artery", "pulmonary_vein"]}

# Grouped labels for Cardiovascular (multi-color)
CV_COLOR_GROUPS = {
    "heart": ["heart"],
    "arteries": ["aorta", "pulmonary_artery"],
    "veins": ["pulmonary_vein", "vena_cava_inferior", "vena_cava_superior", "portal_vein"]
}

# Organ keywords for external mask detection
ORGAN_KEYWORDS = {
    "Lungs": ["lung", "trachea", "airway"],
    "Heart": ["heart", "aorta", "pulmonary"],
    "Brain": ["brain", "cerebellum", "brainstem"],
    "Kidneys": ["kidney"],
    "Liver": ["liver"],
    "Spleen": ["spleen"],
    "Spine": ["vertebra", "spinal", "sacrum"],
    "Ribcage": ["rib", "sternum", "costal", "ribcage"],
    "Thyroid": ["thyroid"],
    "Trachea": ["trachea"],
    "Adrenal": ["adrenal"],
    "Gallbladder": ["gallbladder"],
    "Pancreas": ["pancreas"],
    "Prostate": ["prostate"],
    "Esophagus": ["esophagus"],
    "Stomach": ["stomach"],
    "Duodenum": ["duodenum"],
    "Small Bowel": ["small_bowel", "small bowel"],
    "Colon": ["colon"],
    "Urinary Bladder": ["urinary_bladder", "urinary bladder", "bladder"],
    "Cardiovascular": ["heart", "aorta", "artery", "vein", "cava", "portal"]
}

# Default settings
DEFAULT_OUTPUT_DIR = "data/totalsegmentator_output"
DEFAULT_PLAYBACK_SPEED = 100  # milliseconds between frames
DEFAULT_PLAYBACK_FPS = 10
DEFAULT_WINDOW_SIZE = (1800, 1200)
DEFAULT_MIN_WINDOW_SIZE = (1000, 700)
