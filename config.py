# TRAINING PARAMETERS ------------------------------------
# Data Augmentation parameters
ROT_ANGLE = 60
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
BRIGHTNESS_RANGE = [0.95, 1.05]
ZOOM_RANGE = [0.9, 1.05]
CHANNEL_SHIFT_RANGE = 0.05

# Learning Rate Finder parameters
START_LR = 1e-6
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Path to save model
saved_models_path = "saved_models/run_0_arch"

# INSPECTION PARAMETERS -----------------------------------
# Filenames of validation images used for inspection
FILENAMES_VAL_INSPECTION = [
    "good/a00_003.png",
    "good/a00_103.png",
    "good/a00_153.png",
    "good/a00_203.png",
    "good/a00_303.png",
    "good/a45_003.png",
    "good/a45_103.png",
    "good/a45_153.png",
    "good/a45_203.png",
    "good/a45_303.png",
]

# # Filenames of test images used for inspection
# FILENAMES_TEST_INSPECTION = [
#     "good/a00_008.png",
#     # "good/a00_188.png",
#     "good/a45_008.png",
#     # "good/a45_188.png",
#     # "02_added/a00_009.png",
#     "02_added/a00_028.png",
#     "02_added/a00_085.png",
#     # "02_added/a00_129.png",
#     "02_added/a45_001.png",
#     "02_added/a45_074.png",
#     # "02_added/a45_132.png",
#     "03_missing/a00_025.png",
#     "03_missing/a00_089.png",
#     # "03_missing/a00_103.png",
#     # "03_missing/a45_129.png",
#     "03_missing/a45_037.png",
#     # "04_shifted/a00_090.png",
#     # "04_shifted/a00_315.png",
#     # "04_shifted/a45_001.png",
#     "04_shifted/a45_090.png",
#     "05_color/a00_001.png",
#     # "05_color/a00_225.png",
#     "05_color/a45_001.png",
#     # "05_color/a45_225.png",
#     "06_crack/a00_012.png",
#     # "06_crack/a00_314.png",
#     # "06_crack/a45_001.png",
#     # "06_crack/a45_315.png",
#     # "07_fracture/a00_311.png",
#     # "07_fracture/a00_358.png",
#     "07_fracture/a45_315.png",
#     # "07_fracture/a45_360.png",
#     "08_scratch/a00_090.png",
#     # "08_scratch/a00_133.png",
#     # "08_scratch/a45_197.png",
#     # "08_scratch/a45_136.png",
#     # "09_hole/a00_160.png",
#     "09_hole/a00_224.png",
#     # "09_hole/a45_167.png",
#     "09_hole/a45_224.png",
#     # "10_stain/a00_154.png",
#     "10_stain/a00_224.png",
#     # "10_stain/a45_161.png",
#     "10_stain/a45_225.png",
# ]

FILENAMES_TEST_INSPECTION = [
    # High Contrast & Large Area
    "03_missing/a00_089.png",  # 0
    "03_missing/a45_001.png",  # 1
    "10_stain/a00_224.png",  # 2
    "10_stain/a45_225.png",  # 3
    # High Contrast & Small Area
    # detected
    "09_hole/a00_224.png",  # 4
    "09_hole/a45_224.png",  # 5
    # undetected
    "07_fracture/a45_315.png",  # 6
    "09_hole/a00_209.png",  # 7
    "09_hole/a45_200.png",  # 8
    "10_stain/a00_157.png",  # 9
    "10_stain/a45_168.png",  # 10
    # Low Contrast & Large Area
    "02_added/a00_001.png",  # 11
    "02_added/a45_001.png",  # 12
    "03_missing/a45_041.png",  # 13
    "05_color/a00_272.png",  # 14
    "05_color/a45_001.png",  # 15
    # Low Contrast & Small Area
    "06_crack/a00_012.png",  # 16
    "06_crack/a45_315.png",  # 17
    "08_scratch/a00_136.png",  # 18
    "08_scratch/a45_136.png",  # 19
]

# FILENAMES_TEST_INSPECTION = ["03_missing/a45_037.png"]

# VALIDATION PARAMETERS ---------------------------------------------------
MIN_AREA_VAL_a00 = 25
MIN_AREA_VAL_a45 = 25

# TEST PARAMETERS ---------------------------------------------------------
MIN_AREA_TEST_a00 = 25
THRESHOLD_TEST_a00 = 0.5950040829181671

MIN_AREA_TEST_a45 = 25
THRESHOLD_TEST_a00 = 0.3950000596046448

# FINETUNING PARAMETERS ---------------------------------------------------
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005

LOW_CONTRAST_MIN_PIXEL_INTENSITY = 0.05
LOW_CONTRAST_MAX_PIXEL_INTENSITY = 0.2
LOW_CONTRAST_STEP_PIXEL_INTENSITY = 2e-3

HIGH_CONTRAST_MIN_PIXEL_INTENSITY = 0.2
HIGH_CONTRAST_MAX_PIXEL_INTENSITY = 1.0
HIGH_CONTRAST_STEP_PIXEL_INTENSITY = 2e-3
