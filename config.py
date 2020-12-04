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

# Filenames of test images used for inspection
FILENAMES_TEST_INSPECTION = [
    "good/a00_008.png",
    "good/a00_188.png",
    "good/a45_008.png",
    "good/a45_188.png",
    # low contrast: between added legobrick and reconstrcuted legobricks
    "02_added/a00_009.png",
    # high contrast: between added legobrick and reconstrcuted white background
    "02_added/a00_028.png",
    # high contrast: between added legobrick and reconstrcuted white background
    "02_added/a00_085.png",
    # high contrast: between added legobrick and reconstrcuted white background
    "02_added/a00_129.png",
    # low contrast: between added legobrick and reconstrcuted legobricks
    "02_added/a45_001.png",
    # high contrast: between added legobrick and reconstrcuted white background
    "02_added/a45_074.png",
    # high contrast: between added legobrick and reconstrcuted white background
    "02_added/a45_132.png",
    # low contrast: between added legobrick and reconstrcuted legobricks
    "03_missing/a00_025.png",
    "03_missing/a00_089.png",  # high contrast: white background behind missing legobrick
    "03_missing/a00_103.png",  # high contrast: white background behind missing legobrick
    # low contrast: between missing legobrick and reconstrcuted legobricks
    "03_missing/a45_129.png",
    # medium contrast: between gray background( due to missing legobrick) and reconstructed legobricks
    "03_missing/a45_037.png",
    # low contrast: shifted legobrick not noticable from this angle
    "04_shifted/a00_090.png",
    # high contrast: right half of shifted legobrick and reconstructed white background
    "04_shifted/a00_315.png",
    # hight contrast: left half of shifted legobrick and reconstructed white background
    "04_shifted/a45_001.png",
    # high contrast: upper half of shifted legobrick and reconstructed white background
    "04_shifted/a45_090.png",
    # low constrast: swapped blue brick (lvl0) with yellow brick (lvl0) of lego pyramid
    "05_color/a00_001.png",
    # low constrast: swapped yellow brick (lvl1) with gray brick (lvl2) of lego pyramid
    "05_color/a00_225.png",
    # low constrast: swapped blue brick (lvl0) with yellow brick (lvl0) of lego pyramid
    "05_color/a45_001.png",
    # low constrast: swapped yellow brick (lvl1) with gray brick (lvl2) of lego pyramid
    "05_color/a45_225.png",
    "06_crack/a00_012.png",  # medium contrast: crack and reconstructed yellow brick
    "06_crack/a00_314.png",  # medium contrast: crack and reconstructed yellow brick
    "06_crack/a45_001.png",  # medium contrast: crack and reconstructed yellow brick
    "06_crack/a45_315.png",  # medium contrast: crack and reconstructed yellow brick
    "07_fracture/a00_311.png",  # medium contrast: fracture and reconstructed blue brick
    "07_fracture/a00_358.png",  # medium contrast: fracture and reconstructed blue brick
    "07_fracture/a45_315.png",  # medium contrast: fracture and reconstructed blue brick
    "07_fracture/a45_360.png",  # medium contrast: fracture and reconstructed blue brick
    # high contrast & difficult angle: scratch and reconstructed red brick (not very noticeble from angle)
    "08_scratch/a00_090.png",
    "08_scratch/a00_133.png",  # high contrast: scratch and reconstructed red brick
    # high contrast & difficult angle: scratch and reconstructed red brick
    "08_scratch/a45_197.png",
    "08_scratch/a45_136.png",  # high contrast: scratch and reconstructed red brick
    # medium contrast & difficult angle: hole and reconstructed red brick (not very noticeble from angle)
    "09_hole/a00_160.png",
    "09_hole/a00_224.png",  # high contrast: hole and reconstructed red brick
    # medium contrast & difficult angle: hole and reconstructed red brick (not very noticeble from angle)
    "09_hole/a45_167.png",
    "09_hole/a45_224.png",  # high contrast: hole and reconstructed red brick
    "10_stain/a00_154.png",  # high contrast & difficult angle: white stain on gray legobrick
    "10_stain/a00_224.png",  # high contrast: white stain on gray legobrick
    "10_stain/a45_161.png",  # high contrast & difficult angle: white stain on gray legobrick
    "10_stain/a45_225.png",  # high contrast: white stain on gray legobrick
]

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
