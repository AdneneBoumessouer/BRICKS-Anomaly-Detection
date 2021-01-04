# TRAINING PARAMETERS ------------------------------------
# Data Augmentation parameters
ROT_ANGLE = 60
W_SHIFT_RANGE = 0.05
H_SHIFT_RANGE = 0.05
BRIGHTNESS_RANGE = [0.95, 1.05]
ZOOM_RANGE = [0.9, 1.05]
CHANNEL_SHIFT_RANGE = 0.05

train_datagen_args = dict(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    zca_epsilon=1e-6,
    rotation_range=ROT_ANGLE,
    width_shift_range=W_SHIFT_RANGE,
    height_shift_range=H_SHIFT_RANGE,
    brightness_range=BRIGHTNESS_RANGE,
    shear_range=0.0,
    zoom_range=ZOOM_RANGE,
    channel_shift_range=CHANNEL_SHIFT_RANGE,
    fill_mode="nearest",
    cval=0.0,
    horizontal_flip=False,
    vertical_flip=False,
    rescale=1 / 255,
    preprocessing_function=None,
    data_format="channels_last",
    validation_split=0.0,
    dtype="float32",
)

val_test_datagen_args = dict(
    rescale=1 / 255, data_format="channels_last", preprocessing_function=None,
)

# Learning Rate Finder parameters
START_LR = 1e-6
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Path to save model
saved_models_path = "saved_models/run_0_arch"

# FINETUNING PARAMETERS ---------------------------------------------------
START_MIN_AREA_HC = 5
STEP_MIN_AREA_HC = 5
STOP_MIN_AREA_HC = 200


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
    # Anomaly Free
    "good/a00_008.png",
    "good/a45_008.png",
]

