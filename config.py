# Data augmentation parameters (only for training)
ROT_ANGLE = 5  # 5
W_SHIFT_RANGE = 0.05  # 0.05
H_SHIFT_RANGE = 0.05  # 0.05
FILL_MODE = "nearest"
BRIGHTNESS_RANGE = [0.95, 1.05]
# VAL_SPLIT = 0.1

# Learning Rate Finder parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Training parameters
# mvtecCAE, resnetCAE
# EARLY_STOPPING = 6  # 12
# REDUCE_ON_PLATEAU = 3  # 6

# baselineCAE, inceptionCAE
EARLY_STOPPING = 12  # 12
REDUCE_ON_PLATEAU = 6  # 6

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005


FILENAMES_VAL_INSPECTION = [
    "A_good_sv/004.png",
    "A_good_sv/104.png",
    "A_good_sv/154.png",
    "A_good_sv/204.png",
    "A_good_sv/304.png",
]

FILENAMES_TEST_INSPECTION = [
    "A_good_sv/010.png",
    "A_good_sv/055.png",
    "A_good_sv/080.png",
    "A_good_sv/120.png",
    "A_good_sv/240.png",
    "B_added_sv/020.png",
    "B_added_sv/040.png",
    "B_added_sv/080.png",
    "B_added_sv/090.png",
    "B_added_sv/120.png",
    "B_added_sv/230.png",
    "B_missing_sv/010.png",
    "B_missing_sv/020.png",
    "B_missing_sv/030.png",
    "B_missing_sv/040.png",
    "B_missing_sv/080.png",
    "B_missing_sv/100.png",
    "B_missing_sv/110.png",
    "B_missing_sv/260.png",
    "B_missing_sv/270.png",
    "B_missing_sv/320.png",
    "B_shifted_sv/010.png",
    "B_shifted_sv/030.png",
    "B_shifted_sv/040.png",
    "B_shifted_sv/100.png",
    "B_shifted_sv/200.png",
    "B_shifted_sv/230.png",
    "C_color_1_sv/020.png",
    "C_color_1_sv/050.png",
    "C_color_1_sv/110.png",
    "C_color_2_sv/010.png",
    "C_color_2_sv/030.png",
    "C_color_2_sv/080.png",
    "C_color_3_sv/001.png",
    "C_color_3_sv/010.png",
    "C_color_3_sv/180.png",
]
