# Data augmentation parameters (only for training)
ROT_ANGLE = 8  # 5
W_SHIFT_RANGE = 0.1  # 0.05
H_SHIFT_RANGE = 0.1  # 0.05
FILL_MODE = "nearest"
BRIGHTNESS_RANGE = [0.95, 1.05]
# VAL_SPLIT = 0.1

# Learning Rate Finder parameters
START_LR = 1e-5
LR_MAX_EPOCHS = 10
LRF_DECREASE_FACTOR = 0.85

# Finetuning parameters
FINETUNE_SPLIT = 0.1
STEP_MIN_AREA = 5
START_MIN_AREA = 5
STOP_MIN_AREA = 1005

# Inspection val and test image filenames for SV dataset
SV_FILENAMES_VAL_INSPECTION = [
    "good/004.png",
    "good/104.png",
    "good/154.png",
    "good/204.png",
    "good/304.png",
]

SV_FILENAMES_TEST_INSPECTION = [
    "good/010.png",
    "good/055.png",
    "good/080.png",
    "good/120.png",
    "good/240.png",
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

# Inspection val and test image filenames for TV dataset

TV_FILENAMES_VAL_INSPECTION = [
    "good/004.png",
    "good/104.png",
    "good/154.png",
    "good/204.png",
    "good/304.png",
]

TV_FILENAMES_TEST_INSPECTION = [
    "good/010.png",
    "good/055.png",
    "good/080.png",
    "good/120.png",
    "good/240.png",
    "B_added_tv/020.png",
    "B_added_tv/040.png",
    "B_added_tv/080.png",
    "B_added_tv/090.png",
    "B_added_tv/120.png",
    "B_added_tv/230.png",
    "B_missing_tv/010.png",
    "B_missing_tv/020.png",
    "B_missing_tv/030.png",
    "B_missing_tv/040.png",
    "B_missing_tv/080.png",
    "B_missing_tv/100.png",
    "B_missing_tv/110.png",
    "B_missing_tv/260.png",
    "B_missing_tv/270.png",
    "B_missing_tv/320.png",
    "B_shifted_tv/010.png",
    "B_shifted_tv/030.png",
    "B_shifted_tv/040.png",
    "B_shifted_tv/100.png",
    "B_shifted_tv/200.png",
    "B_shifted_tv/230.png",
    "C_color_1_tv/020.png",
    "C_color_1_tv/050.png",
    "C_color_1_tv/110.png",
    "C_color_2_tv/010.png",
    "C_color_2_tv/030.png",
    "C_color_2_tv/080.png",
    "C_color_3_tv/001.png",
    "C_color_3_tv/010.png",
    "C_color_3_tv/180.png",
]
