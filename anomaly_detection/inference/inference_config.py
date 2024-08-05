# Inference Configuration

# Paths, thresholds estimation
DATASET_DIR = "./dataset_threshold_estimation"
MODEL_DIR = "../checkpoints/checkpoints_ubuntu22/epoch=99-step=2000.ckpt"
NOMINAL_FEATURES_PATH = "../features/trained_features_v1/train_features.pt"
ANOM_FRAMES_JSON_PATH = "./anom_frames_json/dataset_threshold_calculation.json"
SAVE_FEATURES_DIR = "./save_val_features"


THRESH_CALC_DATASET_PATH = "./save_val_features"


# find thresholds using the scripts, calculate_threshold.py and enter the threshold value here
THRESHOLD = 29.72

# dataset path for which the metrics needs to be calculated
METRICS_DATASET_PATH = "./save_val_features"
