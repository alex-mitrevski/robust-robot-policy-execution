# Dataset
# DATASET_PATH = "/home/bsanth2s/AD_repo/Training_data_master_AD/train_n_5_batach_1_2_3"
DATASET_PATH="./datasets/train_n_5_batach_1_2_3"
# DataLoader
BATCH_SIZE = 64
NUM_WORKERS = 12

# Model
WARMUP_TEACHER_TEMP_EPOCHS = 10
LEARNING_RATE = 0.001

# Training
MAX_EPOCHS = 120
DEVICES = "auto"
STRATEGY = "ddp"
SYNC_BATCHNORM = True
REPLACE_SAMPLER_DDP = True
LOG_EVERY_N_STEPS = 20

# Logging
# LOGS_DIR = "/home/bsanth2s/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/monitoring/detecting_anomalies_ssl/anomaly_detection/trained_model/1_8"
LOGS_DIR = "./trained_model/1_8"
EXPERIMENT_NAME = "DINO_ep120"

# Feature Extraction (Nominal) image path represents the path for the training data, because we extract features from all trianing data
# CHECKPOINT_PATH = "/home/bsanth2s/AD_repo/trained_model_DINO/6_6_2024/DINO_ep100/version_0/checkpoints/epoch=99-step=2900.ckpt"
CHECKPOINT_PATH = "./checkpoints/epoch=99-step=2900.ckpt"
# IMAGE_PATH = "/home/bsanth2s/AD_repo/Training_data_master_AD/train_n_5_batach_1_2_3"
IMAGE_PATH="./datasets/train_n_5_batach_1_2_3"
# FEATURES_SAVE_PATH = "/home/bsanth2s/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/monitoring/detecting_anomalies_ssl/anomaly_detection/features/nominal_features.pt"
FEATURES_SAVE_PATH = "./features/nominal_features.pt"
# Image Preprocessing
RESIZE = 256
CENTER_CROP = 224
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]