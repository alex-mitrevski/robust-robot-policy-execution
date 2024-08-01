### Anomaly detection

#### Training 

The training parameters are in the config file, `anomaly_detection/config.py`

Adjust the datasets paths, checkpoint save path in the config file. The paths to be adjusted are given in the config.py. The training dataset should not contain any anomalies

```
cd anomaly_detection
python3 train_DINO.py
```

#### Extracting training features

We use the trained model to extract features from all training data. 

Adjust the dataset paths, checkpoint save path in the config file if required. `anomaly_detection/config.py`

```
cd anomaly_detection
python3 extract_nom_features.py
```


#### Threshold estimation

During testing, we extract features of the test data and find the nearest neighbour to the training features estimated in the previous step. We have a labeled validation dataset containing both nominal and anomalous frames, which we use to extract features and plot Precision-Recall curve for a wide range of thresholds to estimate the thresholds.

Extract  features for the validation dataset. Use `inference_config.py` to adjust the parameters and paths if required

```
cd inference
python3 extract_val_features.py
```

To estimate the thresholds,

```
cd inference
python3 calculae_thresholds.py
```


#### Estimate performance on test data (Not required if directly deployed on the robot)

We deploy the anomaly detection pipeline directly on the robot. We pass each test image to the trained model to extract features and find the nearest neighbour in the training data features. If the distance exceeds thresholds estimated in the previous step, we flag that image as anomalous. However, if you want to test it out on a test data, 

```
cd inference
python3 extract_val_features.py
python3 calculate_metrics.py

```
