"""
Recovery Pipeline for Robotic Tasks

Author: Bharath Santhanam
Email: bahrathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script implements a recovery pipeline for door reaching task using Kinova arm. It includes
anomaly detection using DINO and KNN, and implements recovery strategies
such as pausing, perturbing, and using GMM for pose recovery.

Usage:
Follow the Readme.md file in the repository to run the recovery pipeline.

Dependencies:
- ROS Noetic
- Python 3.8
- PyTorch 1.9
- Stable Baselines3
- Kortex Drivers (for Kinova Gen3 robot)
"""

import rospy
from sensor_msgs.msg import Image, JointState
from cv_bridge import CvBridge, CvBridgeError
import cv2
import sys
import time
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import GetLinkState
from tf.transformations import euler_from_quaternion
from kortex_driver.msg import TwistCommand
import kortex_driver
from kortex_driver.srv import *
from kortex_driver.msg import *
import numpy as np
import ipdb
from stable_baselines3 import PPO, SAC
from DINO_model import DINO
import torch
from visualize_anomalies import extract_features, get_train_mean_std, normalize_features
from torchvision import transforms
from sklearn.cluster import KMeans
from scipy.spatial import distance
from PIL import Image as PILImage
from gmm import fitGMM, sample_from_gmm

# Load the features
from sklearn.neighbors import NearestNeighbors

# import ROS classes
from arm import Arm
from camera import Camera
from joints import JointPositions
from pose import Pose
import config


def get_anomaly_score(model_DINO, image_tensor, nominal_mean, nominal_std, knn_model):
    with torch.no_grad():
        features = model_DINO.extract_features(image_tensor)

    anomalous_features_np = normalize_features(
        features.cpu(), nominal_mean, nominal_std
    )

    # Find the distance to and index of the k-th nearest neighbors in the nominal data for each test feature
    distances, indices = knn_model.kneighbors(
        anomalous_features_np
    )  # anomalous_features  or normalized_anomalous_features

    # Use the distance to the k-th nearest neighbor as the anomaly score
    anomaly_scores = distances[:, 0]
    return anomaly_scores


def get_img_tensor(img_callback, save_path, i):
    image = img_callback.get_camera_data(resize_shape=(256, 256))
    # save PIL image in the disk
    image = image[:, :, ::-1]  # reverse the channels
    # convert the image numpy array to PIL
    image = PILImage.fromarray(image)
    # name the test image as "image_{i}"
    image.save(save_path + "image_{}.png".format(i))

    # open the saved image
    img = PILImage.open(save_path + "image_{}.png".format(i))

    image_tensor = preprocess(img).unsqueeze(0).to(device)
    return image_tensor, image


def main():
    rospy.init_node("integration")

    time.sleep(0.5)
    img = Camera()
    time.sleep(5)
    img.get_camera_data()
    arm = Arm()
    pose = Pose()
    img_callback = Camera()
    time.sleep(0.1)

    # initialize to the pose
    current_pose = pose.get_current_state()
    initial_pose = current_pose
    print(current_pose)

    model = PPO.load(config.model_path, use_sde=False, verbose=2)

    # TEST JOINT Positions
    joint = JointPositions()
    time.sleep(2)

    # load AD model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # model trained on batch 1,2 and 3
    model_DINO = DINO.load_from_checkpoint(
        checkpoint_path=config.DINO_CHECKPT_PATH,
    )
    model_DINO.to(device)
    model_DINO.eval()
    # Assuming you're using a CUDA device if available, else CPU
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    nominal_features_path = config.NOMINAL_FEATURES_PATH
    nom_features_tensor = extract_features(nominal_features_path)
    nominal_mean, nominal_std = get_train_mean_std(nom_features_tensor)
    nom_features_np = normalize_features(nom_features_tensor, nominal_mean, nominal_std)
    # compute the nearest neighbour
    # Initialize the NearestNeighbors model
    k = 5  # Number of neighbors to use for KNN
    knn_model = NearestNeighbors(n_neighbors=k)

    # Fit the model on the nominal features
    knn_model.fit(nom_features_np)  # nom_features or normalized_nominal_features

    # fit gmm model for recovery
    gmm = fitGMM(1)
    n_samples = 1
    sample = sample_from_gmm(gmm, n_samples)

    anomaly_scores_frames = []

    for i in range(120):
        current_pose = pose.get_current_state()
        print("current pose: ", current_pose)
        joint_positions = joint.get_current_joint_positions()
        print("joint positions: ", joint_positions)

        # concatenate the joint positions to the current pose

        end_effector_pose = np.array(current_pose[:3])
        joint_pos = np.array(joint_positions)
        sensor_features = np.concatenate((end_effector_pose, joint_pos))
        print("sensor features: ", sensor_features)

        img = img_callback.get_camera_data(resize_shape=(320, 180))
        img_features = img.transpose((2, 0, 1))
        obv_space = {"image": img_features, "vector": sensor_features}
        action, _states = model.predict(obv_space, deterministic=False)
        print("predicted action:", action)

        # get image tensor for AD model
        save_path = config.SAVE_IMG_ANOMSCORE_DIR
        image_tensor, im = get_img_tensor(img_callback, save_path, i)
        denorm_action = []
        lower_limits = [-0.02, -0.02, -0.02]
        upper_limits = [0.02, 0.02, 0.02]
        for num, low, high in zip(action, lower_limits, upper_limits):
            new = np.interp(num, (-1, 1), (low, high))
            denorm_action.append(new)
        print("denorm action: ", denorm_action)

        # add the action which is offsets, to the current pose (x,y,z)
        current_pose[:3] = [sum(x) for x in zip(current_pose[:3], denorm_action)]
        print("current pose: ", current_pose)
        # ipdb.set_trace()
        action_completed = arm.move_to_pose(current_pose)
        time.sleep(1)

        print("action state: ", action_completed)
        if action_completed:
            # # detect anomalies (finish this funcitons)

            # get anomaly score
            anomaly_scores = get_anomaly_score(
                model_DINO, image_tensor, nominal_mean, nominal_std, knn_model
            )
            im.save(f"{save_path}image_{i}_{anomaly_scores[0]:.2f}.png")
            # im.save(save_path + "image_{}.png".format(anomaly_scores[0]))
            anomaly_scores_frames.append(anomaly_scores)
            print(
                "--------Anamoly scores-----------------------------------------",
                anomaly_scores,
            )
            # ipdb.set_trace()
            if len(anomaly_scores_frames) == 2:
                count = 0
                # if all three elements in the list are above 25, flag it as anomalies
                for i in anomaly_scores_frames:
                    if i > 29.7:
                        print("anomaly scores for successive frames:", i)
                        count += 1

                    if count == 2:
                        # time.sleep(6)
                        print("Anomaly detected")

                        # wait for 4 seconds
                        print("recovery mode 1: pause for 4 seconds")
                        time.sleep(4)

                        # check for anomalies again:
                        print("Checking for anomalies again")
                        image_tensor, im = get_img_tensor(img_callback, save_path, i)
                        anomaly_scores = get_anomaly_score(
                            model_DINO,
                            image_tensor,
                            nominal_mean,
                            nominal_std,
                            knn_model,
                        )
                        # save the image as image_i_anomaly_scores[0]
                        im.save(f"{save_path}image_{i}_{anomaly_scores[0]:.2f}.png")
                        # im.save(save_path + "image_{}.png".format(anomaly_scores[0]))
                        print(
                            "anomaly scores after recovery mode 1:pause",
                            anomaly_scores,
                        )
                        if anomaly_scores > 29.7:
                            print("recovery mode 2: Wiggle the arm by 2 cm")
                            # sample wiggle directions
                            # wiggle the arm by 2 cm in x axis

                            # choose x,y or z in random
                            # choose the direction in random
                            direction = np.random.choice([0, 1, 2])
                            current_pose[direction] += 0.02
                            action_completed = arm.move_to_pose(current_pose)
                            time.sleep(3)
                            if action_completed:
                                print("Check for anomalies again")
                                image_tensor, im = get_img_tensor(
                                    img_callback, save_path, i
                                )
                                anomaly_scores = get_anomaly_score(
                                    model_DINO,
                                    image_tensor,
                                    nominal_mean,
                                    nominal_std,
                                    knn_model,
                                )
                                im.save(
                                    f"{save_path}image_{i}_{anomaly_scores[0]:.2f}.png"
                                )
                                # im.save(save_path + "image_{}.png".format(anomaly_scores[0]))
                                print(
                                    "anomaly scores after recovery mode 2:wiggle",
                                    anomaly_scores,
                                )
                                if anomaly_scores > 29.7:
                                    print(
                                        "Recovery mode 3: Move to recovery pose using GMM"
                                    )
                                    # recovery using gmm
                                    sample = sample_from_gmm(gmm, n_samples)

                                    recovery_pose = (
                                        sample[0][0].tolist() + config.orientation
                                    )
                                    print("Recovery pose:", recovery_pose)
                                    action_completed = arm.move_to_pose(recovery_pose)
                                    # action_completed = arm.move_to_pose(config.home_position)
                                    time.sleep(6)
                                    if action_completed:
                                        print("Reset to initial pose")
                                        break
                                    else:
                                        print("Recovery failed")
                                        break
                            else:
                                break

                anomaly_scores_frames = []

            continue
        else:
            break


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Recovery pipeline error")
    except Exception as e:
        rospy.logerr(f"error occurred: {str(e)}")
