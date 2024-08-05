"""
Camera class for the Kinova Gen3 robot to get the camera data.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script implements the Camera class for the Kinova Gen3 robot to get the camera data.
The class is used in the recovery pipeline to get the camera data.

Usage:
Follow the Readme.md file in the repository to run the recovery pipeline.

Dependencies:
- ROS Noetic
- Python 3.8
- Kortex Drivers (for Kinova Gen3 robot)
- ROS Kortex vision module to connect to the camera (https://github.com/Kinovarobotics/ros_kortex_vision)


References:
Relevant ROS topics are found using the ROS Kortex vision documentation
https://github.com/Kinovarobotics/ros_kortex_vision
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
from scipy.spatial import distance
from PIL import Image as PILImage


class Camera:
    def __init__(self):
        self.camera_data = None
        self.bridge = CvBridge()

        self.subscriber = rospy.Subscriber(
            "/camera/color/image_raw", Image, self.camera_callback
        )

    def camera_callback(self, data):
        # Convert ROS image message to OpenCV image format
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data)
            self.camera_data = cv2.cvtColor(
                cv_image, cv2.COLOR_BGR2RGB
            )  # Convert to RGB
        except CvBridgeError as e:
            print(e)

    def get_camera_data(self, resize_shape=(320, 180)):
        if self.camera_data is not None:
            resized_image = cv2.resize(self.camera_data, resize_shape)

        return resized_image
