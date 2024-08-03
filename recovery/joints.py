"""
Class to extract joint positions from the robot.

Author: Bharath Santhanam
Email:bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
Relevant topics to use to extract joint information is obtained from the official ROS Kortex documentation
https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_driver

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


class JointPositions:
    def __init__(self):
        self.current_state = None
        self.joint_positions_subscriber = rospy.Subscriber(
            "/my_gen3/joint_states",
            JointState,
            self.joint_positions_Callback,
        )

    def joint_positions_Callback(self, data):
        self.current_state = data.position[:7]

    def get_current_joint_positions(self):
        if self.current_state is not None:
            joint_positions = self.current_state
            return joint_positions
        else:
            return None
