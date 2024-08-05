"""
Class to extract end effector positions from the robot.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg

References:
Relevant topics to use to extract end effector information is obtained from the official ROS Kortex documentation
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


class Pose:
    def __init__(self):
        self.current_state = None
        self.enf_effector_subscriber = rospy.Subscriber(
            "/my_gen3/base_feedback",
            BaseCyclic_Feedback,
            self.end_effector_positions_Callback,
        )

    def end_effector_positions_Callback(self, data):
        self.current_state = data

    def get_current_state(self):
        # This method returns the latest end effector positions
        if self.current_state is not None:
            tool_pose_x = self.current_state.base.tool_pose_x
            tool_pose_y = self.current_state.base.tool_pose_y
            tool_pose_z = self.current_state.base.tool_pose_z
            tool_pose_theta_x = self.current_state.base.tool_pose_theta_x
            tool_pose_theta_y = self.current_state.base.tool_pose_theta_y
            tool_pose_theta_z = self.current_state.base.tool_pose_theta_z
            return [
                tool_pose_x,
                tool_pose_y,
                tool_pose_z,
                tool_pose_theta_x,
                tool_pose_theta_y,
                tool_pose_theta_z,
            ]
        else:
            return None
