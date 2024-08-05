"""
Arm class for Kinova Gen3 robot to move the robot to a specified cartesian pose.

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script implements the Arm class for the Kinova Gen3 robot to move the robot to a specified cartesian pose.
The class is used in the recovery pipeline to move the robot to a specified cartesian pose.

Usage:
Follow the Readme.md file in the repository to run the recovery pipeline.

Dependencies:
- ROS Noetic
- Python 3.8
- Kortex Drivers (for Kinova Gen3 robot)

References:
This script is based on:
 https://github.com/HBRS-SDP/ws23-door-opening/blob/main/src/door_opening/src/scripts/force_transformation.py
 https://github.com/HBRS-SDP/ws23-door-opening/blob/main/src/door_opening/src/scripts/open_door.py

The below class "Arm" is completely adapted from the above scripts. Specific lines referred are mentioned in the code.

source gitHub Repository: ws23-door-opening
source repo Authors: PRan3, Maira Liaqat, Oviya Rajavel.
URL: https://github.com/HBRS-SDP/ws23-door-opening/tree/main
Accessed on: 21/5/2024


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


class Arm:
    def __init__(self):
        self.last_action_notif_type = None
        self.action_topic_sub = None
        self.all_notifs_succeeded = True

        rospy.wait_for_service("/my_gen3/base/execute_action")
        self.execute_action = rospy.ServiceProxy(
            "/my_gen3/base/execute_action", ExecuteAction
        )
        rospy.wait_for_service("/my_gen3/base/activate_publishing_of_action_topic")
        self.activate_publishing_of_action_notification = rospy.ServiceProxy(
            "/my_gen3/base/activate_publishing_of_action_topic",
            OnNotificationActionTopic,
        )

        request = OnNotificationActionTopicRequest(input=NotificationOptions(0, 0, 0))

        try:
            response = self.activate_publishing_of_action_notification(request)

        except rospy.ServiceException as e:
            print("Service call failed:", e)

        self.action_topic_sub = rospy.Subscriber(
            "/my_gen3/action_topic", ActionNotification, self.cb_action_topic
        )

        self.agent_pose = []
        self.block_pose = []

    # function directly from https://github.com/HBRS-SDP/ws23-door-opening/blob/c494867b53a7507a0fb48378265129844ab9d82a/src/door_opening/src/scripts/open_door.py#L175
    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event
    
    # function adapted from 
    # https://github.com/HBRS-SDP/ws23-door-opening/blob/c494867b53a7507a0fb48378265129844ab9d82a/src/door_opening/src/scripts/open_door.py#L221
    def move_to_pose(self, pose):
        """
        Move the robot to the specified pose
        Takes in a list of 6 elements [x, y, z, tx, ty, tz]
        Output: True if the action is completed successfully, False otherwise

        """
        x, y, z, tx, ty, tz = pose
        command = ConstrainedPose()
        command.target_pose.x = x
        command.target_pose.y = y
        command.target_pose.z = z
        command.target_pose.theta_x = tx
        command.target_pose.theta_y = ty
        command.target_pose.theta_z = tz

        req1 = SetCartesianReferenceFrameRequest()
        req1.input.reference_frame = (
            CartesianReferenceFrame.CARTESIAN_REFERENCE_FRAME_MIXED
        )
        cartesian_speed = CartesianSpeed()
        cartesian_speed.translation = 0.9  # m/s linear_speed
        cartesian_speed.orientation = 18  # deg/s angular_speed
        command.constraint.oneof_type.speed.append(cartesian_speed)
        req = ExecuteActionRequest()
        req.input.oneof_action_parameters.reach_pose.append(command)
        req.input.handle.action_type = ActionType.REACH_POSE
        req.input.handle.identifier = 1001
        rospy.loginfo("Sending pose ...")
        try:
            self.execute_action(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to send  current pose ")
        else:
            rospy.loginfo("Waiting for the current pose to finish...")
        action_completed = self.wait_for_action_end_or_abort()
        return action_completed

    # function directly from
    # https://github.com/HBRS-SDP/ws23-door-opening/blob/c494867b53a7507a0fb48378265129844ab9d82a/src/door_opening/src/scripts/open_door.py#L178
    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)
