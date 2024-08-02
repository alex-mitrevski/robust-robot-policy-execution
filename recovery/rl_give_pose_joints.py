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

rospy.init_node("integration")


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
        # Create the request object and set its fields
        request = OnNotificationActionTopicRequest(input=NotificationOptions(0, 0, 0))

        try:
            # Call the service and receive the response
            response = self.activate_publishing_of_action_notification(request)
            # print(
            #     "Service call response:", response.output
            # )  # Assuming 'output' is a field in the response
        except rospy.ServiceException as e:
            print("Service call failed:", e)

        # rospy.wait_for_service("/my_gen3/base/send_gripper_command")
        # self.send_gripper_command = rospy.ServiceProxy(
        #     "/my_gen3/base/send_gripper_command", SendGripperCommand
        # )

        # rospy.wait_for_service("/gazebo/get_link_state")
        # self.get_link_state = rospy.ServiceProxy("/gazebo/get_link_state", GetLinkState)
        # Activate publishing of action notifications
        # try:
        #     self.activate_publishing_of_action_notification(
        #         NotificationOptions
        #     )  # Assuming True activates it; check the expected parameter
        #     rospy.loginfo("Activated publishing of action notifications.")
        # except rospy.ServiceException as e:
        #     rospy.logerr(
        #         "Failed to activate publishing of action notifications: %s" % e
        #     )

        self.action_topic_sub = rospy.Subscriber(
            "/my_gen3/action_topic", ActionNotification, self.cb_action_topic
        )

        self.agent_pose = []
        self.block_pose = []
        self.offset = 0.05
        self.scale = 1024

    def cb_action_topic(self, notif):
        self.last_action_notif_type = notif.action_event

    def get_block_pose(self, link_name, frame_name):
        try:
            block_pose_full = self.get_link_state(link_name, frame_name)
            x = block_pose_full.link_state.pose.position.x
            y = block_pose_full.link_state.pose.position.y
            self.block_pose = [x * arm.scale, y * arm.scale]
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)

    def move_to_pose(self, pose):
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
        cartesian_speed.translation = 0.8  # m/s translation_speed
        cartesian_speed.orientation = 15  # deg/s orientation_speed
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
        # self.agent_pose = [x * self.scale, y * self.scale]

    def wait_for_action_end_or_abort(self):
        while not rospy.is_shutdown():
            # print(ActionEvent.ACTION_END, self.last_action_notif_type)
            if self.last_action_notif_type == ActionEvent.ACTION_END:
                rospy.loginfo("Received ACTION_END notification")
                return True
            elif self.last_action_notif_type == ActionEvent.ACTION_ABORT:
                rospy.loginfo("Received ACTION_ABORT notification")
                self.all_notifs_succeeded = False
                return False
            else:
                time.sleep(0.01)

    def gripper(self, value):
        # Initialize the request
        # Close the gripper
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = value
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION
        rospy.loginfo("Sending the gripper command...")
        # Call the service
        try:
            self.send_gripper_command(req)
        except rospy.ServiceException:
            rospy.logerr("Failed to call SendGripperCommand")
            return False
        else:
            time.sleep(0.5)
            return True


class Camera:
    def __init__(self):
        self.camera_data = None
        self.bridge = CvBridge()
        # Assuming camera topic publishes image messages
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

    def get_camera_data(self):
        # self.camera_data = cv2.imread('/home/pusht.png')
        if self.camera_data is not None:
            resized_image = cv2.resize(self.camera_data, (320, 180))

        return resized_image


class JointPositions:
    def __init__(self):
        self.current_state = None
        self.joint_positions_subscriber = rospy.Subscriber(
            "/my_gen3/joint_states",
            JointState,
            self.joint_positions_Callback,
        )

    def joint_positions_Callback(self, data):
        # rospy.loginfo("Received joint positions!")
        # print(data.position)
        self.current_state = data.position[:7]

    def get_current_joint_positions(self):
        if self.current_state is not None:
            joint_positions = self.current_state
            return joint_positions
        else:
            return None


class Pose:
    def __init__(self):
        self.current_state = None  # Store the latest state here
        self.enf_effector_subscriber = rospy.Subscriber(
            "/my_gen3/base_feedback",
            BaseCyclic_Feedback,
            self.end_effector_positions_Callback,
        )

    def end_effector_positions_Callback(self, data):
        # Store the latest data instead of printing it
        # rospy.loginfo("Callback received data")
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
            return None  # or raise an exception, or return a default value


time.sleep(0.5)

img = Camera()
time.sleep(7)
img.get_camera_data()
arm = Arm()
pose = Pose()
img_callback = Camera()
time.sleep(0.1)

# working code for trial
"""
for i in range(10):
    current_pose = pose.get_current_state()
    print(current_pose)
    ipdb.set_trace()
    current_pose[0] = current_pose[0] + 0.02
    target_pose = np.array(current_pose)
    action_completed = arm.move_to_pose(target_pose)
    time.sleep(1)

    print("action state: ", action_completed)
    if action_completed:
        img = img_callback.get_camera_data()
        cv2.imwrite("img1.png", img)
        # img_uint = (img * 255).astype(np.uint8)
        # img_PIL = Image.fromarray(convert_PIL)
        # img_array = np.array(img)
        # ipdb.set_trace()
        img_features = img.transpose((2, 0, 1))
        print(img_features.shape)
        break
        # cv2.imwrite(str(i) + ".png", img)
        continue
    else:
        break
    # pass the camera image and the current pose to the model to get feedback on actions
"""
# model_path = "/home/bharath/ROS_RL/ROS_RL/saved_models/position_control/v4_working/gpu_4000_steps.zip"
# PPO
# model_path = "/media/bharath/PortableSSD/sim2real/position_controlv5/PPO_with_Entropy/820a45_PPO10_entropy/gpu_100000_steps.zip"
model_path = "/home/bharath/ROS_RL/ROS_RL/saved_models/position_control/v5_working/PPO_without_entropy_with_DR/gpu_100000_steps.zip"  # this is with DR and works perfectly!
# model_path = "/home/bharath/ROS_RL/ROS_RL/saved_models/position_control/v5_working/PPO_without_entropy_without_DR/PPO_without_entropy/fd7d95_PPO2/gpu_100000_steps.zip"
# initialize to the pose
current_pose = pose.get_current_state()
print(current_pose)
# initial_pose = [0.4534, -0.0481, 1.0634 - 0.6]
# current_pose[:3] = [sum(x) for x in zip(current_pose[:3], initial_pose)]
# arm.move_to_pose(current_pose)
# time.sleep(5)
model = PPO.load(model_path, use_sde=False, verbose=2)

# TEST JOINT Positions
joint = JointPositions()
time.sleep(2)


for i in range(30):
    current_pose = pose.get_current_state()
    print("step number and current pose: ", i, current_pose)
    joint_positions = joint.get_current_joint_positions()
    # print("joint positions: ", joint_positions)

    # concatenate the joint positions to the current pose

    end_effector_pose = np.array(current_pose[:3])
    joint_pos = np.array(joint_positions)
    sensor_features = np.concatenate((end_effector_pose, joint_pos))
    # print("sensor features: ", sensor_features)

    img = img_callback.get_camera_data()
    img_features = img.transpose((2, 0, 1))
    obv_space = {"image": img_features, "vector": sensor_features}

    action, _states = model.predict(obv_space, deterministic=False)
    # print("predicted action:", action)
    denorm_action = []
    lower_limits = [-0.02, -0.02, -0.02]
    upper_limits = [0.02, 0.02, 0.02]
    for num, low, high in zip(action, lower_limits, upper_limits):
        new = np.interp(num, (-1, 1), (low, high))
        denorm_action.append(new)
    # print("denorm action: ", denorm_action)

    # add the action which is offsets, to the current pose (x,y,z)
    current_pose[:3] = [sum(x) for x in zip(current_pose[:3], denorm_action)]
    # print("current pose: ", current_pose)
    # ipdb.set_trace()
    action_completed = arm.move_to_pose(current_pose)
    time.sleep(1)

    print("action state: ", action_completed)
    if action_completed:
        continue
    else:
        break
