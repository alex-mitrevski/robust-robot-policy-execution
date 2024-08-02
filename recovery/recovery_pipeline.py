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

rospy.init_node("integration")


#constants
##########################################################################################################################################################################
# PPO
model_path = "/home/bharath/ROS_RL/ROS_RL/saved_models/position_control/v5_working/PPO_without_entropy_with_DR/gpu_100000_steps.zip"
home_position = [
    0.48758944869041443,
    -0.07694503664970398,
    0.4082964062690735,
    40.666446685791016,
    -97.42293548583984,
    139.19334411621094,
]
orientation = [40.666446685791016, -97.42293548583984, 139.19334411621094]
DINO_CHECKPT_PATH="/home/bharath/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/ROS_integration/AD_trained_model/trained_model_DINO_7_6_2024/6_6_2024/DINO_ep100/version_0/checkpoints/epoch=99-step=2900.ckpt"
# trainfeatures on batch 1 and 2
# nominal_features_path = "/home/bharath/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/ROS_integration/train_features.pt"
# nominal_features on batch 1,2 and 3
NOMINAL_FEATURES_PATH= "/home/bharath/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/ROS_integration/nominal_features.pt"


########################################################################################################################################################################
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

        except rospy.ServiceException as e:
            print("Service call failed:", e)

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
        cartesian_speed.translation = 0.9  # m/s translation_speed
        cartesian_speed.orientation = 18  # deg/s orientation_speed
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

    def get_camera_data(self, resize_shape=(320, 180)):
        # self.camera_data = cv2.imread('/home/pusht.png')
        if self.camera_data is not None:
            resized_image = cv2.resize(self.camera_data, resize_shape)

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

model = PPO.load(model_path, use_sde=False, verbose=2)

# TEST JOINT Positions
joint = JointPositions()
time.sleep(2)


# load AD model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# model trained on batch 1,2 and 3
model_DINO = DINO.load_from_checkpoint(
    checkpoint_path=DINO_CHECKPT_PATH,
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


nominal_features_path = NOMINAL_FEATURES_PATH
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
    save_path = "/home/bharath/MA_Thesis_Master_repo/Robot_task_execution_monitoring_and_recovery/ROS_integration/save_rollouts/"
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
                        model_DINO, image_tensor, nominal_mean, nominal_std, knn_model
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
                            im.save(f"{save_path}image_{i}_{anomaly_scores[0]:.2f}.png")
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
                                orientation = [
                                    40.666446685791016,
                                    -97.42293548583984,
                                    139.19334411621094,
                                ]
                                recovery_pose = sample[0][0].tolist() + orientation
                                print("Recovery pose:", recovery_pose)
                                action_completed = arm.move_to_pose(recovery_pose)
                                # action_completed = arm.move_to_pose(home_position)
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
