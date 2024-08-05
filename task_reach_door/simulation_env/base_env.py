"""
DoorEnv class is a base class that is used by the door_env_with_joints.py  scripts

Author: Bharath Santhanam
Email: bharathsanthanamdev@gmail.com
Organization: Hochschule Bonn-Rhein-Sieg


Description:
This script contains the DoorEnv class which is a base class that is used by the door_env_with_joints.py
scripts. This class contains methods that are used to set the initial joint positions of the robot,
get the camera parameters, check for collisions between the robot and objects, draw target axes, and randomize colors.


References:
This script is based on:

Refered to the in-built functions of PyBullet using the official PyBullet documentation: https://pybullet.org/wordpress/index.php/forum-2/
Some part of this code is referred from the cluttered pushing repository: https://github.com/NilsDengler/cluttered-pushing/tree/main. Specific lines are mentioned in the code.

"""

import pybullet as p
import numpy as np
import yaml


class DoorEnv(object):
    def __init__(self, config_path):
        with open(config_path, "r") as config_file:
            self.config = yaml.safe_load(config_file)
        self.end_effector_link_index = self.config["robot"]["end_effector_link_index"]

    def get_camera_params(self):
        """
        Compute the view matrix and projection matrix of the camera
        """
        camera_link_index = self.config["camera"]["link_index"]

        # referred from https://github.com/NilsDengler/cluttered-pushing/blob/48171e76668656f911681c359dedf87f46922a52/push_gym/push_gym/environments/base_environment.py#L75C21-L75C23
        state = p.getLinkState(
            self.kinova, camera_link_index, computeForwardKinematics=True
        )
        camera_pos, camera_ori = state[0], state[1]

        camera_forward_vector = self.config["camera"]["forward_vector"]
        camera_up_vector = self.config["camera"]["up_vector"]
        camera_forward_vector = p.rotateVector(camera_ori, camera_forward_vector)
        camera_up_vector = p.rotateVector(camera_ori, camera_up_vector)

        camera_target_pos = [
            camera_pos[0] + camera_forward_vector[0],
            camera_pos[1] + camera_forward_vector[1],
            camera_pos[2] + camera_forward_vector[2],
        ]

        view_matrix = p.computeViewMatrix(
            camera_pos, camera_target_pos, camera_up_vector
        )
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.config["camera"]["projection_matrix"]["fov"],
            aspect=self.config["camera"]["projection_matrix"]["aspect"],
            nearVal=self.config["camera"]["projection_matrix"]["near_val"],
            farVal=self.config["camera"]["projection_matrix"]["far_val"],
        )
        return view_matrix, projection_matrix

    def check_collision(self, robot_id, objects_to_check):
        for obj_id in objects_to_check:
            contact_points = p.getContactPoints(bodyA=robot_id, bodyB=obj_id)
            if contact_points:
                return True
        return False

    def set_initial_joint_positions(self, joint_angles_deg):
        joint_angles_rad = [np.radians(angle) for angle in joint_angles_deg]
        joint_indices = range(7)

        # Referred from https://github.com/NilsDengler/cluttered-pushing/blob/48171e76668656f911681c359dedf87f46922a52/push_gym/push_gym/environments/ur5_environment.py#L153C12-L153C84
        for joint_index, joint_angle in zip(joint_indices, joint_angles_rad):
            p.resetJointState(self.kinova, joint_index, joint_angle)
        self.state_id = p.saveState()
        return self.state_id

    def draw_target_axes(self, target_pos1, target_pos2, target_pos3):
        axis_length = self.config["debug"]["axis_length"]
        axis_width = self.config["debug"]["axis_width"]
        for target_pos in [target_pos1, target_pos2, target_pos3]:
            x_end = target_pos + np.array([axis_length, 0, 0])
            y_end = target_pos + np.array([0, axis_length, 0])
            z_end = target_pos + np.array([0, 0, axis_length])

            p.addUserDebugLine(target_pos, x_end, [1, 0, 0], axis_width)
            p.addUserDebugLine(target_pos, y_end, [0, 1, 0], axis_width)
            p.addUserDebugLine(target_pos, z_end, [0, 0, 1], axis_width)

    def randomize_color(self, base_color):
        variation_range = self.config["domain_randomization"]["color_variation_range"]
        randomized_color = [
            np.clip(
                base_color[i] + np.random.uniform(-variation_range, variation_range),
                0,
                1,
            )
            for i in range(3)
        ] + [base_color[3]]  # Keep the alpha channel unchanged
        return randomized_color
