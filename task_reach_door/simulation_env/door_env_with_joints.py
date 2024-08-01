import pybullet as p
import pybullet_data
import time
import numpy as np
from .base_env import DoorEnv
import gymnasium
from gymnasium import spaces

class ReachDoor(DoorEnv, gymnasium.Env):
    def __init__(self, config_path="config/config_simulation.yaml") -> None:
        super().__init__(config_path)
        if not p.isConnected():
            p.connect(p.DIRECT if self.config['env']['headless'] else p.GUI)
            p.setTimeStep(1.0 / 240.0)
        p.setPhysicsEngineParameter(enableFileCaching=0)
        p.resetSimulation()

        self.target_pos1 = np.array(self.config['target']['pos1'])
        self.target_pos2 = np.array(self.config['target']['pos2'])
        self.target_pos3 = np.array(self.config['target']['pos3'])
        self.create_env()

        self.end_effector_link_index = self.config['robot']['end_effector_link_index']
        self.joint_angles_deg = self.config['robot']['initial_joint_angles']

        self.state_id = self.set_initial_joint_positions(self.joint_angles_deg)

        self.initial_distance_to_goal = self.reward_distance_to_goal()
        initial_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        self.initial_end_effector_pos = np.array(initial_end_effector_state[4])
        self.initial_end_effector_ori = np.array(initial_end_effector_state[5])

        self.observation_space = spaces.Dict(
            {
                "image": spaces.Box(low=0, high=255, shape=tuple(self.config['observation']['image_shape']), dtype=np.uint8),
                "vector": spaces.Box(
                    low=np.array(self.config['observation']['vector_low']),
                    high=np.array(self.config['observation']['vector_high']),
                    shape=(10,),
                    dtype=float,
                ),
            }
        )

        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=float)

        self.episode_length = self.config['env']['episode_length']
        self.step_counter = 0

    def reset_base_simulation(self):
        p.resetSimulation()

    def create_env(self):
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        table_height = self.config['scene']['table_height']
        robot_base_height = self.config['robot']['base_height']

        robot_start_pos = [0, 0, table_height + robot_base_height]
        robot_start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.kinova = p.loadURDF(
            self.config['robot']['urdf_path'],
            basePosition=robot_start_pos,
            baseOrientation=robot_start_orientation,
            useFixedBase=True,
        )

        self.set_robot_color()

        self.planeId = p.loadURDF(self.config['scene']['plane_urdf'])
        self.tableId = p.loadURDF(
            self.config['scene']['table_urdf'], 
            self.config['scene']['table_start_pos'], 
            p.getQuaternionFromEuler([0, 0, 0])
        )
        self.door = p.loadURDF(
            self.config['scene']['door_urdf'],
            self.config['scene']['door_start_pos'],
            p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True,
        )

        self.draw_target_axes(self.target_pos1, self.target_pos2, self.target_pos3)

    def set_robot_color(self):
        robot_color = self.config['colors']['robot']
        num_joints = p.getNumJoints(self.kinova)
        for link_index in range(-1, num_joints):
            p.changeVisualShape(self.kinova, link_index, rgbaColor=robot_color)

    def apply_color_randomization(self):
        lever_color = self.config['colors']['lever']
        door_color = self.config['colors']['door']
        protrusion_color = self.config['colors']['protrusion']

        random_lever_color = self.randomize_color(lever_color)
        random_door_color = self.randomize_color(door_color)
        random_protrusion_color = self.randomize_color(protrusion_color)

        p.changeVisualShape(self.door, self.config['visual_shape']['lever'], rgbaColor=random_lever_color)
        p.changeVisualShape(self.door, self.config['visual_shape']['door'], rgbaColor=random_door_color)
        p.changeVisualShape(self.door, self.config['visual_shape']['protrusion'], rgbaColor=random_protrusion_color)
        p.changeVisualShape(self.door, self.config['visual_shape']['knob'], rgbaColor=random_lever_color)

    def step(self, action):
        done = False
        truncated = False
        info = {}
        current_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        current_end_effector_pose = np.array(current_end_effector_state[4])

        prev_distance_to_goal = self.reward_distance_to_goal()

        denorm_action = []
        for num, low, high in zip(action, self.config['action']['lower_limits'], self.config['action']['upper_limits']):
            new = np.interp(num, [-1, 1], [low, high])
            denorm_action.append(new)

        target_position = current_end_effector_pose + np.array(denorm_action)
        target_joint_positions = p.calculateInverseKinematics(
            self.kinova,
            self.end_effector_link_index,
            target_position,
            self.initial_end_effector_ori,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )

        for joint_index in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.kinova,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target_joint_positions[joint_index],
            )

        for i in range(15):
            p.stepSimulation()
            time.sleep(0.01)
            collision = self.check_collision(
                robot_id=self.kinova,
                objects_to_check=[self.door, self.tableId, self.planeId],
            )

            if collision:
                print("Collision detected")

            time.sleep(1.0 / 240.0)

        success_reward, success = self.reward_success()
        distance_reward = (
            prev_distance_to_goal - self.reward_distance_to_goal()
        ) / self.initial_distance_to_goal
        reward = (distance_reward * self.config['reward']['distance_reward_multiplier']) + success_reward
        
        if self.step_counter % self.episode_length == 0:
            done = False
            truncated = True
            print("step count after completing all steps", self.step_counter)
            info = {"Cause": "Reset timeout"}
            if self.config['env']['use_domain_randomization']:
                self.apply_color_randomization()

        if success:
            done = True
            info = {"Cause": "Success"}
            print("reward after success:", reward)
            if self.config['env']['use_domain_randomization']:
                self.apply_color_randomization()

        if collision:
            done = True
            reward += self.config['reward']['collision_penalty']
            info = {"Cause": "Collision"}
            if self.config['env']['use_domain_randomization']:
                self.apply_color_randomization()

        self.step_counter += 1
        print("step counter during step", self.step_counter)
        print("reward:", reward)

        return self._get_state(), reward, done, truncated, info

    def reward_distance_to_goal(self):
        current_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        current_end_effector_pose = np.array(current_end_effector_state[4])
        dist1 = np.linalg.norm(self.target_pos1 - current_end_effector_pose)
        dist2 = np.linalg.norm(self.target_pos2 - current_end_effector_pose)
        dist3 = np.linalg.norm(self.target_pos3 - current_end_effector_pose)

        return min(dist1, dist2, dist3)

    def reward_success(self):
        current_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        dist_threshold = self.config['reward']['success_threshold']
        success = False
        success_reward = self.config['reward']['default_success_reward']

        if (np.linalg.norm(self.target_pos1 - current_end_effector_state[4]) < dist_threshold) or \
           (np.linalg.norm(self.target_pos2 - current_end_effector_state[4]) < dist_threshold) or \
           (np.linalg.norm(self.target_pos3 - current_end_effector_state[4]) < dist_threshold):
            success = True
            success_reward = self.config['reward']['success_reward']
        return success_reward, success

    def reset(self, seed=666):
        info = {}
        self.setup_scene()
        return self._get_state(), info

    def setup_scene(self):
        p.restoreState(self.state_id)
        current_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        current_end_effector_pose = np.array(current_end_effector_state[4])
        random_offset = np.random.uniform(
            self.config['domain_randomization']['initial_pose_variation'][0],
            self.config['domain_randomization']['initial_pose_variation'][1],
            size=3
        )
        random_end_effector_pose = current_end_effector_pose + random_offset
        rand_start_joint_positions = p.calculateInverseKinematics(
            self.kinova,
            self.end_effector_link_index,
            random_end_effector_pose,
            self.initial_end_effector_ori,
            maxNumIterations=100,
            residualThreshold=1e-5,
        )
        for joint_index in range(7):
            p.setJointMotorControl2(
                bodyUniqueId=self.kinova,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=rand_start_joint_positions[joint_index],
            )
        for _ in range(50):
            p.stepSimulation()
            time.sleep(0.02)

    def _get_state(self):
        view_matrix, projection_matrix = self.get_camera_params()
        width, height, rgb_img, depth_img, seg_img = p.getCameraImage(
            width=self.config['camera']['width'],
            height=self.config['camera']['height'],
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
        )
        rgb_array = np.reshape(rgb_img, (height, width, 4))
        rgb_only = rgb_array[:, :, :3].transpose((2, 0, 1))

        current_end_effector_state = p.getLinkState(
            self.kinova, self.end_effector_link_index, computeForwardKinematics=True
        )
        current_end_effector_pose = np.array(current_end_effector_state[4])
        
        current_end_effector_pose[2] -= self.config['state']['end_effector_offset']
        current_end_effector_pose = np.clip(
            current_end_effector_pose, 
            self.config['state']['clip_low'], 
            self.config['state']['clip_high']
        )

        current_joint_states = p.getJointStates(self.kinova, list(range(7)))
        current_joint_positions = np.array([state[0] for state in current_joint_states])
        
        kinova_state = np.concatenate((current_end_effector_pose, current_joint_positions))
        obv_space = {"image": rgb_only, "vector": kinova_state}

        return obv_space

    def shutdown(self):
        p.disconnect()