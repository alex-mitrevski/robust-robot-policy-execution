

### References

We present a high-level overview of various sources we referred to during the implementation. Specific details of reference are also mentioned in the code scripts.

### References for task execution : reaching the door handle

1. The URDF for Kinova arm is from the official Kinova kortex repository and it is available in `./urdf/kortex_description`

   ```
   https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/arms/gen3/7dof/urdf
   https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/grippers/robotiq_2f_85/urdf
   ```
2. The door and the handle is designed using the online CAD tool and imported to the URDF using the STL format. `https://www.onshape.com/en/`
3. We referred to the following repo to get the structure of our scripts

   | scripts in this repo                                                                                       | source repo ( Git repo we referred to)                                                                                                                                                                                   | source script (scripts in the git repo we referred to)                                                                                                                                                                                      |
   | :--------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
   | ./task_reach_door/vmp_trainer.py<br />./task_reach_door/evaluate.py                                        | https://github.com/NJ-2020-thesis/PyRep/tree/feature/examples/vmp                                                                                                                                                        | https://github.com/NJ-2020-thesis/PyRep/blob/feature/examples/vmp/vmp_trainer.py                                                                                                                                                            |
   | ./task_reach_door/simulation_env/base_env.py<br />./task_reach_door/simulation_env/door_env_with_joints.py | https://github.com/NJ-2020-thesis/PyRep/blob/feature/examples/vmp/<br />https://github.com/NilsDengler/cluttered-pushing/tree/main                                                                                       | https://github.com/NJ-2020-thesis/PyRep/blob/feature/examples/vmp/vmp_environment.py<br />https://github.com/NilsDengler/cluttered-pushing/blob/48171e76668656f911681c359dedf87f46922a52/push_gym/push_gym/environments/base_environment.py |
   | ./task_reach_door/urdf/kortex_description                                                                  | https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/arms/gen3/7dof/urdf<br />https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/grippers/robotiq_2f_85/urdf |                                                                                                                                                                                                                                             |
4. Official PyBullet documentation:  `https://pybullet.org/wordpress/index.php/forum-2/`
5. Official gymanasium documentation: `https://gymnasium.farama.org/`
6. Official Stable-baselines3 documentation: `https://stable-baselines3.readthedocs.io/en/master/`

 Specific usages of above references are also given as comments in the code scripts.


### References for anomaly_detection

1. LightlySSL framework for DINO implementation. LightlySSL uses [MIT Licence](https://github.com/lightly-ai/lightly/blob/master/LICENSE.txt), Lightly SSL framework link: https://github.com/lightly-ai/lightly/tree/master

| scripts in the repo                                                                                                                        | URL/ link to source repo ( Git repo we referred to)                                       | source script (scripts in the git repo we referred to)                               |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------ |
| ./anomaly_detection/train_DINO.py<br />./anomaly_detection/extract_nom_features.py<br />./anomaly_detection/inference/extract_feat_test.py | https://github.com/lightly-ai/lightly/tree/master                                         | https://github.com/lightly-ai/lightly/blob/master/examples/pytorch_lightning/dino.py |
| ./anomaly_detection/inference/calculate_metrics.py<br />./anomaly_detection/inference/calculate_threshold.py                               | https://scikit-learn.org/stable/api/sklearn.metrics.html                                  |                                                                                      |
| ./anomaly_detection/inference/extract_feat_test.py                                                                                         | https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.NearestNeighbors.html |                                                                                      |
| ./anomaly_detection/zero_shot_AD/zero_shot_AD.py                                                                                           | https://github.com/google-gemini/generative-ai-python                                     |                                                                                      |
| ./anomaly_detection/zero_shot_AD/arrange_img.py                                                                                            | Github Co-pilot : https://github.com/features/copilot                                    |                                                                                      |



### References for recovery pipeline

| scripts in the repo                          | URL/ Link to source repo (Git repo we referred to)                                   | source script (scripts in the git repo we referred to)                                                                                                                                                                                                                                      |
| -------------------------------------------- | ------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ./recovery/arm.py                            | https://github.com/HBRS-SDP/ws23-door-opening/tree/main                              | https://github.com/HBRS-SDP/ws23-door-opening/blob/c494867b53a7507a0fb48378265129844ab9d82a/src/door_opening/src/scripts/open_door.py<br />https://github.com/HBRS-SDP/ws23-door-opening/blob/c494867b53a7507a0fb48378265129844ab9d82a/src/door_opening/src/scripts/force_transformation.py |
| ./recovery/camera.py                         | https://github.com/Kinovarobotics/ros_kortex_vision                                  |                                                                                                                                                                                                                                                                                             |
| ./recovery/DINO_model.py                     | https://github.com/lightly-ai/lightly/blob/master/examples/pytorch_lightning/dino.py |                                                                                                                                                                                                                                                                                             |
| ./recovery/joints.py<br />./recovery/pose.py | https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_drive          |                                                                                                                                                                                                                                                                                             |
| ./recovery/gmm.py                            | https://scikit-learn.org/0.15/modules/generated/sklearn.mixture.GMM.html             |                                                                                                                                                                                                                                                                                             |
