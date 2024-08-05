### Task execution : Reach a door handle

We train a Reinforcement Learning model to train a Kinova arm to reach a door handle in simulation. We use PyBullet physics simulation engine for Reinforcement Learning training.

#### Dependencies

To run only the RL simulation, install dependencies in `/task_reach_door/requirements.txt`

```
cd my_project  # path to install python virtual environment
python3.8 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### Train the model

```
source venv/bin/activate
cd ./task_reach_door
python3 vmp_trainer.py
```

Adjust the paths and parameters in the following config files if required

`./task_reach_door/config/config_ppo.yaml`

`./task_reach_door/config/config_simulation.yaml`

#### Evaluate the model

Based on whether the model is trained with or without Domain randomization place the trained model in either `./task_reach_door/saved_models/with_DR` or `./task_reach_door/saved_models/without_DR`, and start the evaluation using the script,

```
source venv/bin/activate
cd ./tas_reach_door
python3 evaluate.py
```

#### References:

1. The URDF for Kinova arm is from the official Kinova kortex repository
   ```
   * https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/arms/gen3/7dof/urdf`
   * https://github.com/Kinovarobotics/ros_kortex/tree/noetic-devel/kortex_description/grippers/robotiq_2f_85/urdf
   ```
2. The door and the handle is designed using the online CAD tool and imported to the URDF using the STL format. `https://www.onshape.com/en/`
3. We referred to the following repo to understand the structure of `./task_reach_door/vmp_trainer.py` and `./task_reach_door/evaluate.py`Repo:` https://github.com/NJ-2020-thesis/PyRep/tree/feature/examples/vmp`
4. We referred to the following repo to understand PyBullet and it's application for manipulation tasks.
   `https://github.com/NilsDengler/cluttered-pushing/tree/main`
5. Official PyBullet documentation:  `https://pybullet.org/wordpress/index.php/forum-2/`
6. Official gymanasium documentation: `https://gymnasium.farama.org/`
7. Official Stable-baselines3 documentation: `https://stable-baselines3.readthedocs.io/en/master/`

 Specific usages of above references are given as comments in the code scripts.
