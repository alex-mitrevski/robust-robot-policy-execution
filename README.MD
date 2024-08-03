### Master Thesis: Robust robot task execution using monitoring and recovery in anomalous conditions

The project consists of three parts

- [Task execution](./task_reach_door/)
- [Anomaly detection](./anomaly_detection)
- [Recovery ](./recovery)

Every module has it's own dependencies to run , which can be found within the directories. Use python virtual environment to install the requirements, conda environments had some issues. 

- Running pybullet requires GUI access, if you are running on server use ssh with `-X` or `-Y`
- Training requires GPU, we used one RTX 3090 for training the RL model and two RTX 3090 for training our Anomaly detection model. Make sure compatible NVIDIA drivers are installed to install` torch` with GPU support. 
- GPU not required to run testing on the robots.

To run only [task_reach_door](./task_reach_door) and [anomaly_detection](./anomaly_detection), use the [requirement.txt](./requirements.txt)
