#### References for recovery pipeline

We referred to multiple sources while intergating our task execution with the anomaly detection and recovey.

Below, we refer to the scripts under ` /recovery` directory and mention all relevant sources and url we referred to while implemeting in code

`/recovery/arm.py`

To implement `arm.py` we used the following SDP project to refer to the appropriate ROS service, topics  linked to the Kinova arm

1. SDP project github : [Door-opening  SDP project](https://github.com/HBRS-SDP/ws23-door-opening/blob/main/src/door_opening/src/scripts/open_door.py)
2. SDP project github: [Door-opening SDP project](https://github.com/HBRS-SDP/ws23-door-opening/blob/main/src/door_opening/src/scripts/force_transformation.py)

`/recovery/DINO_model.py`

DINO architecture implemented is adapted from [LightlySSL](https://docs.lightly.ai/self-supervised-learning/examples/dino.html)

/recovery/gmm.py

We used scikit learn library's GMM implementation for our use case

source: [GMM, Scikit learn](https://scikit-learn.org/0.15/modules/generated/sklearn.mixture.GMM.html)
