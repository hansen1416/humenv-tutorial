import os
from pathlib import Path

# import mediapy as media
import sys
import inspect
import numpy as np
import json
from gymnasium.wrappers import FlattenObservation, TransformObservation
import cv2

# humenv
import humenv
from humenv import make_humenv
from humenv.env import make_from_name
from humenv import rewards as humenv_rewards


os.environ["MUJOCO_GL"] = os.environ.get("MUJOCO_GL", "egl")
os.environ["PYOPENGL_PLATFORM"] = os.environ["MUJOCO_GL"]


# make a single environment
kwargs = {
    "state_init": "MoCap",
}
motions_base_path = Path("data_preparation/humenv_amass")


from humenv.bench import TrackingEvaluation


class RandomAgent:
    def __init__(self, env):
        self.env = env

    def act(self, obs, z):
        return self.env.action_space.sample()

    def reward_inference(self, task):
        return None

    def goal_inference(self, goal_pose):
        return None

    def tracking_inference(self, next_obs):
        return None


num_envs = 1
# env is needed here only for RandomAgent
env, mp_info = make_humenv(num_envs=num_envs)
random_agent = RandomAgent(env)

motions_base_path = Path("data_preparation/humenv_amass")

if not motions_base_path.exists():
    assert (
        True
    ), "[WARNING] You should generate the data before running these instructions"


# Let's use only 5 motions for faster loading:
motions = [str(x.name) for x in motions_base_path.glob("*.hdf5")][0:5]

tracking_eval = TrackingEvaluation(
    motions=motions,
    num_envs=num_envs,
    motion_base_path=motions_base_path,
)
metrics = tracking_eval.run(agent=random_agent)
print()
print("Results")
for k, v in metrics.items():
    print(f"task='{k}'")
    for k2, v2 in v.items():
        print(f"  {k2}: {v2}")

# As in gymnasium, `close` should be called when the environment is not needed anymore
env.close()
if mp_info is not None:
    # In this case, mp_info will contain a manager multiprocessing.Manager()
    # (mp_info["manager"]) and a shared motion buffer (mp_info["motion_buffer"]).
    # Note that since mp_info["motion_buffer"] is a shared object,
    # any change to this class will propagate to the processes.
    # You should call mp_info["manager"].shutdown() before exiting the application.
    mp_info["manager"].shutdown()
