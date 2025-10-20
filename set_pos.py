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
env, mp_info = make_humenv(max_episode_steps=1000)
print(f"time step: {env.spec.max_episode_steps}")
print(f"mp_info: {mp_info}")

_, info = env.reset()
for k, v in info.items():
    print(f"{k}: {v.shape} -> {v[:11]}")
# Floating base:
# qpos: 7 (3 pos + 4 rot)
# qvel: 6 (3 linear vel + 3 angular vel)
# Plus 69 joints (for instance):
# qpos: 69 → 7 + 69 = 76
# qvel: 69 → 6 + 69 = 75
print((info["qpos"] == env.unwrapped.data.qpos).all())
print((info["qvel"] == env.unwrapped.data.qvel).all())


print("-" * 10)
_, _, _, _, info = env.step(env.action_space.sample())
for k, v in info.items():
    print(f"{k}: {v.shape} -> {v[:11]}")

print(f"dt={env.unwrapped.model.opt.timestep}")
print(f"action_repeat={env.unwrapped.action_repeat}")
print(
    f"control frequency: {env.unwrapped.model.opt.timestep * env.unwrapped.action_repeat}"
)

frame_0 = env.render()
new_qpos = np.array(
    [
        0.13769039,
        -0.20029453,
        0.42305034,
        0.21707786,
        0.94573617,
        0.23868944,
        0.03856998,
        -1.05566834,
        -0.12680767,
        0.11718296,
        1.89464102,
        -0.01371153,
        -0.07981451,
        -0.70497424,
        -0.0478,
        -0.05700732,
        -0.05363342,
        -0.0657329,
        0.08163511,
        -1.06263979,
        0.09788937,
        -0.22008936,
        1.85898192,
        0.08773695,
        0.06200327,
        -0.3802791,
        0.07829525,
        0.06707749,
        0.14137152,
        0.08834448,
        -0.07649805,
        0.78328658,
        0.12580912,
        -0.01076061,
        -0.35937259,
        -0.13176489,
        0.07497022,
        -0.2331914,
        -0.11682692,
        0.04782308,
        -0.13571422,
        0.22827948,
        -0.23456622,
        -0.12406075,
        -0.04466465,
        0.2311667,
        -0.12232673,
        -0.25614032,
        -0.36237662,
        0.11197906,
        -0.08259534,
        -0.634934,
        -0.30822742,
        -0.93798716,
        0.08848668,
        0.4083417,
        -0.30910404,
        0.40950143,
        0.30815359,
        0.03266103,
        1.03959336,
        -0.19865537,
        0.25149713,
        0.3277561,
        0.16943092,
        0.69125975,
        0.21721349,
        -0.30871948,
        0.88890484,
        -0.08884043,
        0.38474549,
        0.30884107,
        -0.40933304,
        0.30889523,
        -0.29562966,
        -0.6271498,
    ]
)

env.unwrapped.set_physics(
    qpos=new_qpos, qvel=np.random.rand(75), ctrl=np.zeros(69)
)  # qvel and ctrl are optionals
# ctrl correspond to the action

#  we can see that we moved the environment in a new state
combined = np.concatenate([env.render(), frame_0], axis=1)
cv2.imshow("Combined Frame", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()

# As in gymnasium, `close` should be called when the environment is not needed anymore
env.close()
if mp_info is not None:
    # In this case, mp_info will contain a manager multiprocessing.Manager()
    # (mp_info["manager"]) and a shared motion buffer (mp_info["motion_buffer"]).
    # Note that since mp_info["motion_buffer"] is a shared object,
    # any change to this class will propagate to the processes.
    # You should call mp_info["manager"].shutdown() before exiting the application.
    mp_info["manager"].shutdown()
