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

all_rewards = inspect.getmembers(sys.modules["humenv.rewards"], inspect.isclass)
for reward_class_name, reward_cls in all_rewards:
    if not inspect.isabstract(reward_cls):
        print(reward_class_name)

print(f"Number of predefined tasks: {len(humenv.ALL_TASKS)}")
print("Examples:")
print(humenv.LOCOMOTION_TASKS[:10])
print(humenv.ROTATION_TASKS[:10])
print("...")

reward_fn = make_from_name("jump-2")
print(reward_fn)
print(humenv_rewards.JumpReward(jump_height=1.6))

frames = [env.render()]
for i in range(60):
    env.step(env.action_space.sample())
    frames.append(env.render())


# Get frame size from first frame
frame_h, frame_w, channels = frames[0].shape

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter("output.mp4", fourcc, 30, (frame_w, frame_h))

for frame in frames:
    # Ensure frame is uint8
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Convert RGB to BGR for OpenCV
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    out.write(frame_bgr)

out.release()
print("Video saved to output.mp4")

# As in gymnasium, `close` should be called when the environment is not needed anymore
env.close()
if mp_info is not None:
    # In this case, mp_info will contain a manager multiprocessing.Manager()
    # (mp_info["manager"]) and a shared motion buffer (mp_info["motion_buffer"]).
    # Note that since mp_info["motion_buffer"] is a shared object,
    # any change to this class will propagate to the processes.
    # You should call mp_info["manager"].shutdown() before exiting the application.
    mp_info["manager"].shutdown()
