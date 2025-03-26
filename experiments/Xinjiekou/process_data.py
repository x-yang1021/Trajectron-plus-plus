import os
import dill
import glob
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
sys.path.append("../../trajectron")
from environment import Environment, Scene, Node
from environment import derivative_of
from utils import maybe_makedirs

# === Configuration ===
data_path = '/home/xiangmin/PycharmProjects/Xinjiekou'
data_root = os.path.join(data_path, 'Data/Xinjiekou')
map_path = os.path.join(data_path, 'benchmark/Trajectron/map.pkl')
output_root = os.path.join(data_path, 'benchmark/Trajectron')

step_length = 4
dt = 0.4
obs_len = 6   # Length of observed trajectory
pred_len = 9  # Length of predicted trajectory
traj_len = obs_len + pred_len

origin = [-455,52322]

standardization_by_time = {
    "AM": {
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': 11.27, 'std': 4.64},
                'y': {'mean': 67.19, 'std': 45.52}
            },
            'velocity': {
                'x': {'mean': -0.0062, 'std': 0.1164},
                'y': {'mean': -0.0261, 'std': 0.8565}
            },
            'acceleration': {
                'x': {'mean': -0.0004, 'std': 0.1736},
                'y': {'mean': -0.0025, 'std': 1.4307}
            }
        }
    },
    "NOON": {
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': 15.07, 'std': 4.97},
                'y': {'mean': 77.24, 'std': 47.63}
            },
            'velocity': {
                'x': {'mean': 0.0044, 'std': 0.1276},
                'y': {'mean': 0.0278, 'std': 1.2361}
            },
            'acceleration': {
                'x': {'mean': 0.0006, 'std': 0.1934},
                'y': {'mean': -0.0071, 'std': 2.1033}
            }
        }
    },
    "PM": {
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': 13.21, 'std': 4.32},
                'y': {'mean': 76.42, 'std': 44.60}
            },
            'velocity': {
                'x': {'mean': 0.0005, 'std': 0.0989},
                'y': {'mean': 0.0237, 'std': 0.9244}
            },
            'acceleration': {
                'x': {'mean': -0.0002, 'std': 0.1296},
                'y': {'mean': -0.0033, 'std': 1.5227}
            }
        }
    }
}

for time_period in ["AM", "NOON", "PM"]:
    path = os.path.join(data_root, 'North', time_period)
    all_files = glob.glob(path + "/*.txt")

    trajectories = []
    for file in all_files:
        df = pd.read_csv(file, sep="\t", header=None)
        if df.shape[0] < traj_len * step_length:
            continue
        x = -(df.iloc[:, 2] - origin[0])  # origin x
        y = df.iloc[:, 4] - origin[1]       # origin y
        df_proc = pd.DataFrame({"x": x, "y": y})

        vx = derivative_of(df_proc["x"].values, dt)
        vy = derivative_of(df_proc["y"].values, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        data = pd.DataFrame({
            ("position", "x" ): df_proc["x"].values,
            ("position", "y" ): df_proc["y"].values,
            ("velocity", "x" ): vx,
            ("velocity", "y" ): vy,
            ("acceleration", "x"): ax,
            ("acceleration", "y"): ay,
        })
        data.columns = pd.MultiIndex.from_tuples(data.columns)
        trajectories.append(data)

    # === Train/Test/Val Split ===
    train_traj, temp_traj = train_test_split(trajectories, test_size=0.2, random_state=1)
    val_traj, test_traj = train_test_split(temp_traj, test_size=0.5, random_state=1)


    def make_env(trajs, time_period):
        env = Environment(["PEDESTRIAN"], standardization=standardization_by_time[time_period])
        env.attention_radius = {("PEDESTRIAN", "PEDESTRIAN"): 4.0}
        with open(map_path, "rb") as f:
            env.map = dill.load(f)
        data_columns = pd.MultiIndex.from_product([["position", "velocity", "acceleration"], ["x", "y"]])

        scenes = []
        for i, traj in enumerate(trajs):
            if len(traj) < obs_len + pred_len:
                continue
            scene = Scene(timesteps=len(traj), dt=dt, name=f"scene_{i}")
            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=str(i), data=traj[data_columns], first_timestep=0)
            scene.nodes.append(node)
            scenes.append(scene)
        print(len(scenes), time_period)
        env.scenes = scenes
        return env

    env_train = make_env(train_traj, time_period)
    env_val   = make_env(val_traj, time_period)
    env_test  = make_env(test_traj, time_period)

    # === Save ===
    save_path = os.path.join(output_root, time_period)
    maybe_makedirs(save_path)

    with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
        dill.dump(env_train, f, protocol=dill.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'val.pkl'), 'wb') as f:
        dill.dump(env_val, f, protocol=dill.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'test.pkl'), 'wb') as f:
        dill.dump(env_test, f, protocol=dill.HIGHEST_PROTOCOL)

    print(f"✅ Processed and saved: {time_period}")