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

data_path = '/home/xiangmin/PycharmProjects/Xinjiekou'
data_root = os.path.join(data_path, 'Data/Xinjiekou')
map_path = os.path.join(data_path, 'benchmark/Trajectron/map.pkl')
output_root = os.path.join(data_path, 'benchmark/Trajectron')

step_length = 4
dt = 0.4
obs_len = 8   # Length of observed trajectory
pred_len = 12  # Length of predicted trajectory
traj_len = obs_len + pred_len
Heading = 1

origin = [-474,52322]

exit = [51,86]

standardization_by_time = {}


for time_period in ["AM", "NOON", "PM"]:
    path = os.path.join(data_root, 'North', time_period)
    all_files = glob.glob(path + "/*.txt")

    trajectories = []
    all_positions = []
    all_velocities = []
    all_accelerations = []

    if time_period == "AM":
        vertical_threshold = 5
        horizontal_threshold = 5
    elif time_period == "NOON":
        vertical_threshold = 20
        horizontal_threshold = 5
    else:
        vertical_threshold = 10
        horizontal_threshold = 10

    for file in all_files:
        df = pd.read_csv(file, sep="\t", header=None)
        if df.shape[0] < traj_len * step_length:
            continue
        x = df.iloc[:, 2] - origin[0]
        y = df.iloc[:, 4] - origin[1]
        heading = int(df.iloc[-1, 4] - df.iloc[0, 4] > 0)
        if Heading != heading:
            continue
        if x.max() - x.min() > horizontal_threshold and y.max() - y.min() < vertical_threshold:
            continue
        df_proc = pd.DataFrame({"x": x, "y": y})

        vx = derivative_of(df_proc["x"].values, dt)
        vy = derivative_of(df_proc["y"].values, dt)
        ax = derivative_of(vx, dt)
        ay = derivative_of(vy, dt)

        all_positions.append(np.stack([x.values, y.values], axis=1))
        all_velocities.append(np.stack([vx, vy], axis=1))
        all_accelerations.append(np.stack([ax, ay], axis=1))

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

    all_positions = np.concatenate(all_positions, axis=0)
    all_velocities = np.concatenate(all_velocities, axis=0)
    all_accelerations = np.concatenate(all_accelerations, axis=0)

    stats = {
        'position': {
            'x': {'mean': np.mean(all_positions[:, 0]), 'std': np.std(all_positions[:, 0])},
            'y': {'mean': np.mean(all_positions[:, 1]), 'std': np.std(all_positions[:, 1])}
        },
        'velocity': {
            'x': {'mean': np.mean(all_velocities[:, 0]), 'std': np.std(all_velocities[:, 0])},
            'y': {'mean': np.mean(all_velocities[:, 1]), 'std': np.std(all_velocities[:, 1])}
        },
        'acceleration': {
            'x': {'mean': np.mean(all_accelerations[:, 0]), 'std': np.std(all_accelerations[:, 0])},
            'y': {'mean': np.mean(all_accelerations[:, 1]), 'std': np.std(all_accelerations[:, 1])}
        }
    }

    standardization_by_time[time_period] = {'PEDESTRIAN': stats}
    print(f"\n✅ {time_period} Standardization Stats:")
    for k in stats:
        for axis in ['x', 'y']:
            print(f"{k}.{axis}: mean={stats[k][axis]['mean']:.4f}, std={stats[k][axis]['std']:.4f}")


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


    save_path = os.path.join(output_root, time_period)
    maybe_makedirs(save_path)

    with open(os.path.join(save_path, 'train.pkl'), 'wb') as f:
        dill.dump(env_train, f, protocol=dill.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'val.pkl'), 'wb') as f:
        dill.dump(env_val, f, protocol=dill.HIGHEST_PROTOCOL)
    with open(os.path.join(save_path, 'test.pkl'), 'wb') as f:
        dill.dump(env_test, f, protocol=dill.HIGHEST_PROTOCOL)

    print(f"Processed and saved: {time_period}")