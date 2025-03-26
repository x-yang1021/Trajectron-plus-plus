import os
import dill
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from collections import defaultdict

from experiments.pedestrians.process_data import standardization

sys.path.append("../../trajectron")
from environment import Environment, Scene, Node
from environment import derivative_of
from utils import maybe_makedirs



distance_threshold = 2.58
dt = 0.5
obs_len = 6   # Length of observed trajectory
pred_len = 9  # Length of predicted trajectory
traj_len = obs_len + pred_len

data_path = '/home/xiangmin/Documents/GitHub/Pedestrian-training'

# Load all CSV files
df_test = pd.read_csv(data_path + '/Data/Experiment 1.csv')   # test
df_val  = pd.read_csv(data_path + '/Data/Experiment 2.csv')   # val
df_train= pd.read_csv(data_path + '/Data/Experiment 3.csv')   # train

# ========== STANDARDIZATION ==========
standardization = {
    'PEDESTRIAN': {
        'position': {
            'x': {'mean': 0.04, 'std': 0.54},
            'y': {'mean': -1.40, 'std': 0.51}
        },
        'velocity': {
            'x': {'mean': 0.28, 'std': 0.43},
            'y': {'mean': 0.28, 'std': 0.43}
        },
        'acceleration': {
            'x': {'mean': -0.03, 'std': 0.47},
            'y': {'mean': -0.03, 'std': 0.47}
        }
    }
}

# ========== ENVIRONMENT SETUP ==========
def setup_env():
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization)
    env.attention_radius = {("PEDESTRIAN", "PEDESTRIAN"): 4.0}
    return env

env_train = setup_env()
env_val   = setup_env()
env_test  = setup_env()
data_columns = pd.MultiIndex.from_product([['position', 'velocity', 'acceleration'], ['x', 'y']])

# ========== PROCESS FUNCTION ==========
def process_scene(df, scene_id):
    df = df.dropna(subset=["ID", "Time", "Positionx", "Positiony", "Distance"]).copy()
    df["ID"] = df["ID"].astype(int)
    df = df[df["Distance"] <= distance_threshold]
    df = df.rename(columns={"Time": "timestep", "Positionx": "x", "Positiony": "y"})
    df["frame_id"] = (df["timestep"] / dt).round().astype(int)
    df["track_id"] = df["ID"].astype(str) + "_" + df["Trajectory"].astype(str)
    df["node_type"] = "PEDESTRIAN"

    scene = Scene(timesteps=df["frame_id"].max() + 1, dt=dt, name=f"scene_{scene_id}")

    for agent_id, agent_group in df.groupby("ID"):
        for traj_id, traj_group in agent_group.groupby("Trajectory"):
            traj_group = traj_group.sort_values("frame_id")
            if len(traj_group) < traj_len:
                continue

            track_ID = traj_group["track_id"].iloc[0]
            x = traj_group["x"].values
            y = traj_group["y"].values
            vx = derivative_of(x, dt)
            vy = derivative_of(y, dt)
            ax = derivative_of(vx, dt)
            ay = derivative_of(vy, dt)

            data = pd.DataFrame({
                ("position", "x"): x,
                ("position", "y"): y,
                ("velocity", "x"): vx,
                ("velocity", "y"): vy,
                ("acceleration", "x"): ax,
                ("acceleration", "y"): ay,
            }, columns=data_columns)

            node = Node(node_type=env.NodeType.PEDESTRIAN, node_id=str(track_ID), data=data,
                        first_timestep=traj_group["frame_id"].iloc[0])
            scene.nodes.append(node)

    print(scene)
    return scene if len(scene.nodes) > 0 else None

# ========== BUILD SCENES ==========
scene_id = 1

for df, env, label in zip([df_train, df_val, df_test], [env_train, env_val, env_test], ["Train", "Val", "Test"]):
    scene = process_scene(df, scene_id)
    if scene:
        env.scenes = [scene]
        print(f"✅ Added {label} scene: {scene.name} with {len(scene.nodes)} nodes")
        scene_id += 1
    else:
        print(f"⚠️ No valid {label} scene created")

# ========== SAVE TO FILE ==========
save_base = data_path + '/benchmark/Trajectron/'
subfolder = 'combined'
save_dir = os.path.join(save_base, subfolder)
os.makedirs(save_dir, exist_ok=True)

dill.dump(env_train, open(os.path.join(save_dir, "train.pkl"), "wb"))
dill.dump(env_val,   open(os.path.join(save_dir, "val.pkl"), "wb"))
dill.dump(env_test,  open(os.path.join(save_dir, "test.pkl"), "wb"))

print("✅ All environment files saved!")


