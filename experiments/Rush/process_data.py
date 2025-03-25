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


cluster = 2
if cluster == 1:
    distance_threshold = 2.58
else:
    distance_threshold = 9
dt = 0.5
obs_len = 6   # Length of observed trajectory
pred_len = 9  # Length of predicted trajectory
traj_len = obs_len + pred_len

data_path = '/home/xiangmin/Documents/GitHub/Pedestrian-training'

df = pd.read_csv(data_path +'/Data/clustered.csv')

Cluster_IDs = []
Cluster_Trajectories_num = {}
df_cluster = df[df['Cluster']==cluster]
ID = df_cluster.iloc[0]['ID']
Cluster_Trajectories = defaultdict(list)
trajectory = 0
traj_num = 1
lengths = []
length = 0
for j in range(df_cluster.shape[0]):
    if ID not in Cluster_IDs:
        Cluster_IDs.append(ID)
    if df_cluster.iloc[j]['ID'] != ID:
        Cluster_Trajectories_num[ID] = traj_num
        if length:
            lengths.append(length)
        ID = df_cluster.iloc[j]['ID']
        Cluster_Trajectories[ID] = []
        traj_num = 0
        length = 0
        trajectory = 0
    if df_cluster.iloc[j]['Trajectory'] != trajectory:
        trajectory = df_cluster.iloc[j]['Trajectory']
        traj_num+=1
        Cluster_Trajectories[ID].append(trajectory)
        if length:
            lengths.append(length)
            length = 0
    length += 1
Cluster_Trajectories_num[ID] = traj_num
Cluster_Trajectories[ID].append(trajectory)

# Load all CSV files
df_test = pd.read_csv(data_path + '/Data/Experiment 1.csv')   # test
df_val  = pd.read_csv(data_path + '/Data/Experiment 2.csv')   # val
df_train= pd.read_csv(data_path + '/Data/Experiment 3.csv')   # train

# ========== STANDARDIZATION ==========
standardization_by_cluster = {
    1: {
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': -0.00, 'std': 1.37},
                'y': {'mean': -3.25, 'std': 2.66}
            },
            'velocity': {
                'x': {'mean': 0.51, 'std': 0.73},
                'y': {'mean': 0.51, 'std': 0.73}
            },
            'acceleration': {
                'x': {'mean': -0.01, 'std': 0.81},
                'y': {'mean': -0.01, 'std': 0.81}
            }
        }
    },
    2: {
        'PEDESTRIAN': {
            'position': {
                'x': {'mean': -0.10, 'std': 1.54},
                'y': {'mean': -3.78, 'std': 2.82}
            },
            'velocity': {
                'x': {'mean': 0.66, 'std': 0.81},
                'y': {'mean': 0.66, 'std': 0.81}
            },
            'acceleration': {
                'x': {'mean': -0.02, 'std': 0.95},
                'y': {'mean': -0.02, 'std': 0.95}
            }
        }
    }
}

# ========== ENVIRONMENT SETUP ==========
def setup_env():
    env = Environment(node_type_list=["PEDESTRIAN"], standardization=standardization_by_cluster[cluster])
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
            if traj_id not in Cluster_Trajectories[agent_id]:
                continue
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
subfolder = "patient" if cluster == 1 else "impatient"
save_dir = os.path.join(save_base, subfolder)
os.makedirs(save_dir, exist_ok=True)

dill.dump(env_train, open(os.path.join(save_dir, "train.pkl"), "wb"))
dill.dump(env_val,   open(os.path.join(save_dir, "val.pkl"), "wb"))
dill.dump(env_test,  open(os.path.join(save_dir, "test.pkl"), "wb"))

print("✅ All environment files saved!")


