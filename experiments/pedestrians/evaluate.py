import sys
import os
import dill
import json
import argparse
import torch
import numpy as np
import pandas as pd

sys.path.append("../../trajectron")
from tqdm import tqdm
from model.model_registrar import ModelRegistrar
from model.trajectron import Trajectron
import evaluation

# Seed for reproducibility
seed = 0
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, help="model full path")
parser.add_argument("--checkpoint", type=int, help="model checkpoint to evaluate")
parser.add_argument("--data", type=str, help="full path to data file")
parser.add_argument("--output_path", type=str, help="path to output csv file")
parser.add_argument("--output_tag", type=str, help="name tag for output file")
parser.add_argument("--node_type", type=str, help="node type to evaluate")
args = parser.parse_args()

def load_model(model_dir, env, ts=100):
    model_registrar = ModelRegistrar(model_dir, 'cpu')
    model_registrar.load_models(ts)
    with open(os.path.join(model_dir, 'config.json'), 'r') as config_json:
        hyperparams = json.load(config_json)
    trajectron = Trajectron(model_registrar, hyperparams, None, 'cpu')
    trajectron.set_environment(env)
    trajectron.set_annealing_params()
    return trajectron, hyperparams

def save_results(values, metric, eval_type, output_path, output_tag):
    values = np.asarray(values).astype(float)
    df = pd.DataFrame({
        'value': values,
        'metric': [metric] * len(values),
        'type': [eval_type] * len(values)
    })
    df.to_csv(os.path.join(output_path, f"{output_tag}_{metric}_{eval_type}.csv"), index=False)

if __name__ == "__main__":
    with open(args.data, 'rb') as f:
        env = dill.load(f, encoding='latin1')

    eval_stg, hyperparams = load_model(args.model, env, ts=args.checkpoint)

    if 'override_attention_radius' in hyperparams:
        for attention_radius_override in hyperparams['override_attention_radius']:
            node_type1, node_type2, attention_radius = attention_radius_override.split(' ')
            env.attention_radius[(node_type1, node_type2)] = float(attention_radius)

    scenes = env.scenes
    print("-- Preparing Node Graph")
    for scene in tqdm(scenes):
        scene.calculate_scene_graph(env.attention_radius,
                                    hyperparams['edge_addition_filter'],
                                    hyperparams['edge_removal_filter'])

    ph = hyperparams['prediction_horizon']
    max_hl = hyperparams['maximum_history_length']

    with torch.no_grad():
        for mode in ['full']:
            print(f"-- Evaluating Mode: {mode}")
            eval_ade_batch_errors, eval_fde_batch_errors, eval_kde_nll = [], [], []

            for i, scene in enumerate(scenes):
                print(f"---- Evaluating Scene {i + 1}/{len(scenes)}")
                timestep_range = [0] if mode == 'ml' else range(0, scene.timesteps, 10)

                for t in timestep_range:
                    timesteps = np.arange(t, t + 10) if mode != 'ml' else np.arange(scene.timesteps)

                    predictions = eval_stg.predict(scene,
                                                   timesteps,
                                                   ph,
                                                   num_samples=(1 if mode == 'ml' else 2000 if mode in ['z_mode', 'full'] else 20),
                                                   min_history_timesteps=7,
                                                   min_future_timesteps=12,
                                                   z_mode=(mode == 'z_mode'),
                                                   gmm_mode=(mode == 'ml'),
                                                   full_dist=(mode == 'ml'))

                    if not predictions:
                        continue

                    batch_error_dict = evaluation.compute_batch_statistics(predictions,
                                                                           scene.dt,
                                                                           max_hl=max_hl,
                                                                           ph=ph,
                                                                           node_type_enum=env.NodeType,
                                                                           map=None,
                                                                           prune_ph_to_future=True,
                                                                           kde=(mode in ['z_mode', 'full']),
                                                                           best_of=(mode == 'best_of'))

                    eval_ade_batch_errors.extend(batch_error_dict[args.node_type]['ade'])
                    eval_fde_batch_errors.extend(batch_error_dict[args.node_type]['fde'])
                    if 'kde' in batch_error_dict[args.node_type]:
                        eval_kde_nll.extend(batch_error_dict[args.node_type]['kde'])

            save_results(eval_ade_batch_errors, 'ade', mode, args.output_path, args.output_tag)
            save_results(eval_fde_batch_errors, 'fde', mode, args.output_path, args.output_tag)
            if eval_kde_nll:
                save_results(eval_kde_nll, 'kde', mode, args.output_path, args.output_tag)
