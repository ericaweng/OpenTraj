import sys
import os
import math
import numpy as np
import pandas as pd
import sympy as sp 
import pickle
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from toolkit.loaders.loader_eth import load_eth
from toolkit.core.trajdataset import TrajDataset
from toolkit.benchmarking.load_all_datasets import get_datasets, get_trajlets
from toolkit.core.trajlet import split_trajectories
from toolkit.benchmarking.utils.histogram_sampler import normalize_samples_with_histogram
from copy import deepcopy


def mean_traj_speed(all_frames):
    pass

def mean_traj_length(all_frames):
    pass

def mean_var_angular_velocity(all_frames):
    pass

def mean_var_angle(all_frames):
    pass
    
def mean_start_goal_distance(all_trajs):
    """assumes trajectories are sorted in ascending timestep"""

    all_dts = []
    all_euclidean_dists = []
    all_mean_vels = []
    all_headings = []
    all_angular_vels = []
    all_total_dists = []
    all_dist_diffs = []
    all_dist_quots = []
    for group_i, (name_of_the_group, group) in enumerate(all_trajs):
        if group.shape[0] < 2:
            continue
        
        # td between start and goal of each agent
        dt = group.iloc[-1]["timestamp"] - group.iloc[0]["timestamp"] 
        all_dts.append(dt)

        # tds between each timestep; used for velocity calculation
        time_diff = group["timestamp"].diff().to_numpy()[1:]

        # agent velocity
        if np.any(np.isnan(group["vel_x"].to_numpy())):
            if group["vel_x"].isnull().sum() == group["vel_x"].shape[0]:
                print("{} vs. {}\n\n".format(group["vel_x"].isnull().sum(), group["vel_x"].shape[0]))
            # fill-in empty vels
            vx = group["pos_x"].diff()[1:] / time_diff
            vy = group["pos_y"].diff()[1:] / time_diff
            nan_idxs = np.isnan(group["vel_x"].to_numpy())
        else:
            vx = group["vel_x"]
            vy = group["vel_y"]
        vel = np.linalg.norm((vx, vy), axis=0)
        mean_vel = np.mean(vel)
        all_mean_vels.append(mean_vel)

        # std_dev of heading (angle): how much each agent deviates from its mean heading
        heading = np.arctan2(vy, vx)
        all_headings.append(np.std(heading))

        # std_dev of angular velocity (angle finite differences)
        assert heading.shape[0] == time_diff.shape[0] or heading.shape[0] - 1 == time_diff.shape[0]
        angular_diffs = heading.diff()[1:] / time_diff[:heading.shape[0] - 1]
        all_angular_vels.append(np.std(angular_diffs))

        # euclidean distance from start to goal
        dx = group.iloc[-1]["pos_x"] - group.iloc[0]["pos_x"] 
        dy = group.iloc[-1]["pos_y"] - group.iloc[0]["pos_y"] 
        euclidean_dist = np.linalg.norm((dx, dy))
        all_euclidean_dists.append(euclidean_dist)

        # total distance from start to goal
        pos_x_diffs = group["pos_x"].diff().to_numpy()[1:]
        pos_y_diffs = group["pos_y"].diff().to_numpy()[1:]
        dist_diffs = np.linalg.norm((pos_x_diffs, pos_y_diffs), axis=0)
        total_dist = np.sum(dist_diffs)
        all_total_dists.append(total_dist)

        # total dist - euclidean dist
        all_dist_diffs.append(total_dist - euclidean_dist)

        # total dist / euclidean dist
        if euclidean_dist != 0:
            all_dist_quots.append(total_dist / euclidean_dist)

    all_data = [all_dts, all_mean_vels, all_headings, all_angular_vels, 
                all_euclidean_dists, all_total_dists, all_dist_diffs, all_dist_quots]
    means = map(lambda x: np.mean(x), all_data)
    std_devs = map(lambda x: np.std(x), all_data)
    return means, std_devs, all_data


def run(datasets, output_dir):
    rows = []
    colnames = ["num_traj", "area", "mean_area", "scene_height", "scene_width", "total traj_time", "traj_time", "mean_vel", 
                "heading", "angular_vel", "start-goal dist", "sg total_dist", "dist_diffs", "dist_quotients"]    
    print("Name\t{}".format("\t".join(colnames)))

    print_string = ""

    with open("../../../rows.tsv", 'w') as f:
        f.write(print_string)
    for ds_name in datasets.keys():
        dataset = datasets[ds_name]

        all_frames = dataset.get_frames()
        all_trajs = dataset.get_trajectories()

        # area calculation
        scenes_maxX = dataset.data.groupby(['scene_id'])['pos_x'].max() 
        scenes_minX = dataset.data.groupby(['scene_id'])['pos_x'].min()
        scenes_maxY = dataset.data.groupby(['scene_id'])['pos_y'].max()
        scenes_minY = dataset.data.groupby(['scene_id'])['pos_y'].min()

        area = pd.DataFrame(data=[], columns=['area', 'height', 'width'])
        for idx in scenes_maxX.index:
            x_range = scenes_maxX.loc[idx]-scenes_minX.loc[idx]
            y_range = scenes_maxY.loc[idx]-scenes_minY.loc[idx]
            area.loc[idx, 'area'] = x_range * y_range
            area.loc[idx, 'height'] = y_range
            area.loc[idx, 'width'] = x_range

        string_ntraj = str(len(all_trajs))
        string_area_data = " / ".join(['"{}": {:0.0f}'.format(idx, area.loc[idx, "area"]) 
            for idx in scenes_maxX.index])
        string_area_mean = "{:0.0f} / {:0.0f}".format(area["area"].mean(), area["area"].std())  # if len(scenes_maxX.index) > 1 else "-"
        string_scene_hw = "{:0.0f}\t{:0.0f}".format(area['height'].mean(), area['width'].mean())
        string_metadata = "\t".join([string_ntraj, string_area_data, string_area_mean, string_scene_hw])

        means, std_devs, all_data = mean_start_goal_distance(all_trajs)
        string_total_scene_len = "{:0.0f}".format(np.sum(all_data[0]) / 60)
        string_mean_std_data = "\t".join(["{:0.2f} / {:0.2f}".format(mean, std) for mean, std in zip(means, std_devs)])
        all_string_data = "\t".join([string_metadata, string_total_scene_len, string_mean_std_data])
        print_string += "{}\t{}\n".format(ds_name, all_string_data)
        rows.append(all_data)

    with open("../../../rows.tsv", 'w') as f:
        f.write(print_string)
    print(datasets.keys())
    # pickle.dump(rows, open(os.path.join(output_dir, "rows.pkl"), 'wb'))

if __name__ == "__main__":
    opentraj_root = "../../../"#sys.argv[1]
    output_dir = "../../output_human"#sys.argv[2]
    all_dataset_names = [
    # 'ETH-Univ',
    # 'ETH-Hotel',
    # 'UCY-Zara',
    # 'UCY-Univ',
    'PETS-S2l1',
    'GC',
    # 'SDD-coupa',
    # 'SDD-bookstore',
    # 'SDD-deathCircle',
    # 'SDD-gates',
    # 'SDD-hyang',
    # 'SDD-little',
    # 'SDD-nexus',
    # 'SDD-quad',
    'BN-1d-w180',
    'BN-2d-w160',
    'Edinburgh',
    'TownCenter',
    'WildTrack',
    'KITTI',
    # 'LCas-Minerva','InD-1','InD-2'
    ]

    datasets = get_datasets(opentraj_root, all_dataset_names)
    run(datasets, output_dir)
