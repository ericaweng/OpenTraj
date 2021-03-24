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

all_dataset_names = [
    'ETH-Univ',
    # 'ETH-Hotel',

    # 'UCY-Zara',
    # 'UCY-Zara1',
    # 'UCY-Zara2',

    # 'UCY-Univ',
    # 'UCY-Univ3',

    # 'PETS-S2l1',

    # 'SDD-coupa',
    # 'SDD-bookstore',
    # 'SDD-deathCircle',
    # 'SDD-gates',
    # 'SDD-hyang',
    # 'SDD-little',
    # 'SDD-nexus',
    # 'SDD-quad',

    # 'GC',
]

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
        # mean time an agent is in the scene
        if group.shape[0] < 2:
            continue

        dt = group.iloc[-1]["timestamp"] - group.iloc[0]["timestamp"] 
        all_dts.append(dt)

        # mean / std_dev velocity
        if np.any(np.isnan(group["vel_x"].to_numpy())):
            assert np.sum(np.isnan(group["vel_x"].to_numpy())) == group["vel_x"].to_numpy().shape[0]
            vx = group["pos_x"].diff() / group["timestamp"].diff()
            vy = group["pos_y"].diff() / group["timestamp"].diff()
            print(vx, vy)
        else:
            vx = group["vel_x"]
            vy = group["vel_y"]
        vel = np.linalg.norm((vx, vy), axis=0)
        mean_vel = np.mean(vel)
        all_mean_vels.append(mean_vel)

        # std_dev of heading (angle): how much each agent deviates from its mean heading
        heading = np.arctan2(vy, vx)
        all_headings.append(np.std(heading))

        # std_dev of angular velocity
        # angular velocity: change in angle from this step to the next (finite difference)
        time_diff = group["timestamp"][1:].to_numpy() - group["timestamp"][:-1].to_numpy()
        angular_diffs = (heading[1:].to_numpy() - heading[:-1].to_numpy()) / (time_diff)  
        all_angular_vels.append(np.std(angular_diffs))

        # euclidean distance from start to goal
        dx = group.iloc[-1]["pos_x"] - group.iloc[0]["pos_x"] 
        dy = group.iloc[-1]["pos_y"] - group.iloc[0]["pos_y"] 
        euclidean_dist = np.linalg.norm((dx, dy))
        all_euclidean_dists.append(euclidean_dist)

        # total dist from start to goal
        # pos_x_diffs = group["pos_x"][1:].to_numpy() - group["pos_x"][:-1].to_numpy()
        # pos_y_diffs = group["pos_y"][1:].to_numpy() - group["pos_y"][:-1].to_numpy()
        pos_x_diffs = group["pos_x"].diff().to_numpy()[1:]
        pos_y_diffs = group["pos_y"].diff().to_numpy()[1:]
        exit()
        dist_diffs = np.linalg.norm((pos_x_diffs, pos_y_diffs), axis=0)
        total_dist = np.sum(dist_diffs)
        all_total_dists.append(total_dist)

        # total dist - euclidean dist
        all_dist_diffs.append(total_dist - euclidean_dist)

        # total dist / euclidean dist
        all_dist_quots.append(total_dist / euclidean_dist)


    all_data = [all_dts, all_mean_vels, all_headings, all_angular_vels, 
                all_euclidean_dists, all_total_dists, all_dist_diffs, all_dist_quots]
    means = map(lambda x: np.mean(x), all_data)
    std_devs = map(lambda x: np.std(x), all_data)
    return means, std_devs, all_data

def global_density(all_frames,area):
    #calculate global density as numebr of agents in the scene area at time t

    frame_density_samples = []
    new_frames = []
    for frame in all_frames:
        if len(frame)>0:
            oneArea = area.loc[frame['scene_id'].values[0],'area']
            frame_density_samples.append(len(frame) / oneArea)
            

    return frame_density_samples 

def run(datasets, output_dir):
    rows = []
    colnames = ["area", "mean_area", "scene_height", "scene_width", "total traj_time", "traj_time", "mean_vel", 
                "heading", "angular_vel", "start-goal dist", "sg total_dist", "dist_diffs"]    
    print("Name\t{}".format("\t".join(colnames)))
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

        string_area_data = " / ".join(["{}: {:0.0f}".format(idx, area.loc[idx, "area"]) 
            for idx in scenes_maxX.index])
        string_area_mean = "{:0.0f}".format(area.loc[idx, "area"].mean())  # if len(scenes_maxX.index) > 1 else "-"
        string_scene_hw = "{:0.0f}\t{:0.0f}".format(area.loc[idx, 'height'], area.loc[idx, 'width'])
        string_scene_dims = "\t".join([string_area_data, string_area_mean, string_scene_hw])

        means, std_devs, all_data = mean_start_goal_distance(all_trajs)
        total_scene_td = np.sum(all_data[0])
        string_total_scene_len = "{:0.2f}".format(total_scene_td / 60)
        string_mean_std_data = "\t".join(["{:0.2f} / {:0.2f}".format(mean, std) for mean, std in zip(means, std_devs)])
        all_string_data = "\t".join([string_scene_dims, string_total_scene_len, string_mean_std_data])
        print("{}\t{}".format(ds_name, all_string_data))
        rows.append(all_data)

    print(datasets.keys())


if __name__ == "__main__":
    opentraj_root = "../../../"#sys.argv[1]
    output_dir = "../../output_human"#sys.argv[2]
    all_dataset_names = [
    'ETH-Univ',
    'ETH-Hotel',]
    # 'UCY-Zara',
    # 'UCY-Univ',
    # 'SDD-coupa',
    # 'SDD-bookstore',
    # 'SDD-deathCircle',
    # 'SDD-gates',
    # 'SDD-hyang',
    # 'SDD-little',
    # 'SDD-nexus',
    # 'SDD-quad',
    # 'PETS-S2l1',
    # 'TownCenter',
    # 'GC',
    # 'KITTI',
    # 'WildTrack',
    # 'Edinburgh',
    # 'BN-1d-w180',
    # 'BN-2d-w160']
    # 'LCas-Minerva','InD-1','InD-2',]

    datasets = get_datasets(opentraj_root, all_dataset_names)
    run(datasets, output_dir)
