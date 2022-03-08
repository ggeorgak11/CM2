
import multiprocessing as mp
from multiprocessing import Pool, TimeoutError
import numpy as np
from datasets.dataloader import HabitatDataVLN_UnknownMap
import datasets.util.utils as utils
import os
import argparse
import torch
import random
import json


class Params(object):
    def __init__(self):
        self.parser = argparse.ArgumentParser()

        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val_seen', 'val_unseen', 'test'])

        self.parser.add_argument('--global_dim', type=int, dest='global_dim', default=512)
        self.parser.add_argument('--grid_dim', type=int, dest='grid_dim', default=192)
        self.parser.add_argument('--cell_size', type=float, dest='cell_size', default=0.05)
        self.parser.add_argument('--turn_angle', type=int, dest='turn_angle', default=15)
        self.parser.add_argument('--forward_step_size', type=float, dest='forward_step_size', default=0.25)
        self.parser.add_argument('--n_object_classes', type=int, dest='n_object_classes', default=27)
        self.parser.add_argument('--n_spatial_classes', type=int, dest='n_spatial_classes', default=3)

        self.parser.add_argument('--heatmap_size', dest='heatmap_size', type=int, default=24,
                                 help='Waypoint heatmap size, should match hourglass output size.')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)

        self.parser.add_argument('--max_num_episodes', dest='max_num_episodes', type=int, default=2500)

        self.parser.add_argument('--num_poses_per_example', dest='num_poses_per_example', type=int, default=10,
                                help='when storing episodes for vln how many poses to use in the same episode')

        self.parser.add_argument('--num_waypoints', dest='num_waypoints', type=int, default=10,
                                 help='Number of waypoints sampled for each trajectory.')

        self.parser.add_argument('--random_poses', dest='random_poses', default=False, action='store_true',
                                help='Enable random pose sampling instead of following gt path')
        self.parser.add_argument('--pose_noise', dest='pose_noise', type=float, default=1.0, 
                                help='(-value,value) range to sample noise for the poses')

        self.parser.add_argument('--check_existing', dest='check_existing', default=False, action='store_true',
                                help='If enabled does not repeat the same episode id (i)')        

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                help='used when estimating occupancy from depth')

        self.parser.add_argument('--config_file', type=str, dest='config_file',
                                default='configs/habitat_config.yaml')

        self.parser.add_argument('--scenes_list', nargs='+')

        self.parser.add_argument('--datasets', nargs='+', default=['R2R_VLNCE_v1-2'],
                                help='Datasets to use when generating training episodes for vln')


        self.parser.add_argument('--root_path', type=str, dest='root_path', default="../")

        self.parser.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default="../",
                                    help='job path that contains the pre-trained img segmentation model')
        self.parser.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale') 

        self.parser.add_argument('--episodes_save_dir', type=str, dest='episodes_save_dir', default="../")
        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='../')

        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)


def store_episodes(options, config_file, scene_id):

    save_path = options.episodes_save_dir + options.split + "/" + scene_id + "/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    with open(os.path.join(options.episodes_save_dir+options.split+"/", 'options.json'), "w") as f:
        json.dump(vars(options), f, indent=4)

    existing_episode_list = os.listdir(save_path) # keep track of previously saved episodes

    data = HabitatDataVLN_UnknownMap(options, config_file, scene_id=scene_id, existing_episode_list=existing_episode_list, random_poses=options.random_poses, pose_noise=options.pose_noise)

    print(len(data))

    ep_count = len(existing_episode_list)
    for i in range(len(data)):

        if i in data.existing_episode_list and options.check_existing:
            print("Episode", i, 'already exists!')
            continue        

        if ep_count >= options.max_num_episodes:
            break

        ex = data[i] # collect example

        if ex is None:
            continue

        ep_count+=1

        abs_pose = ex['abs_pose']
        goal_heatmap = ex['goal_heatmap'].cpu()
        map_semantic = ex['map_semantic'].cpu()
        map_occupancy = ex['map_occupancy'].cpu()
        step_ego_grid_maps = ex['step_ego_grid_maps'].cpu()
        ego_segm_maps = ex['ego_segm_maps'].cpu()
        instruction = ex['instruction']
        visible_waypoints = ex['visible_waypoints']
        covered_waypoints = ex['covered_waypoints']

        dataset = ex['dataset']
        episode_id = ex['episode_id']

        print('Saving episode', ep_count, 'of id', i, 'scene', scene_id)

        filepath = save_path+'ep_'+str(ep_count)+'_'+str(i)+"_"+scene_id+"_"+dataset
        np.savez_compressed(filepath+'.npz',
                            abs_pose=abs_pose,
                            goal_heatmap=goal_heatmap,
                            map_semantic=map_semantic,
                            map_occupancy=map_occupancy,
                            step_ego_grid_maps=step_ego_grid_maps,
                            ego_segm_maps=ego_segm_maps,
                            instruction=instruction,
                            visible_waypoints=visible_waypoints,
                            covered_waypoints=covered_waypoints,
                            episode_id=episode_id
                            )

    data.sim.close()


if __name__ == '__main__':

    mp.set_start_method('forkserver', force=True)
    options = Params().parser.parse_args()

    print("options:")
    for k in options.__dict__.keys():
        print(k, options.__dict__[k])

    config_file = options.config_file

    scene_ids = options.scenes_list

    # Create iterables for map function
    n = len(scene_ids)
    options_list = [options] * n
    config_files = [config_file] * n
    args = [*zip(options_list, config_files, scene_ids)]

    with Pool(processes=options.gpu_capacity) as pool:

        pool.starmap(store_episodes, args)

    # exiting the 'with'-block has stopped the pool
    print("Now the pool is closed and no longer available")
