
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random
import habitat
from habitat.config.default import get_config
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import os
import gzip
import json
from models.semantic_grid import SemanticGrid
import test_utils as tutils
import time
import quaternion
from transformers import BertTokenizer
import math
from models.img_segmentation import get_img_segmentor_from_options
import torch.nn as nn


class HabitatDataVLN(Dataset):

    # Loads necessary data for the actual VLN task

    def __init__(self, options, config_file, scene_id, existing_episode_list=[], random_poses=False, pose_noise=1):

        self.options = options
        self.scene_id = scene_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_poses_per_example = options.num_poses_per_example

        self.parse_episodes(self.options.datasets)
        
        self.number_of_episodes = len(self.scene_data["episodes"])

        cfg = habitat.get_config(config_file)
        cfg.defrost()
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb'
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.object_labels = options.n_object_classes
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.global_dim = (options.global_dim, options.global_dim)
        self.cell_size = options.cell_size
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)
        self.num_waypoints = options.num_waypoints
        self.min_angle_noise = np.radians(-15)
        self.max_angle_noise = np.radians(15)
        self.img_size = (options.img_size, options.img_size)
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

        # Initialize the semantic grid only to use the spatialTransformer. The crop_size (heatmap_size) argument does not matter here
        self.sg = SemanticGrid(1, self.grid_dim, options.heatmap_size, self.cell_size,
                                    spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)

        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]
        
        self.random_poses = random_poses
        self.pose_noise = pose_noise # used during store_vln episodes


    def parse_episodes(self, sets):

        self.scene_data = {'episodes': []}

        for s in sets:

            if s=='R2R_VLNCE_v1-2':
                root_rxr_dir = self.options.root_path + s + "/"
                episode_file = root_rxr_dir + self.options.split + "/" + self.options.split + ".json.gz"
                with gzip.open(episode_file, "rt") as fp:
                    self.data = json.load(fp)

                # Load the gt information from R2R_VLNCE
                episode_file_gt = self.options.root_path+s+"_preprocessed/"+self.options.split +"/"+self.options.split+"_gt.json.gz"
                with gzip.open(episode_file_gt, "rt") as fp:
                    self.data_gt = json.load(fp)
                
                # Need to keep only episodes that belong to current scene
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    if sc_id == self.scene_id:                        
                        # check whether goal is at the same height as start position
                        start_pos = self.data['episodes'][i]['start_position']
                        goal_pos = self.data['episodes'][i]['goals'][0]['position']
                        if np.absolute(start_pos[1] - goal_pos[1]) < 0.2:
                            self.data['episodes'][i]['scene_id'] = self.scene_id
                            self.data['episodes'][i]['dataset'] = s
                            
                            # get gt info
                            gt_info = self.data_gt[ str(self.data['episodes'][i]['episode_id']) ] # locations, forward_steps, actions
                            self.data['episodes'][i]['waypoints'] = gt_info['locations']
                            self.data['episodes'][i]['actions'] = gt_info['actions']

                            self.scene_data['episodes'].append(self.data['episodes'][i])
                            

    def __len__(self):
        return self.number_of_episodes

    def get_covered_waypoints(self, waypoints_pose_coords, pose_coords):
        covered = torch.zeros((len(waypoints_pose_coords)))
        dist = np.linalg.norm(waypoints_pose_coords.cpu().numpy() - pose_coords.cpu().numpy(), axis=-1)
        ind = np.argmin(dist)
        covered[:ind] = 1
        return covered

    def sample_random_poses(self, episode):
        idx_pos = random.sample(list(range(len(episode['waypoints']))), self.num_poses_per_example)
        idx_pos.sort()
        idx_pos[0] = 0 # always include the initial position
        init_positions = np.asarray(episode['waypoints'])[idx_pos]
        sim_positions = np.zeros((init_positions.shape[0],3))
        # add noise to the positions, need to check whether the new location is navigable
        for i in range(len(init_positions)):
            valid=False
            while not valid:
                x_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                z_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                loc = init_positions[i].copy()
                loc[0] = loc[0] + x_noise
                loc[2] = loc[2] + z_noise
                if self.sim.is_navigable(loc):
                    valid=True
            sim_positions[i,:] = loc
        # randomly select the orientations
        theta_rand = np.random.uniform(low=-np.pi, high=np.pi, size=len(sim_positions))
        sim_rotations = []
        for k in range(len(sim_positions)):
            sim_rotations.append( quaternion.from_euler_angles([0, theta_rand[k], 0]) )
        sim_positions = sim_positions.tolist()
        return sim_positions, sim_rotations


    def __getitem__(self, idx):
        
        episode = self.scene_data['episodes'][idx]

        instruction = episode['instruction']['instruction_text']
        
        init_waypoints = episode['waypoints']
        actions = episode['actions'][:-1]
        goal_position = episode['goals'][0]['position']

        k = math.ceil(len(init_waypoints) / (self.num_waypoints))
        waypoints = init_waypoints[::k]

        if len(waypoints) == self.num_waypoints:
            waypoints = waypoints[:-1]
            waypoints.append(goal_position) # remove last point and put the goal
        else:
            while len(waypoints) < self.num_waypoints:
                waypoints.append(goal_position)

        if len(waypoints) > self.num_waypoints:
            raise Exception('Waypoints contains more than '+str(self.num_waypoints))

        # set simulator pose at episode start
        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])

        # To sample locations with noise, randomly select 10 locations from episode['waypoints']
        # and randomly select orientation and add noise to the position. Move the simulator directly to those locations
        if self.random_poses:
            sim_positions, sim_rotations = self.sample_random_poses(episode)
            iterations = len(sim_positions)
        else:
            iterations = len(actions)

        abs_poses = []
        gt_maps = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        goal_maps = torch.zeros((iterations, self.num_waypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=torch.float32, device=self.device)
        visible_waypoints = torch.zeros((iterations, self.num_waypoints))
        covered_waypoints = torch.zeros((iterations, self.num_waypoints))

        ### Get egocentric map at each waypoint location along with its corresponding relative goal
        for t in range(iterations):

            if self.random_poses:
                self.sim.set_agent_state(sim_positions[t], sim_rotations[t])

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())
            abs_poses.append(agent_pose)

            # get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq, color_pcloud = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_objects.copy(),
                                                                height=y_height,
                                                                color_pcloud=self.color_pcloud)

            gt_map_semantic, gt_map_color = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, color_pcloud=color_pcloud, z=z)
            
            gt_maps[t,:,:,:] = gt_map_semantic

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
            _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
            _rel_pose = _rel_pose.to(self.device)
            pose_coords = tutils.get_coord_pose(self.sg, _rel_pose, abs_poses[t], self.grid_dim[0], self.cell_size, self.device) # B x T x 3
            #print(pose_coords) # should always be in the middle of the map

            # Transform waypoints with respect to agent current pose
            waypoints_pose_coords = torch.zeros((len(waypoints), 2))
            for k in range(len(waypoints)):
                point_pose_coords, visible = tutils.transform_to_map_coords(sg=self.sg, position=waypoints[k], abs_pose=abs_poses[t], 
                                                                                    grid_size=self.grid_dim[0], cell_size=self.cell_size, device=self.device)
                waypoints_pose_coords[k,:] = point_pose_coords.squeeze(0).squeeze(0)
                visible_waypoints[t,k] = visible

            # Find the waypoints already covered in the episode
            covered_waypoints[t,:] = self.get_covered_waypoints(waypoints_pose_coords, pose_coords.squeeze(0))

            waypoints_heatmaps = utils.locs_to_heatmaps(keypoints=waypoints_pose_coords, img_size=self.grid_dim, out_size=self.heatmap_size)
            goal_maps[t,:,:,:] = waypoints_heatmaps

            if not self.random_poses:
                action_id = actions[t]
                self.sim.step(action_id)
        
        abs_poses = torch.from_numpy(np.asarray(abs_poses)).float()    

        if not self.random_poses:
            # Sample snapshots along the episode to get num_poses_per_example size
            k = math.ceil(len(actions) / (self.num_poses_per_example))
            inds = list(range(0,len(actions),k))
            avail = list(set(list(range(0,len(actions)))) - set(inds))
            while len(inds) < self.num_poses_per_example:
                inds.append(random.sample(avail, 1)[0])
            while len(inds) > self.num_poses_per_example:
                inds = inds[-1]
            inds.sort()
            abs_poses = abs_poses[inds, :] # num_poses x 3
            goal_maps = goal_maps[inds, :, :, :] # num_poses x num_waypoints x 64 x 64
            gt_maps = gt_maps[inds, :, :, :] # num_poses x 1 x 192 x 192
            visible_waypoints = visible_waypoints[inds, :] # num_poses x num_waypoints
            covered_waypoints = covered_waypoints[inds, :] # num_poses x num_waypoints

        item = {}
        item['goal_heatmap'] = goal_maps #goal_heatmap
        item['map_semantic'] = gt_maps #gt_map_semantic.cpu()
        item['abs_pose'] = abs_poses
        item['goal_position'] = goal_position # absolute goal position, consistent within an episode
        item['instruction'] = instruction #episode['instruction']['instruction_text']
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints

        item['dataset'] = episode['dataset']
        item['episode_id'] = episode['episode_id']
        
        return item



class HabitatDataVLNOffline(Dataset):
    # Loads stored episodes for the VLN task

    def __init__(self, options, eval_set, use_all=False, offline_eval=False):

        self.episodes_file_list = []
        self.episodes_file_list += self.collect_stored_episodes(options, split=options.split)
        self.vln_no_map = options.vln_no_map
        self.use_all = use_all
        self.offline_eval = offline_eval
        self.sample_1 = options.sample_1

        if options.dataset_percentage < 1: # Randomly choose the subset of the dataset to be used
            random.shuffle(self.episodes_file_list)
            self.episodes_file_list = self.episodes_file_list[ :int(len(self.episodes_file_list)*options.dataset_percentage) ]


        if self.use_all: # no train/eval split
            self.episodes_idx = list(range(len(self.episodes_file_list)))
        else:
            # Do train-test split to ensure that all scenes are observed during training
            # dictionary for scene occurences
            scenes={}
            for i in range(len(self.episodes_file_list)):
                ep_path = self.episodes_file_list[i]
                scene_id = ep_path.split('/')[-2]            
                if scene_id not in scenes:
                    scenes[scene_id] = [] #0
                scenes[scene_id].append(i)  #+= 1

            train_set, val_set = [], []
            for key in scenes.keys():
                idxs = scenes[key]
                cut = int(len(idxs)*0.95)
                train_set += idxs[:cut]
                val_set += idxs[cut:]
                #print(cut, len(train_set), len(val_set))

            if not eval_set:
                self.episodes_idx = train_set
            else:
                self.episodes_idx = val_set        
        
        self.number_of_episodes = len(self.episodes_idx)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = 512 # maximum sequence length for instruction that BERT can take


    def collect_stored_episodes(self, options, split):
        episodes_dir = options.stored_episodes_dir + split + "/"
        episodes_file_list = []
        _scenes_dir = os.listdir(episodes_dir)
        scenes_dir = [ x for x in _scenes_dir if os.path.isdir(episodes_dir+x) ]
        for scene in scenes_dir:
            for fil in os.listdir(episodes_dir+scene+"/"):
                to_add = False
                # add training examples only from the selected datasets
                for dataset in options.datasets:
                    if dataset in fil:
                        to_add = True
                if to_add:
                    episodes_file_list.append(episodes_dir+scene+"/"+fil)
        return episodes_file_list


    def __len__(self):
        return self.number_of_episodes


    def __getitem__(self, idx):
        ind = self.episodes_idx[idx] # get the next episode index
        ep_file = self.episodes_file_list[ind]
        ep = np.load(ep_file)

        goal_heatmap = torch.from_numpy(ep['goal_heatmap']) # num_poses x num_waypoints x 64 x 64
        map_semantic = torch.from_numpy(ep['map_semantic']) # num_poses x 1 x 192 x 192
        visible_waypoints = torch.from_numpy(ep['visible_waypoints']) # num_poses x num_waypoints
        covered_waypoints = torch.from_numpy(ep['covered_waypoints']) # num_poses x num_waypoints

        if self.vln_no_map:
            step_ego_grid_maps = torch.from_numpy(ep['step_ego_grid_maps']) # num_poses x 3 x 192 x 192
            map_occupancy = torch.from_numpy(ep['map_occupancy']) # num_poses x 1 x 192 x 192
            ego_segm_maps = torch.from_numpy(ep['ego_segm_maps']) # num_poses x 27 x 192 x 192

        if self.sample_1: # forced to do this due to memory issues
            ind = random.randint(0,map_semantic.shape[0]-1) # Randomly choose one example from the sequence
            #ind=0
            goal_heatmap = goal_heatmap[ind,:,:,:].unsqueeze(0) # 1 x num_waypoints x h x w
            map_semantic = map_semantic[ind,:,:,:].unsqueeze(0)
            visible_waypoints = visible_waypoints[ind,:].unsqueeze(0)
            covered_waypoints = covered_waypoints[ind,:].unsqueeze(0)
            if self.vln_no_map:
                step_ego_grid_maps = step_ego_grid_maps[ind,:,:,:].unsqueeze(0)
                map_occupancy = map_occupancy[ind,:,:,:].unsqueeze(0)
                ego_segm_maps = ego_segm_maps[ind,:,:,:].unsqueeze(0)

        instruction = ep['instruction'].tobytes().decode("utf-8")
        instruction = "[CLS] " + instruction + " [SEP]"
        
        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(instruction)
        # Map the token strings to their vocabulary indices.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Truncate very large instructions to max length (512)
        if tokens_tensor.shape[1] > self.max_seq_length:
            tokens_tensor = tokens_tensor[:,:self.max_seq_length]
            segments_tensors = segments_tensors[:,:self.max_seq_length]

        item = {}

        item['goal_heatmap'] = goal_heatmap # T x num_waypoints x 64 x 64
        item['map_semantic'] = map_semantic # T x 1 x 192 x 192
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints
        item['tokens_tensor'] = tokens_tensor
        item['segments_tensors'] = segments_tensors

        if self.offline_eval:
            item['tokens'] = tokenized_text
            item['instruction'] = ep['instruction'] #instruction

        if self.vln_no_map:
            item['step_ego_grid_maps'] = step_ego_grid_maps
            item['map_occupancy'] = map_occupancy
            item['ego_segm_maps'] = ego_segm_maps

        return item



### Dataloader for storing data in the unknown map case

class HabitatDataVLN_UnknownMap(Dataset):

    # Loads necessary data for the actual VLN task

    def __init__(self, options, config_file, scene_id, existing_episode_list=[], random_poses=False, pose_noise=1):

        self.options = options
        self.scene_id = scene_id
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.num_poses_per_example = options.num_poses_per_example

        self.parse_episodes(self.options.datasets)
        
        self.number_of_episodes = len(self.scene_data["episodes"])


        cfg = habitat.get_config(config_file)
        cfg.defrost()
        cfg.SIMULATOR.SCENE = options.root_path + options.scenes_dir + "mp3d/" + scene_id + '/' + scene_id + '.glb'
        #cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH = False
        cfg.SIMULATOR.TURN_ANGLE = options.turn_angle
        cfg.SIMULATOR.FORWARD_STEP_SIZE = options.forward_step_size
        cfg.freeze()

        self.sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)

        seed = 0
        self.sim.seed(seed)

        self.hfov = float(cfg.SIMULATOR.DEPTH_SENSOR.HFOV) * np.pi / 180.
        self.cfg_norm_depth = cfg.SIMULATOR.DEPTH_SENSOR.NORMALIZE_DEPTH
        self.object_labels = options.n_object_classes
        self.spatial_labels = options.n_spatial_classes
        self.global_dim = (options.global_dim, options.global_dim)
        self.grid_dim = (options.grid_dim, options.grid_dim)
        self.cell_size = options.cell_size
        self.heatmap_size = (options.heatmap_size, options.heatmap_size)
        self.num_waypoints = options.num_waypoints
        self.min_angle_noise = np.radians(-15)
        self.max_angle_noise = np.radians(15)
        self.img_size = (options.img_size, options.img_size)
        self.img_segm_size = (options.img_segm_size, options.img_segm_size)
        self.normalize = True
        self.pixFormat = 'NCHW'
        self.max_depth = cfg.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH
        self.min_depth = cfg.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH

        self.preprocessed_scenes_dir = options.root_path + options.scenes_dir + "mp3d_scene_pclouds/"

         # get point cloud and labels of scene
        self.pcloud, self.label_seq_spatial, self.label_seq_objects = utils.load_scene_pcloud(self.preprocessed_scenes_dir,
                                                                                                    self.scene_id, self.object_labels)
        self.color_pcloud = utils.load_scene_color(self.preprocessed_scenes_dir, self.scene_id)

        # Initialize the semantic grid only to use the spatialTransformer. The crop_size (heatmap_size) argument does not matter here
        self.sg = SemanticGrid(1, self.grid_dim, options.heatmap_size, self.cell_size,
                                    spatial_labels=options.n_spatial_classes, object_labels=options.n_object_classes)

        if len(existing_episode_list)!=0:
            self.existing_episode_list = [ int(x.split('_')[2]) for x in existing_episode_list ]
        else:
            self.existing_episode_list=[]
        
        self.random_poses = random_poses
        self.pose_noise = pose_noise # used during store_vln episodes

        self.occupancy_height_thresh = options.occupancy_height_thresh

        # Build 3D transformation matrices for the occupancy egocentric grid
        self.xs, self.ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_size[0]), np.linspace(1,-1,self.img_size[1])), device='cuda')
        self.xs = self.xs.reshape(1,self.img_size[0],self.img_size[1])
        self.ys = self.ys.reshape(1,self.img_size[0],self.img_size[1])
        K = np.array([
            [1 / np.tan(self.hfov / 2.), 0., 0., 0.],
            [0., 1 / np.tan(self.hfov / 2.), 0., 0.],
            [0., 0.,  1, 0],
            [0., 0., 0, 1]])
        self.inv_K = torch.tensor(np.linalg.inv(K), device='cuda')
        # create the points2D containing all image coordinates
        x, y = torch.tensor(np.meshgrid(np.linspace(0, self.img_size[0]-1, self.img_size[0]), np.linspace(0, self.img_size[1]-1, self.img_size[1])), device='cuda')
        xy_img = torch.vstack((x.reshape(1,self.img_size[0],self.img_size[1]), y.reshape(1,self.img_size[0],self.img_size[1])))
        points2D_step = xy_img.reshape(2, -1)
        self.points2D_step = torch.transpose(points2D_step, 0, 1) # Npoints x 2


        ## Load the pre-trained img segmentation model
        self.img_segmentor = get_img_segmentor_from_options(options)
        self.img_segmentor = self.img_segmentor.to(self.device)
        self.img_segmentor = nn.DataParallel(self.img_segmentor)
        latest_checkpoint = tutils.get_latest_model(save_dir=options.img_segm_model_dir)
        print("Loading image segmentation checkpoint", latest_checkpoint)
        checkpoint = torch.load(latest_checkpoint)
        self.img_segmentor.load_state_dict(checkpoint['models']['img_segm_model'])         
        self.img_segmentor.eval()
        

        ## Build necessary info for ground-projecting the semantic segmentation
        self._xs, self._ys = torch.tensor(np.meshgrid(np.linspace(-1,1,self.img_segm_size[0]), np.linspace(1,-1,self.img_segm_size[1])), device=self.device)
        self._xs = self._xs.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        self._ys = self._ys.reshape(1,self.img_segm_size[0],self.img_segm_size[1])
        _x, _y = torch.tensor(np.meshgrid(np.linspace(0, self.img_segm_size[0]-1, self.img_segm_size[0]), 
                                                    np.linspace(0, self.img_segm_size[1]-1, self.img_segm_size[1])), device=self.device)
        _xy_img = torch.cat((_x.reshape(1,self.img_segm_size[0],self.img_segm_size[1]), _y.reshape(1,self.img_segm_size[0],self.img_segm_size[1])), dim=0)
        _points2D_step = _xy_img.reshape(2, -1)
        self._points2D_step = torch.transpose(_points2D_step, 0, 1) # Npoints x 2        


    def parse_episodes(self, sets):

        self.scene_data = {'episodes': []}

        for s in sets:

            if s=='R2R_VLNCE_v1-2':
                root_rxr_dir = self.options.root_path + s + "/"
                episode_file = root_rxr_dir + self.options.split + "/" + self.options.split + ".json.gz"
                with gzip.open(episode_file, "rt") as fp:
                    self.data = json.load(fp)

                # Load the gt information from R2R_VLNCE
                episode_file_gt = self.options.root_path+s+"_preprocessed/"+self.options.split +"/"+self.options.split+"_gt.json.gz"
                with gzip.open(episode_file_gt, "rt") as fp:
                    self.data_gt = json.load(fp)
                
                # Need to keep only episodes that belong to current scene
                for i in range(len(self.data['episodes'])):
                    sc_path = self.data['episodes'][i]['scene_id']
                    sc_id = sc_path.split('/')[-1].split('.')[0]
                    if sc_id == self.scene_id:                        
                        # check whether goal is at the same height as start position
                        start_pos = self.data['episodes'][i]['start_position']
                        goal_pos = self.data['episodes'][i]['goals'][0]['position']
                        if np.absolute(start_pos[1] - goal_pos[1]) < 0.2:
                            self.data['episodes'][i]['scene_id'] = self.scene_id
                            self.data['episodes'][i]['dataset'] = s
                            
                            # get gt info
                            gt_info = self.data_gt[ str(self.data['episodes'][i]['episode_id']) ] # locations, forward_steps, actions
                            self.data['episodes'][i]['waypoints'] = gt_info['locations']
                            self.data['episodes'][i]['actions'] = gt_info['actions']

                            self.scene_data['episodes'].append(self.data['episodes'][i])
                            

    def __len__(self):
        return self.number_of_episodes


    def get_covered_waypoints(self, waypoints_pose_coords, pose_coords):
        covered = torch.zeros((len(waypoints_pose_coords)))
        dist = np.linalg.norm(waypoints_pose_coords.cpu().numpy() - pose_coords.cpu().numpy(), axis=-1)
        ind = np.argmin(dist)
        covered[:ind] = 1
        return covered


    def sample_random_poses(self, episode):
        idx_pos = random.sample(list(range(len(episode['waypoints']))), self.num_poses_per_example)
        idx_pos.sort()
        idx_pos[0] = 0 # always include the initial position
        init_positions = np.asarray(episode['waypoints'])[idx_pos]
        sim_positions = np.zeros((init_positions.shape[0],3))
        # add noise to the positions, need to check whether the new location is navigable
        for i in range(len(init_positions)):
            valid=False
            while not valid:
                x_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                z_noise = np.random.uniform(low=-self.pose_noise, high=self.pose_noise, size=1)
                loc = init_positions[i].copy()
                loc[0] = loc[0] + x_noise
                loc[2] = loc[2] + z_noise
                if self.sim.is_navigable(loc):
                    valid=True
            sim_positions[i,:] = loc
        # randomly select the orientations
        theta_rand = np.random.uniform(low=-np.pi, high=np.pi, size=len(sim_positions))
        sim_rotations = []
        for k in range(len(sim_positions)):
            sim_rotations.append( quaternion.from_euler_angles([0, theta_rand[k], 0]) )
        sim_positions = sim_positions.tolist()
        return sim_positions, sim_rotations



    def __getitem__(self, idx):
        
        episode = self.scene_data['episodes'][idx]

        instruction = episode['instruction']['instruction_text']
        
        init_waypoints = episode['waypoints']
        actions = episode['actions'][:-1]
        goal_position = episode['goals'][0]['position']

        k = math.ceil(len(init_waypoints) / (self.num_waypoints))
        waypoints = init_waypoints[::k]

        if len(waypoints) == self.num_waypoints:
            waypoints = waypoints[:-1]
            waypoints.append(goal_position) # remove last point and put the goal
        else:
            while len(waypoints) < self.num_waypoints:
                waypoints.append(goal_position)

        if len(waypoints) > self.num_waypoints:
            raise Exception('Waypoints contains more than '+str(self.num_waypoints))


        # Get the 3D scene semantics
        scene = self.sim.semantic_annotations()
        instance_id_to_label_id = {int(obj.id.split("_")[-1]): obj.category.index() for obj in scene.objects}
        # convert the labels to the reduced set of categories
        instance_id_to_label_id_occ = instance_id_to_label_id.copy()
        instance_id_to_label_id_sem = instance_id_to_label_id.copy()
        for inst_id in instance_id_to_label_id.keys():
            curr_lbl = instance_id_to_label_id[inst_id]
            instance_id_to_label_id_occ[inst_id] = viz_utils.label_conversion_40_3[curr_lbl]
            instance_id_to_label_id_sem[inst_id] = viz_utils.label_conversion_40_27[curr_lbl]      


        # set simulator pose at episode start
        self.sim.reset()
        self.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
        sim_obs = self.sim.get_sensor_observations()
        observations = self.sim._sensor_suite.get_observations(sim_obs)

        # To sample locations with noise, randomly select 10 locations from episode['waypoints']
        # and randomly select orientation and add noise to the position. Move the simulator directly to those locations
        if self.random_poses:
            sim_positions, sim_rotations = self.sample_random_poses(episode)
            iterations = len(sim_positions)
        else:
            iterations = len(actions)

        abs_poses = []
        rel_abs_poses = []
        local3D = []
        gt_maps = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        gt_maps_occ = torch.zeros((iterations, 1, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        ego_segm_maps = torch.zeros((iterations, self.object_labels, self.grid_dim[0], self.grid_dim[1]), dtype=torch.float32, device=self.device)
        goal_maps = torch.zeros((iterations, self.num_waypoints, self.heatmap_size[0], self.heatmap_size[1]), dtype=torch.float32, device=self.device)
        visible_waypoints = torch.zeros((iterations, self.num_waypoints))
        covered_waypoints = torch.zeros((iterations, self.num_waypoints))

        ### Get egocentric map at each waypoint location along with its corresponding relative goal
        for t in range(iterations):

            if self.random_poses:
                self.sim.set_agent_state(sim_positions[t], sim_rotations[t])
                sim_obs = self.sim.get_sensor_observations()
                observations = self.sim._sensor_suite.get_observations(sim_obs)

            img = observations['rgb'][:,:,:3]
            depth_obsv = observations['depth'].permute(2,0,1).unsqueeze(0)

            depth = F.interpolate(depth_obsv.clone(), size=self.img_size, mode='nearest')
            depth = depth.squeeze(0).permute(1,2,0) 

            if self.cfg_norm_depth:
                depth = utils.unnormalize_depth(depth, min=self.min_depth, max=self.max_depth)

            local3D_step = utils.depth_to_3D(depth, self.img_size, self.xs, self.ys, self.inv_K)
            local3D.append(local3D_step)

            agent_pose, y_height = utils.get_sim_location(agent_state=self.sim.get_agent_state())
            abs_poses.append(agent_pose)

            # get the relative pose with respect to the first pose in the sequence
            rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
            rel_abs_poses.append(rel_abs_pose)

            ### Run the img segmentation model to get the ground-projected semantic segmentation
            depth_img = depth.permute(2,0,1).unsqueeze(0)
            depth_img = F.interpolate(depth_img, size=self.img_segm_size, mode='nearest')
            
            imgData = utils.preprocess_img(img, cropSize=self.img_segm_size, pixFormat=self.pixFormat, normalize=self.normalize)
            segm_batch = {'images':imgData.to(self.device).unsqueeze(0).unsqueeze(0),
                        'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

            pred_ego_crops_sseg, _ = utils.run_img_segm(model=self.img_segmentor, 
                                                    input_batch=segm_batch, 
                                                    object_labels=self.object_labels, 
                                                    crop_size=self.grid_dim, 
                                                    cell_size=self.cell_size,
                                                    xs=self._xs,
                                                    ys=self._ys,
                                                    inv_K=self.inv_K,
                                                    points2D_step=self._points2D_step)            
            pred_ego_crops_sseg = pred_ego_crops_sseg.squeeze(0)
            ego_segm_maps[t,:,:,:] = pred_ego_crops_sseg

            ### Get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq, color_pcloud = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_objects.copy(),
                                                                height=y_height,
                                                                color_pcloud=self.color_pcloud)
            gt_map_semantic, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, color_pcloud=color_pcloud, z=z)
            gt_maps[t,:,:,:] = gt_map_semantic
            
            # get gt map from agent pose (pose is at the center looking upwards)
            x, y, z, label_seq_occ = map_utils.slice_scene(x=self.pcloud[0].copy(),
                                                                y=self.pcloud[1].copy(),
                                                                z=self.pcloud[2].copy(),
                                                                label_seq=self.label_seq_spatial.copy(),
                                                                height=y_height)
            gt_map_occupancy = map_utils.get_gt_map(x, y, label_seq_occ, abs_pose=abs_poses[t],
                                                        grid_dim=self.grid_dim, cell_size=self.cell_size, z=z)
            gt_maps_occ[t,:,:,:] = gt_map_occupancy

            # get the relative pose with respect to the first pose in the sequence
            rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
            _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
            _rel_pose = _rel_pose.to(self.device)
            pose_coords = tutils.get_coord_pose(self.sg, _rel_pose, abs_poses[t], self.grid_dim[0], self.cell_size, self.device) # B x T x 3
            #print(pose_coords) # should always be in the middle of the map

            # Transform waypoints with respect to agent current pose
            waypoints_pose_coords = torch.zeros((len(waypoints), 2))
            for k in range(len(waypoints)):
                point_pose_coords, visible = tutils.transform_to_map_coords(sg=self.sg, position=waypoints[k], abs_pose=abs_poses[t], 
                                                                                        grid_size=self.grid_dim[0], cell_size=self.cell_size, device=self.device)
                waypoints_pose_coords[k,:] = point_pose_coords.squeeze(0).squeeze(0)
                visible_waypoints[t,k] = visible

            # Find the waypoints already covered in the episode
            covered_waypoints[t,:] = self.get_covered_waypoints(waypoints_pose_coords, pose_coords.squeeze(0))

            waypoints_heatmaps = utils.locs_to_heatmaps(keypoints=waypoints_pose_coords, img_size=self.grid_dim, out_size=self.heatmap_size)
            goal_maps[t,:,:,:] = waypoints_heatmaps

            if not self.random_poses:
                action_id = actions[t]
                observations = self.sim.step(action_id)


        abs_poses = torch.from_numpy(np.asarray(abs_poses)).float()
        rel_abs_poses = torch.from_numpy(np.asarray(rel_abs_poses)).float()

        ### Get the ground-projected observation from the accumulated projected map
        ego_grid_maps = map_utils.est_occ_from_depth(local3D, grid_dim=self.global_dim, cell_size=self.cell_size, 
                                                    device=self.device, occupancy_height_thresh=self.occupancy_height_thresh)
        step_ego_grid_maps = map_utils.get_acc_proj_grid(ego_grid_maps, rel_abs_poses, abs_poses, self.grid_dim, self.cell_size)
        step_ego_grid_maps = map_utils.crop_grid(grid=step_ego_grid_maps, crop_size=self.grid_dim)


        if not self.random_poses:
            # Sample snapshots along the episode to get num_poses_per_example size
            k = math.ceil(len(actions) / (self.num_poses_per_example))
            inds = list(range(0,len(actions),k))
            avail = list(set(list(range(0,len(actions)))) - set(inds))
            while len(inds) < self.num_poses_per_example:
                inds.append(random.sample(avail, 1)[0])
            while len(inds) > self.num_poses_per_example:
                inds = inds[-1]
            inds.sort()
            abs_poses = abs_poses[inds, :] # num_poses x 3
            goal_maps = goal_maps[inds, :, :, :] # num_poses x num_waypoints x 64 x 64
            step_ego_grid_maps = step_ego_grid_maps[inds, :, :, :] # num_poses x spatial_labels x 192 x 192
            ego_segm_maps = ego_segm_maps[inds, :, :, :]
            gt_maps = gt_maps[inds, :, :, :] # num_poses x 1 x 192 x 192
            gt_maps_occ = gt_maps_occ[inds, :, :, :]
            visible_waypoints = visible_waypoints[inds, :] # num_poses x num_waypoints
            covered_waypoints = covered_waypoints[inds, :] # num_poses x num_waypoints


        item = {}
        item['goal_heatmap'] = goal_maps
        item['step_ego_grid_maps'] = step_ego_grid_maps
        item['ego_segm_maps'] = ego_segm_maps
        item['map_semantic'] = gt_maps
        item['map_occupancy'] = gt_maps_occ
        item['abs_pose'] = abs_poses
        item['instruction'] = instruction
        item['visible_waypoints'] = visible_waypoints
        item['covered_waypoints'] = covered_waypoints

        item['dataset'] = episode['dataset']
        item['episode_id'] = episode['episode_id']
        
        return item