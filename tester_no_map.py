import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets.dataloader import HabitatDataVLN_UnknownMap 
from models import get_model_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import test_utils as tutils
from models.semantic_grid import SemanticGrid
import os
import matplotlib.pyplot as plt
import json
import cv2
import random
import math
from planning.ddppo_policy import DdppoPolicy
from transformers import BertTokenizer


class VLNTesterUnknownMap(object):
    """ Implements testing for prediction models
    """
    def __init__(self, options, scene_id):
        self.options = options
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # build summary dir
        summary_dir = os.path.join(self.options.log_dir, scene_id)
        summary_dir = os.path.join(summary_dir, 'tensorboard')
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)
        # tensorboardX SummaryWriter for use in save_summaries
        self.summary_writer = SummaryWriter(summary_dir)

        self.scene_id = scene_id
        self.test_ds = HabitatDataVLN_UnknownMap(self.options, config_file=self.options.config_file, scene_id=self.scene_id)
        print(len(self.test_ds))

        # Load the goal predictor model
        self.goal_pred_model = get_model_from_options(self.options)
        self.models_dict = {'goal_pred_model':self.goal_pred_model}

        print("Using ", torch.cuda.device_count(), "gpus")
        for k in self.models_dict:
            self.models_dict[k] = nn.DataParallel(self.models_dict[k])

        latest_checkpoint = tutils.get_latest_model(save_dir=self.options.model_exp_dir)
        self.models_dict = tutils.load_model(models=self.models_dict, checkpoint_file=latest_checkpoint)
        print("Model loaded checkpoint:", latest_checkpoint)
        self.models_dict["goal_pred_model"].eval()
        
        self.step_count = 0

        self.metrics = ['distance_to_goal', 'success', 'spl', 'softspl', 'trajectory_length', 'navigation_error', 'oracle_success']
        # initialize metrics
        self.results = {}
        for met in self.metrics:
            self.results[met] = []

        # init local policy model
        if self.options.local_policy_model=="4plus":
            model_ext = 'gibson-4plus-mp3d-train-val-test-resnet50.pth'
        else:
            model_ext = 'gibson-2plus-resnet50.pth'
        model_path = self.options.root_path + "local_policy_models/" + model_ext
        self.l_policy = DdppoPolicy(path=model_path)
        self.l_policy = self.l_policy.to(self.device)

        # To tokenize the instruction
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.max_seq_length = 512 # maximum sequence length for instruction that BERT can take


    def test_navigation(self):

        with torch.no_grad():

            list_dist_to_goal, list_success, list_spl, list_soft_spl = [],[],[],[]
            list_trajectory_length, list_navigation_error, list_oracle_success = [], [], []

            for idx in range(len(self.test_ds)):

                episode = self.test_ds.scene_data['episodes'][idx]

                instruction = episode['instruction']['instruction_text']
                goal_position = episode['goals'][0]['position']

                gt_path = episode['waypoints'] # first waypoint is start position
                ## subample the gt_path to match the one from store_episodes in dataloader (for visualization)
                k = math.ceil(len(gt_path) / (self.options.num_waypoints))
                gt_waypoints = gt_path[::k]
                if len(gt_waypoints) == self.options.num_waypoints:
                    gt_waypoints = gt_waypoints[:-1]
                    gt_waypoints.append(goal_position) # remove last point and put the goal
                else:
                    while len(gt_waypoints) < self.options.num_waypoints:
                        gt_waypoints.append(goal_position)

                print("Ep:", idx, "Instr:", instruction)
                self.step_count+=1 # episode counter for tensorboard


                self.test_ds.sim.reset()
                self.test_ds.sim.set_agent_state(episode["start_position"], episode["start_rotation"])
                sim_obs = self.test_ds.sim.get_sensor_observations()
                observations = self.test_ds.sim._sensor_suite.get_observations(sim_obs)

                # For each episode we need a new instance of a fresh global grid
                sg = SemanticGrid(1, self.test_ds.grid_dim, self.options.heatmap_size, self.options.cell_size,
                                    spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)
                sg_global = SemanticGrid(1, self.test_ds.global_dim, self.options.heatmap_size, self.options.cell_size,
                                    spatial_labels=self.options.n_spatial_classes, object_labels=self.options.n_object_classes)

                # Initialize the local policy hidden state
                self.l_policy.reset()

                abs_poses = []
                agent_height = []
                sim_agent_poses = [] # for estimating the metrics at the end of the episode
                t = 0
                ltg_counter=0
                ltg_abs_coords = torch.zeros((1, 1, 2), dtype=torch.int64).to(self.device)
                ltg_abs_coords_list = []
                agent_episode_distance = 0.0 # distance covered by agent at any given time in the episode
                previous_pos = self.test_ds.sim.get_agent_state().position

                while t < self.options.max_steps:

                    img = observations['rgb'][:,:,:3]
                    depth = observations['depth'].reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)

                    if self.test_ds.cfg_norm_depth:
                        depth_abs = utils.unnormalize_depth(depth, min=self.test_ds.min_depth, max=self.test_ds.max_depth)

                    # 3d info
                    local3D_step = utils.depth_to_3D(depth_abs, self.test_ds.img_size, self.test_ds.xs, self.test_ds.ys, self.test_ds.inv_K)

                    sim_agent_poses.append(self.test_ds.sim.get_agent_state().position)

                    agent_pose, y_height = utils.get_sim_location(agent_state=self.test_ds.sim.get_agent_state())
                    abs_poses.append(agent_pose)
                    agent_height.append(y_height)

                    
                    # get gt map from agent pose for visualization later (pose is at the center looking upwards)
                    x, y, z, label_seq, color_pcloud = map_utils.slice_scene(x=self.test_ds.pcloud[0].copy(),
                                                                        y=self.test_ds.pcloud[1].copy(),
                                                                        z=self.test_ds.pcloud[2].copy(),
                                                                        label_seq=self.test_ds.label_seq_objects.copy(),
                                                                        height=y_height,
                                                                        color_pcloud=self.test_ds.color_pcloud)

                    gt_map_semantic, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[t],
                                                                grid_dim=self.test_ds.grid_dim, cell_size=self.options.cell_size, color_pcloud=color_pcloud, z=z)                    

                    # Keep track of the agent's relative pose from the initial position
                    # abs_pose_coords assumes the global_dim=512
                    rel_abs_pose = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[0])
                    _rel_abs_pose = torch.Tensor(rel_abs_pose).unsqueeze(0).float()
                    _rel_abs_pose = _rel_abs_pose.to(self.device)
                    abs_pose_coords = tutils.get_coord_pose(sg_global, _rel_abs_pose, abs_poses[0], self.test_ds.global_dim[0], self.test_ds.cell_size, self.device) # B x T x 3


                    # We operate in egocentric coords so agent should always be in the middle of the map
                    rel = utils.get_rel_pose(pos2=abs_poses[t], pos1=abs_poses[t])
                    _rel_pose = torch.Tensor(rel).unsqueeze(0).float()
                    _rel_pose = _rel_pose.to(self.device)
                    pose_coords = tutils.get_coord_pose(sg, _rel_pose, abs_poses[t], self.test_ds.grid_dim[0], self.test_ds.cell_size, self.device) # B x T x 3

                    
                    # do ground-projection, update the projected map
                    ego_grid_sseg_3 = map_utils.est_occ_from_depth([local3D_step], grid_dim=self.test_ds.global_dim, cell_size=self.test_ds.cell_size, 
                                                                                    device=self.device, occupancy_height_thresh=self.options.occupancy_height_thresh)
                    # Transform the ground projected egocentric grids to geocentric using relative pose
                    geo_grid_sseg = sg_global.spatialTransformer(grid=ego_grid_sseg_3, pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
                    # step_geo_grid contains the map snapshot every time a new observation is added
                    step_geo_grid_sseg = sg_global.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
                    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
                    step_ego_grid_sseg = sg_global.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=_rel_abs_pose, abs_pose=torch.tensor(abs_poses).to(self.device))
                    # Crop the grid around the agent at each timestep
                    step_ego_grid_maps = map_utils.crop_grid(grid=step_ego_grid_sseg, crop_size=self.test_ds.grid_dim)
                    step_ego_grid_maps = step_ego_grid_maps.unsqueeze(0)
                    

                    ### Run the img segmentation model to get the ground-projected semantic segmentation
                    depth_img = depth_abs.clone().permute(2,0,1).unsqueeze(0)
                    depth_img = F.interpolate(depth_img, size=self.test_ds.img_segm_size, mode='nearest')
                    imgData = utils.preprocess_img(img, cropSize=self.test_ds.img_segm_size, pixFormat=self.test_ds.pixFormat, normalize=self.test_ds.normalize)
                    segm_batch = {'images':imgData.to(self.device).unsqueeze(0).unsqueeze(0),
                                'depth_imgs':depth_img.to(self.device).unsqueeze(0)}

                    pred_ego_crops_sseg, img_segm = utils.run_img_segm(model=self.test_ds.img_segmentor, 
                                                            input_batch=segm_batch, 
                                                            object_labels=self.test_ds.object_labels, 
                                                            crop_size=self.test_ds.grid_dim, 
                                                            cell_size=self.test_ds.cell_size,
                                                            xs=self.test_ds._xs,
                                                            ys=self.test_ds._ys,
                                                            inv_K=self.test_ds.inv_K,
                                                            points2D_step=self.test_ds._points2D_step)                    

                    # Transform gt waypoints with respect to agent current pose (for visualization later)
                    gt_waypoints_pose_coords = torch.zeros((len(gt_waypoints), 2))
                    for k in range(len(gt_waypoints)):
                        point_pose_coords, _ = tutils.transform_to_map_coords(sg=sg, position=gt_waypoints[k], abs_pose=abs_poses[t], 
                                                                                            grid_size=self.test_ds.grid_dim[0], cell_size=self.options.cell_size, device=self.device)
                        gt_waypoints_pose_coords[k,:] = point_pose_coords.squeeze(0).squeeze(0)

                    # Model is trained to predict the waypoints in the egocentric coordinate space
                    # So they are with respect to agent's position in the center of the map
                    pred_waypoints_pose_coords, pred_waypoints_vals, waypoints_cov, pred_maps_spatial, pred_maps_objects = self.run_goal_pred(instruction, sg=sg, ego_occ_maps=step_ego_grid_maps, 
                                                                                                                            ego_segm_maps=pred_ego_crops_sseg, start_pos=episode["start_position"], pose=abs_poses[t])

                    # Since we assume --use_first_waypoint is enabled we need to add the first waypoint in the pred
                    pred_waypoints_pose_coords = torch.cat( (gt_waypoints_pose_coords[0,:].unsqueeze(0), pred_waypoints_pose_coords.squeeze(0)), dim=0 ) # 10 x 2
                    # And assume the initial waypoint is covered
                    waypoints_cov = torch.cat( (torch.tensor([1]).to(self.device), waypoints_cov.squeeze(0)), dim=0 ) # 10

                    ltg_dist = torch.linalg.norm(ltg_abs_coords.clone().float().cpu()-abs_pose_coords.float().cpu())*self.options.cell_size # distance to current long-term goal


                    # Estimate long term goal
                    if ((ltg_counter % self.options.steps_after_plan == 0) or  # either every k steps
                       (ltg_dist < 0.2)): # or we reached ltg

                        goal_confidence = pred_waypoints_vals[0,-1]

                        # if waypoint goal confidence is low then remove it from the waypoints list
                        if goal_confidence < self.options.goal_conf_thresh:
                            pred_waypoints_pose_coords[-1,0], pred_waypoints_pose_coords[-1,1] = -200, -200 

                        # Choose the waypoint following the one that is closest to the current location
                        pred_waypoint_dist = np.linalg.norm(pred_waypoints_pose_coords.cpu().numpy() - pose_coords.squeeze(0).cpu().numpy(), axis=-1)
                        min_point_ind = torch.argmin(torch.tensor(pred_waypoint_dist))
                        if min_point_ind >= pred_waypoints_pose_coords.shape[0]-1:
                            min_point_ind = pred_waypoints_pose_coords.shape[0]-2
                        if pred_waypoints_pose_coords[min_point_ind+1][0]==-200: # case when min_point_ind+1 is goal waypoint but it has low confidence
                            min_point_ind = min_point_ind-1
                        ltg = pred_waypoints_pose_coords[min_point_ind+1].unsqueeze(0).unsqueeze(0)

                        # To keep the same goal in multiple steps first transform ego ltg to abs global coords 
                        ltg_abs_coords = tutils.transform_ego_to_geo(ltg, pose_coords, abs_pose_coords, abs_poses, t)
                        ltg_abs_coords_list.append(ltg_abs_coords)

                        ltg_counter = 0 # reset the ltg counter
                    ltg_counter += 1


                    # transform ltg_abs_coords to current egocentric frame for visualization
                    ltg_sim_abs_pose = tutils.get_3d_pose(pose_2D=ltg_abs_coords.clone().squeeze(0), agent_pose_2D=abs_pose_coords.clone().squeeze(0), agent_sim_pose=agent_pose[:2], 
                                                                y_height=y_height, init_rot=torch.tensor(abs_poses[0][2]), cell_size=self.options.cell_size)
                    ltg_ego_coords, _ = tutils.transform_to_map_coords(sg=sg, position=ltg_sim_abs_pose, abs_pose=abs_poses[t], 
                                                                                grid_size=self.test_ds.grid_dim[0], cell_size=self.options.cell_size, device=self.device)


                    # Option to save visualizations of steps
                    if self.options.save_nav_images:
                        save_img_dir_ = self.options.log_dir + "/" + self.scene_id + "/" + self.options.save_img_dir+'ep_'+str(idx)+'/'
                        if not os.path.exists(save_img_dir_):
                            os.makedirs(save_img_dir_)
                        ### saves egocentric rgb, depth observations
                        viz_utils.display_sample(img.cpu().numpy(), np.squeeze(depth_abs.cpu().numpy()), t, savepath=save_img_dir_)
                        
                        ### visualize the predicted waypoints vs gt waypoints in gt semantic egocentric frame
                        viz_utils.show_waypoint_pred(pred_maps_objects.squeeze(0).squeeze(0), ltg=ltg_ego_coords.clone().cpu().numpy(), pose_coords=pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                                                            pred_waypoints=pred_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_pred.png")

                        viz_utils.show_waypoint_pred(gt_map_semantic, num_points=self.options.num_waypoints,
                                                            gt_waypoints=gt_waypoints_pose_coords, savepath=save_img_dir_+str(t)+"_waypoints_on_gt.png")

                        ### visualize the episode steps in the global geocentric frame
                        gt_map_semantic_global, _ = map_utils.get_gt_map(x, y, label_seq, abs_pose=abs_poses[0],
                                                                grid_dim=self.test_ds.global_dim, cell_size=self.options.cell_size, color_pcloud=color_pcloud, z=z)
                        # project gt waypoints in global geocentric frame
                        gt_waypoints_global = torch.zeros((len(gt_waypoints), 2))
                        for k in range(len(gt_waypoints)):
                            point_global_coords, _ = tutils.transform_to_map_coords(sg=sg_global, position=gt_waypoints[k], abs_pose=abs_poses[0], 
                                                                                    grid_size=self.test_ds.global_dim[0], cell_size=self.options.cell_size, device=self.device)
                            gt_waypoints_global[k,:] = point_global_coords.squeeze(0).squeeze(0)                                  
                        # transform predicted waypoints in global geocentric frame
                        pred_waypoints_global = torch.zeros((len(pred_waypoints_pose_coords), 2))
                        for k in range(len(pred_waypoints_pose_coords)):
                            pred_point_global_coords = tutils.transform_ego_to_geo(pred_waypoints_pose_coords[k].unsqueeze(0).unsqueeze(0), pose_coords, abs_pose_coords, abs_poses, t)
                            pred_waypoints_global[k,:] = pred_point_global_coords.squeeze(0)

                        viz_utils.show_waypoint_pred(gt_map_semantic_global, ltg=ltg_abs_coords.clone().cpu().numpy(), pose_coords=abs_pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                                                            pred_waypoints=pred_waypoints_global, gt_waypoints=gt_waypoints_global, savepath=save_img_dir_+"global_"+str(t)+"_waypoints.png")

                        # saves predicted areas (egocentric)
                        viz_utils.save_map_pred_steps(step_ego_grid_maps, pred_maps_spatial, 
                                                            pred_maps_objects, pred_ego_crops_sseg, save_img_dir_, t)

                        viz_utils.write_tensor_imgSegm(img_segm, save_img_dir_, name="img_segm", t=t, labels=27)

                    ##### Action decision process #####

                    pred_goal = pred_waypoints_pose_coords[-1].unsqueeze(0).unsqueeze(0)
                    pred_goal_dist = torch.linalg.norm(pred_goal.clone().float()-pose_coords.float())*self.options.cell_size # distance to current predicted goal

                    # Check stopping criteria
                    if tutils.decide_stop_vln(pred_goal_dist, self.options.stop_dist) or t==self.options.max_steps-1:
                        t+=1
                        break

                    # the goal is passed with respect to the abs_relative pose
                    action_id = self.run_local_policy(depth=depth, goal=ltg_abs_coords.clone(),
                                                                pose_coords=abs_pose_coords.clone(), rel_agent_o=rel_abs_pose[2], step=t)
                    # if stop is selected from local policy then randomly select an action
                    if action_id==0:
                        action_id = random.randint(1,3)

                    # explicitly clear observation otherwise they will be kept in memory the whole time
                    observations = None

                    # Apply next action
                    observations = self.test_ds.sim.step(action_id)

                    # estimate distance covered by agent
                    current_pos = self.test_ds.sim.get_agent_state().position
                    agent_episode_distance += utils.euclidean_distance(current_pos, previous_pos)
                    previous_pos = current_pos

                    t+=1


                sim_agent_poses = [x.tolist() for x in sim_agent_poses]

                geodesic_distance = self.test_ds.sim.geodesic_distance(episode["start_position"], goal_position)
                nav_metrics = tutils.get_metrics_vln(sim=self.test_ds.sim,
                                                    goal_position=goal_position,
                                                    success_distance=self.options.success_dist,
                                                    start_end_episode_distance=geodesic_distance,
                                                    agent_episode_distance=agent_episode_distance,
                                                    sim_agent_poses=sim_agent_poses,
                                                    stop_signal=True)
                for met in self.metrics:
                    self.results[met].append(nav_metrics[met])

                list_dist_to_goal.append(nav_metrics['distance_to_goal'])
                list_success.append(nav_metrics['success'])
                list_spl.append(nav_metrics['spl'])
                list_soft_spl.append(nav_metrics['softspl'])
                list_trajectory_length.append(nav_metrics['trajectory_length']) 
                list_navigation_error.append(nav_metrics['navigation_error']) 
                list_oracle_success.append(nav_metrics['oracle_success'])

                output = {}
                output['metrics'] = {'mean_dist_to_goal': np.mean(np.asarray(list_dist_to_goal.copy())),
                                     'mean_success': np.mean(np.asarray(list_success.copy())),
                                     'mean_spl': np.mean(np.asarray(list_spl.copy())),
                                     'mean_soft_spl': np.mean(np.asarray(list_soft_spl.copy())),
                                     'mean_trajectory_length': np.mean(np.asarray(list_trajectory_length.copy())),
                                     'mean_navigation_error': np.mean(np.asarray(list_navigation_error.copy())),
                                     'mean_oracle_success': np.mean(np.asarray(list_oracle_success.copy()))}

                # save a snapshot of the last step
                save_path = self.options.log_dir + "/" + self.scene_id + "/"
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                # write the instruction
                with open(save_path+str(idx)+'_instruction.txt', 'w') as outfile:
                    outfile.write(instruction)

                viz_utils.show_waypoint_pred(gt_map_semantic, ltg=ltg_ego_coords.clone().cpu().numpy(), pose_coords=pose_coords.clone().cpu().numpy(), num_points=self.options.num_waypoints,
                                                    pred_waypoints=pred_waypoints_pose_coords, gt_waypoints=gt_waypoints_pose_coords, savepath=save_path+str(idx)+"_waypoints.png")
                
                self.save_test_summaries(output)

                # write results to json at the end of each episode
                for met in self.metrics:
                    self.results["mean_"+met] = np.mean(np.asarray(self.results[met])) # per metric mean

                print(self.results)
                with open(self.options.log_dir+'/results_'+self.scene_id+'.json', 'w') as outfile:
                    json.dump(self.results, outfile, indent=4)

            ## Scene ended ##
            # Close current scene
            self.test_ds.sim.close()


    def run_goal_pred(self, instruction, sg, ego_occ_maps, ego_segm_maps, start_pos, pose):
        ## Prepare model inputs
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

        # Get the heatmap for the start position (initial waypoint)
        # This is with respect to the agent's current location
        point_pose_coords, visible = tutils.transform_to_map_coords(sg=sg, position=start_pos, abs_pose=pose, 
                                                                        grid_size=self.test_ds.grid_dim[0], cell_size=self.options.cell_size, device=self.device)
        start_pos_heatmap = utils.locs_to_heatmaps(keypoints=point_pose_coords.squeeze(0), img_size=self.test_ds.grid_dim, out_size=self.test_ds.heatmap_size)

        input_batch = {'step_ego_grid_maps':ego_occ_maps,
                       'ego_segm_maps':ego_segm_maps,
                       'goal_heatmap':start_pos_heatmap.unsqueeze(0).unsqueeze(0), # using the default name in the vln_goal_predictor
                       'tokens_tensor':tokens_tensor.unsqueeze(0),
                       'segments_tensors':segments_tensors.unsqueeze(0)}

        pred_output = self.models_dict['goal_pred_model'](input_batch)

        pred_waypoints_heatmaps = pred_output['pred_waypoints']
        waypoints_cov_prob = pred_output['waypoints_cov']

        # Convert predicted heatmaps to map coordinates
        pred_waypoints_resized = F.interpolate(pred_waypoints_heatmaps, size=(self.test_ds.grid_dim[0], self.test_ds.grid_dim[1]), mode='nearest')
        pred_locs, pred_vals = utils.heatmaps_to_locs(pred_waypoints_resized)
        # get the predicted coverage
        waypoints_cov = torch.argmax(waypoints_cov_prob, dim=2)

        pred_maps_objects = pred_output['pred_maps_objects']
        pred_maps_spatial = pred_output['pred_maps_spatial']

        return pred_locs, pred_vals, waypoints_cov, pred_maps_spatial, pred_maps_objects


    def run_local_policy(self, depth, goal, pose_coords, rel_agent_o, step):
        planning_goal = goal.squeeze(0).squeeze(0)
        planning_pose = pose_coords.squeeze(0).squeeze(0)
        sq = torch.square(planning_goal[0]-planning_pose[0])+torch.square(planning_goal[1]-planning_pose[1])
        rho = torch.sqrt(sq.float())
        phi = torch.atan2((planning_pose[0]-planning_goal[0]),(planning_pose[1]-planning_goal[1]))
        phi = phi - rel_agent_o
        rho = rho*self.test_ds.cell_size
        point_goal_with_gps_compass = torch.tensor([rho,phi], dtype=torch.float32).to(self.device)
        depth = depth.reshape(self.test_ds.img_size[0], self.test_ds.img_size[1], 1)
        return self.l_policy.plan(depth, point_goal_with_gps_compass, step)


    def save_test_summaries(self, output):
        prefix = 'test/' + self.scene_id + '/'
        for k in output['metrics']:
            self.summary_writer.add_scalar(prefix + k, output['metrics'][k], self.step_count)

