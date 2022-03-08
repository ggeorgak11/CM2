
import numpy as np
import quaternion
import datasets.util.utils as utils
import datasets.util.map_utils as map_utils
import torch
import os


def get_latest_model(save_dir):
    checkpoint_list = []
    for dirpath, _, filenames in os.walk(save_dir):
        for filename in filenames:
            if filename.endswith('.pt'):
                checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
    checkpoint_list = sorted(checkpoint_list)
    latest_checkpoint =  None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
    return latest_checkpoint


def load_model(models, checkpoint_file):
    # Load the latest checkpoint
    checkpoint = torch.load(checkpoint_file)
    for model in models:
        if model in checkpoint['models']:
            models[model].load_state_dict(checkpoint['models'][model])
        else:
            raise Exception("Missing model in checkpoint: {}".format(model))
    return models


def get_2d_pose(position, rotation=None):
    # position is 3-element list
    # rotation is 4-element list representing a quaternion
    position = np.asarray(position, dtype=np.float32)
    x = -position[2]
    y = -position[0]
    height = position[1]

    if rotation is not None:
        rotation = np.quaternion(rotation[0], rotation[1], rotation[2], rotation[3])
        axis = quaternion.as_euler_angles(rotation)[0]
        if (axis%(2*np.pi)) < 0.1 or (axis%(2*np.pi)) > 2*np.pi - 0.1:
            o = quaternion.as_euler_angles(rotation)[1]
        else:
            o = 2*np.pi - quaternion.as_euler_angles(rotation)[1]
        if o > np.pi:
            o -= 2 * np.pi
        pose = x, y, o
    else:
        pose = x, y, 0.0
    return pose, height


def get_3d_pose(pose_2D, agent_pose_2D, agent_sim_pose, y_height, init_rot, cell_size):
    # Given a 2D grid (pose_2D) location, return its 3D abs pose in habitat sim coords
    init_rot = -init_rot
    dist_x = (pose_2D[0,0] - agent_pose_2D[0,0]) * cell_size
    dist_z = (pose_2D[0,1] - agent_pose_2D[0,1]) * cell_size
    init_rot_mat = torch.tensor([[torch.cos(init_rot), -torch.sin(init_rot)],[torch.sin(init_rot),torch.cos(init_rot)]], dtype=torch.float32)
    dist_vect = torch.tensor([dist_x,dist_z])
    dist_vect = dist_vect.reshape((2,1))
    rot_vect = torch.matmul(init_rot_mat,dist_vect)
    sim_pose_tmp_x, sim_pose_tmp_z = agent_sim_pose[0]-rot_vect[1], agent_sim_pose[1]-rot_vect[0]
    # revert changes from utils.get_sim_location
    sim_pose = np.zeros((3), dtype=np.float32)
    sim_pose[0] = -sim_pose_tmp_z
    sim_pose[1] = y_height
    sim_pose[2] = -sim_pose_tmp_x
    return sim_pose.tolist() 


def transform_to_map_coords(sg, position, abs_pose, grid_size, cell_size, device):
    pose, _ = get_2d_pose(position=position)
    agent_rel_pose = utils.get_rel_pose(pos2=pose, pos1=abs_pose)
    agent_rel_pose = torch.Tensor(agent_rel_pose).unsqueeze(0).float()
    agent_rel_pose = agent_rel_pose.to(device)
    _pose_coords = get_coord_pose(sg, agent_rel_pose, abs_pose, grid_size, cell_size, device) # B x T x 3

    visible_position = 1
    # if goal pose coords is 0,0 then goal is outside the current map. Use an empty heatmap
    if _pose_coords[0,0,0]==0 and _pose_coords[0,0,1]==0:
        _pose_coords = torch.tensor([[[-200,-200]]])
        visible_position = 0
    return _pose_coords, visible_position


def transform_ego_to_geo(ego_point, pose_coords, abs_pose_coords, abs_poses, t):
    # ego_point is point to transform
    # pose_coords is agent's ego centric pose (always in the center of the map)
    # abs_pose_coords is agent's pose with respect to first pose in the episode
    rel_rot = torch.tensor(abs_poses[0][2]) - torch.tensor(abs_poses[t][2])
    dist_x = (ego_point[0,0,0] - pose_coords[0,0,0])
    dist_z = (ego_point[0,0,1] - pose_coords[0,0,1])
    rel_rot_mat = torch.tensor([[torch.cos(rel_rot), -torch.sin(rel_rot)],[torch.sin(rel_rot),torch.cos(rel_rot)]], dtype=torch.float32)
    dist_vect = torch.tensor([dist_x,dist_z])
    dist_vect = dist_vect.reshape((2,1))
    rot_vect = torch.matmul(rel_rot_mat,dist_vect)

    abs_coords_x = abs_pose_coords[0,0,0] + rot_vect[0]
    abs_coords_z = abs_pose_coords[0,0,1] + rot_vect[1]
    abs_coords = torch.tensor([[[abs_coords_x, abs_coords_z]]])
    return abs_coords


def get_coord_pose(sg, rel_pose, init_pose, grid_dim, cell_size, device=None):
    # Create a grid where the starting location is always at the center looking upwards (like the ground-projected grids)
    # Then use the spatial transformer to move that location at the right place
    if isinstance(init_pose, list) or isinstance(init_pose, tuple):
        init_pose = torch.tensor(init_pose).unsqueeze(0)
    else:
        init_pose = init_pose.unsqueeze(0)

    zero_pose = torch.tensor([[0., 0., 0.]])
    if device!=None:
        init_pose = init_pose.to(device)
        zero_pose = zero_pose.to(device)

    zero_coords = map_utils.discretize_coords(x=zero_pose[:,0],
                                            z=zero_pose[:,1],
                                            grid_dim=(grid_dim, grid_dim),
                                            cell_size=cell_size)

    pose_grid = torch.zeros((1, 1, grid_dim, grid_dim), dtype=torch.float32)#.to(device)
    pose_grid[0,0,zero_coords[0,0], zero_coords[0,1]] = 1

    pose_grid_transf = sg.spatialTransformer(grid=pose_grid, pose=rel_pose, abs_pose=init_pose)
    
    pose_grid_transf = pose_grid_transf.squeeze(0).squeeze(0)
    inds = utils.unravel_index(pose_grid_transf.argmax(), pose_grid_transf.shape)

    pose_coord = torch.zeros((1, 1, 2), dtype=torch.int64)#.to(device)
    pose_coord[0,0,0] = inds[1] # inds is y,x
    pose_coord[0,0,1] = inds[0]
    return pose_coord



def decide_stop_vln(pred_goal_dist, stop_dist, ltg_cons=True):
    if pred_goal_dist <= stop_dist and ltg_cons:
        return True
    else:
        return False



# Return success, SPL, soft_SPL, distance_to_goal measures
def get_metrics_vln(sim,
                goal_position,
                success_distance,
                start_end_episode_distance,
                agent_episode_distance,
                sim_agent_poses,
                stop_signal):

    curr_pos = sim.get_agent_state().position
    # returns distance to the closest goal position
    distance_to_goal = sim.geodesic_distance(curr_pos, goal_position)

    if distance_to_goal <= success_distance and stop_signal:
        success = 1.0
    else:
        success = 0.0

    spl = success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    ep_soft_success = max(0, (1 - distance_to_goal / start_end_episode_distance) )
    soft_spl = ep_soft_success * (start_end_episode_distance / max(start_end_episode_distance, agent_episode_distance) )

    # Navigation error is the min(geodesic_distance(agent_pos, goal)) over all points in the agent path
    min_dist=99999
    for i in range(len(sim_agent_poses)):
        dist = sim.geodesic_distance(sim_agent_poses[i], goal_position)
        if dist < min_dist:
            min_dist = dist

    # If at any point in the path we were within success distance then oracle success=1
    if min_dist <= success_distance:
        oracle_success = 1.0
    else:
        oracle_success = 0.0

    metrics = {'distance_to_goal':distance_to_goal,
               'success':success,
               'spl':spl,
               'softspl':soft_spl,
               'trajectory_length':agent_episode_distance,
               'navigation_error':min_dist,
               'oracle_success': oracle_success}
    return metrics


