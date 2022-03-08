import numpy as np
import os
import torch
from models.semantic_grid import SemanticGrid


def get_acc_proj_grid(ego_grid_sseg, pose, abs_pose, crop_size, cell_size):
    grid_dim = (ego_grid_sseg.shape[2], ego_grid_sseg.shape[3])
    # sg.sem_grid will hold the accumulated semantic map at the end of the episode (i.e. 1 map per episode)
    sg = SemanticGrid(1, grid_dim, crop_size[0], cell_size, spatial_labels=ego_grid_sseg.shape[1], object_labels=ego_grid_sseg.shape[1])
    # Transform the ground projected egocentric grids to geocentric using relative pose
    geo_grid_sseg = sg.spatialTransformer(grid=ego_grid_sseg, pose=pose, abs_pose=abs_pose)
    # step_geo_grid contains the map snapshot every time a new observation is added
    step_geo_grid_sseg = sg.update_proj_grid_bayes(geo_grid=geo_grid_sseg.unsqueeze(0))
    # transform the projected grid back to egocentric (step_ego_grid_sseg contains all preceding views at every timestep)
    step_ego_grid_sseg = sg.rotate_map(grid=step_geo_grid_sseg.squeeze(0), rel_pose=pose, abs_pose=abs_pose)
    return step_ego_grid_sseg


def est_occ_from_depth(local3D, grid_dim, cell_size, device, occupancy_height_thresh=-0.9):

    ego_grid_occ = torch.zeros((len(local3D), 3, grid_dim[0], grid_dim[1]), dtype=torch.float32, device=device)

    for k in range(len(local3D)):

        local3D_step = local3D[k]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        local3D_step = local3D_step[(z < 3) & (z > 0.5) & (y < 1), :]

        # initialize all locations as unknown (void)
        occ_lbl = torch.zeros((local3D_step.shape[0], 1), dtype=torch.float32, device=device)

        # threshold height to get occupancy and free labels
        thresh = occupancy_height_thresh
        y = local3D_step[:,1]
        occ_lbl[y>thresh,:] = 1
        occ_lbl[y<=thresh,:] = 2

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)
        map_coords = map_coords.to(device)

        ## Replicate label pooling
        grid = torch.empty(3, grid_dim[0], grid_dim[1], device=device)
        grid[:] = 1 / 3

        # If the robot does not project any values on the grid, then return the empty grid
        if map_coords.shape[0]==0:
            ego_grid_occ[k,:,:,:] = grid.unsqueeze(0)
            continue

        concatenated = torch.cat([map_coords, occ_lbl.long()], dim=-1)
        unique_values, counts = torch.unique(concatenated, dim=0, return_counts=True)
        grid[unique_values[:, 2], unique_values[:, 1], unique_values[:, 0]] = counts + 1e-5

        ego_grid_occ[k,:,:,:] = grid / grid.sum(dim=0)

    return ego_grid_occ



def ground_projection(points2D, local3D, sseg, sseg_labels, grid_dim, cell_size):
    ego_grid_sseg = torch.zeros((sseg.shape[0], sseg_labels, grid_dim[0], grid_dim[1]), dtype=torch.float32, device='cuda')

    for i in range(sseg.shape[0]): # sequence length
        sseg_step = sseg[i,:,:,:].unsqueeze(0) # 1 x 1 x H x W
        points2D_step = points2D[i]
        local3D_step = local3D[i]

        # Keep points for which z < 3m (to ensure reliable projection)
        # and points for which z > 0.5m (to avoid having artifacts right in-front of the robot)
        z = -local3D_step[:,2]
        valid_inds = torch.nonzero(torch.where((z<3) & (z>0.5), 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]
        # avoid adding points from the ceiling, threshold on y axis, y range is roughly [-1...2.5]
        y = local3D_step[:,1]
        valid_inds = torch.nonzero(torch.where(y<1, 1, 0)).squeeze(dim=1)
        local3D_step = local3D_step[valid_inds,:]
        points2D_step = points2D_step[valid_inds,:]

        map_coords = discretize_coords(x=local3D_step[:,0], z=local3D_step[:,2], grid_dim=grid_dim, cell_size=cell_size)

        grid_sseg = label_pooling(sseg_step, points2D_step, map_coords, sseg_labels, grid_dim)
        grid_sseg = grid_sseg.unsqueeze(0)

        ego_grid_sseg[i,:,:,:] = grid_sseg

    return ego_grid_sseg


def label_pooling(sseg, points2D, map_coords, sseg_labels, grid_dim):
    # pool the semantic labels
    # For each bin get the frequencies of the class labels based on the labels projected
    # Each grid location will hold a probability distribution over the semantic labels
    grid = torch.ones((sseg_labels, grid_dim[0], grid_dim[1]), device='cuda')*(1/sseg_labels) # initially uniform distribution over the labels

    # If the robot does not project any values on the grid, then return the empty grid
    if map_coords.shape[0]==0:
        return grid
    pix_x, pix_y = points2D[:,0].long(), points2D[:,1].long()
    pix_lbl = sseg[0, 0, pix_y, pix_x]
    # SPEEDUP if map_coords is sorted, can switch to unique_consecutive
    uniq_rows = torch.unique(map_coords, dim=0)
    for i in range(uniq_rows.shape[0]):
        ucoord = uniq_rows[i,:]
        # indices of where ucoord can be found in map_coords
        ind = torch.nonzero(torch.where((map_coords==ucoord).all(axis=1), 1, 0)).squeeze(dim=1)
        bin_lbls = pix_lbl[ind]
        hist = torch.histc(bin_lbls, bins=sseg_labels, min=0, max=sseg_labels)
        hist = hist + 1e-5 # add a very small number to every location to avoid having 0s
        hist = hist / float(bin_lbls.shape[0])
        grid[:, ucoord[1], ucoord[0]] = hist
    return grid



def discretize_coords(x, z, grid_dim, cell_size, translation=0):
    # x, z are the coordinates of the 3D point (either in camera coordinate frame, or the ground-truth camera position)
    # If translation=0, assumes the agent is at the center
    # If we want the agent to be positioned lower then use positive translation. When getting the gt_crop, we need negative translation
    #map_coords = torch.zeros((len(x), 2), device='cuda')
    map_coords = torch.zeros((len(x), 2))
    xb = torch.floor(x[:]/cell_size) + (grid_dim[0]-1)/2.0
    zb = torch.floor(z[:]/cell_size) + (grid_dim[1]-1)/2.0 + translation
    xb = xb.int()
    zb = zb.int()
    map_coords[:,0] = xb
    map_coords[:,1] = zb
    # keep bin coords within dimensions
    map_coords[map_coords>grid_dim[0]-1] = grid_dim[0]-1
    map_coords[map_coords<0] = 0
    return map_coords.long()



def get_gt_crops(abs_pose, pcloud, label_seq_all, agent_height, grid_dim, crop_size, cell_size):
    x_all, y_all, z_all = pcloud[0], pcloud[1], pcloud[2]
    episode_extend = abs_pose.shape[0]
    gt_grid_crops = torch.zeros((episode_extend, 1, crop_size[0], crop_size[1]), dtype=torch.int64)
    for k in range(episode_extend):
        # slice the gt map according to the agent height at every step
        x, y, label_seq = slice_scene(x_all.copy(), y_all.copy(), z_all.copy(), label_seq_all.copy(), agent_height[k])
        gt = get_gt_map(x, y, label_seq, abs_pose=abs_pose[k], grid_dim=grid_dim, cell_size=cell_size)
        _gt_crop = crop_grid(grid=gt.unsqueeze(0), crop_size=crop_size)
        gt_grid_crops[k,:,:,:] = _gt_crop.squeeze(0)
    return gt_grid_crops


def get_gt_map(x, y, label_seq, abs_pose, grid_dim, cell_size, color_pcloud=None, z=None):
    # Transform the ground-truth map to align with the agent's pose
    # The agent is at the center looking upwards
    point_map = np.array([x,y])
    angle = -abs_pose[2]
    rot_mat_abs = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    trans_mat_abs = np.array([[-abs_pose[1]],[abs_pose[0]]]) #### This is important, the first index is negative.
    ##rotating and translating point map points
    t_points = point_map - trans_mat_abs
    rot_points = np.matmul(rot_mat_abs,t_points)
    x_abs = torch.tensor(rot_points[0,:], device='cuda')
    y_abs = torch.tensor(rot_points[1,:], device='cuda')

    map_coords = discretize_coords(x=x_abs, z=y_abs, grid_dim=grid_dim, cell_size=cell_size)

    # Coordinates in map_coords need to be sorted based on their height, floor values go first
    # Still not perfect
    if z is not None:
        z = np.asarray(z)
        sort_inds = np.argsort(z)
        map_coords = map_coords[sort_inds,:]
        label_seq = label_seq[sort_inds,:]

    true_seg_grid = torch.zeros((grid_dim[0], grid_dim[1], 1), device='cuda')
    true_seg_grid[map_coords[:,1], map_coords[:,0]] = label_seq.clone()

    ### We need to flip the ground truth to align with the observations.
    ### Probably because the -y tp -z is a rotation about x axis which also flips the y coordinate for matteport.
    true_seg_grid = torch.flip(true_seg_grid, dims=[0])
    true_seg_grid = true_seg_grid.permute(2, 0, 1)

    if color_pcloud is not None:
        color_grid = torch.zeros((grid_dim[0], grid_dim[1], 3), device='cuda')
        color_grid[map_coords[:,1], map_coords[:,0],0] = color_pcloud[0]
        color_grid[map_coords[:,1], map_coords[:,0],1] = color_pcloud[1]
        color_grid[map_coords[:,1], map_coords[:,0],2] = color_pcloud[2]
        color_grid = torch.flip(color_grid, dims=[0])
        color_grid = color_grid.permute(2, 0 ,1)
        return true_seg_grid, color_grid/255.0
    else:
        return true_seg_grid


def crop_grid(grid, crop_size):
    # Assume input grid is already transformed such that agent is at the center looking upwards
    grid_dim_h, grid_dim_w = grid.shape[2], grid.shape[3]
    cx, cy = int(grid_dim_w/2.0), int(grid_dim_h/2.0)
    rx, ry = int(crop_size[0]/2.0), int(crop_size[1]/2.0)
    top, bottom, left, right = cx-rx, cx+rx, cy-ry, cy+ry
    return grid[:, :, top:bottom, left:right]

def slice_scene(x, y, z, label_seq, height, color_pcloud=None):
    # z = -z
    # Slice the scene below and above the agent
    below_thresh = height-0.2
    above_thresh = height+2.0
    all_inds = np.arange(y.shape[0])
    below_inds = np.where(z<below_thresh)[0]
    above_inds = np.where(z>above_thresh)[0]
    invalid_inds = np.concatenate( (below_inds, above_inds), 0) # remove the floor and ceiling inds from the local3D points
    inds = np.delete(all_inds, invalid_inds)
    x_fil = x[inds]
    y_fil = y[inds]
    z_fil = z[inds]
    label_seq_fil = torch.tensor(label_seq[inds], dtype=torch.float, device='cuda')
    if color_pcloud is not None:
        color_pcloud_fil = torch.tensor(color_pcloud[:,inds], dtype=torch.float, device='cuda')
        return x_fil, y_fil, z_fil, label_seq_fil, color_pcloud_fil
    else:
        return x_fil, y_fil, z_fil, label_seq_fil


def get_explored_grid(grid_sseg, thresh=0.5):
    # Use the ground-projected ego grid to get observed/unobserved grid
    # Single channel binary value indicating cell is observed
    # Input grid_sseg T x C x H x W (can be either H x W or cH x cW)
    # Returns T x 1 x H x W
    T, C, H, W = grid_sseg.shape
    grid_explored = torch.ones((T, 1, H, W), dtype=torch.float32).to(grid_sseg.device)
    grid_prob_max = torch.amax(grid_sseg, dim=1)
    inds = torch.nonzero(torch.where(grid_prob_max<=thresh, 1, 0))
    grid_explored[inds[:,0], 0, inds[:,1], inds[:,2]] = 0
    return grid_explored

