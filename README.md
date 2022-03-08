## Cross-modal Map Learning for Vision and Language Navigation

G.Georgakis, K.Schmeckpeper, K.Wanchoo, S.Dan, E.Miltsakaki, D.Roth, K.Daniilidis

IEEE International Conference on Computer Vision and Pattern Recognition 2022


### Dependencies
```
pip install -r requirements.txt
```
[Habitat-lab](https://github.com/facebookresearch/habitat-lab) and [habitat-sim](https://github.com/facebookresearch/habitat-sim) need to be installed before using our code. We build our method on the latest stable versions for both, so use `git checkout tags/v0.1.7` before installation. Follow the instructions in their corresponding repositories to install them on your system. Note that our code expects that habitat-sim is installed with the flag `--with-cuda`. 


### Trained Models
We provide our trained models for reproducing the navigation results shown in the paper [here](https://drive.google.com/drive/folders/12YLmOFJVtBJdFPioNBA-85Jj7zf3TQIX?usp=sharing).
In addition we provide the semantic segmentation model [here](https://drive.google.com/drive/folders/15wV84nfKnRVRSXGU_z04Hz5p_-je7YMW?usp=sharing). The DD-PPO model (gibson-4plus-mp3d-train-val-test-resnet50.pth) we used for the controller can be found [here](https://github.com/facebookresearch/habitat-lab/tree/main/habitat_baselines/rl/ddppo).

### Data
We use the Vision and Language Navigation in Continuous Environments (VLN-CE) dataset. Episodes can be found [here](https://github.com/jacobkrantz/VLN-CE). VLN-CE is based on the
Matterport3D (MP3D) dataset (the habitat subset and not the entire Matterport3D). Follow the instructions in the habitat-lab repository regarding downloading the data and the dataset folder structure. In addition we provide the following:
- [MP3D Scene Pclouds](https://drive.google.com/file/d/1u4SKEYs4L5RnyXrIX-faXGU1jc16CTkJ/view): An .npz file for each scene that we generated and that contains the 3D point cloud with semantic category labels (40 MP3D categories). This was done for our convenience because the semantic.ply files for each scene provided with the dataset contain instance labels. The folder containing the .npz files should be under `/data/scene_datasets/mp3d`.


### Instructions
Here we provide instructions on how to use our code. All options can be found in `train_options.py`. The episodes from VLN-CE should be under the `--root_path`. The DD-PPO model should be placed under `root_path/local_policy_models`

#### Testing on VLN-CE
To run an evaluation of CM2-GT on a single scene from val-seen: 
```
python main.py --name test_cm2-gt_val-seen --root_path /path/to/habitat-lab/folder/ --scenes_dir /habitat-lab/data/scene_datasets/ --model_exp_dir /path/to/cm2-gt/model/folder/ --log_dir logs/ --scenes_list 1pXnuDYAj8r --gpu_capacity 1 --split val_seen --use_first_waypoint --vln
```

To run an evaluation of CM2 on a single scene from val-seen: 
```
python main.py --name test_cm2_val-seen --root_path /path/to/habitat-lab/folder/ --scenes_dir /habitat-lab/data/scene_datasets/ --model_exp_dir /path/to/cm2/model/folder/ --log_dir logs/ --img_segm_model_dir /path/to/img/segm/model/folder/ --scenes_list 1pXnuDYAj8r --gpu_capacity 1 --split val_seen --use_first_waypoint --goal_conf_thresh 0.2 --vln_no_map
```

To enable visualizations during testing use `--save_nav_images`.


#### Generating training data
To generate the data for a single scene from train split to train the CM2-GT model:
```
python store_episodes_vln.py --root_path /path/to/habitat-lab/folder/ --scenes_dir /habitat-lab/data/scene_datasets/ --episodes_save_dir /path/to/cm2-gt/episodes/save/dir/ --scenes_list 1pXnuDYAj8r --gpu_capacity 1
```

To generate the data for a single scene from train split to train the CM2 model:
```
python store_episodes_vln_no_map.py --root_path /path/to/habitat-lab/folder/ --scenes_dir /habitat-lab/data/scene_datasets/ --episodes_save_dir /path/to/cm2/episodes/save/dir/ --scenes_list 1pXnuDYAj8r --gpu_capacity 1 --img_segm_model_dir /path/to/img/segm/model/folder/
```


#### Training
To train a new CM2-GT model:
```
python main.py --name train_cm2-gt --stored_episodes_dir /path/to/cm2-gt/episodes/save/dir/ --log_dir logs/ --is_train --summary_steps 500 --image_summary_steps 1000 --test_steps 20000 --checkpoint_steps 50000 --pad_text_feat --batch_size 40 --vln --finetune_bert_last_layer --use_first_waypoint --sample_1
```

To train a new CM2 model:
```
python main.py --name train_cm2 --stored_episodes_dir /path/to/cm2/episodes/save/dir/ --log_dir logs/ --is_train --summary_steps 500 --image_summary_steps 1000 --test_steps 20000 --checkpoint_steps 50000 --pad_text_feat --batch_size 10 --finetune_bert_last_layer --use_first_waypoint --vln_no_map --sample_1
```


