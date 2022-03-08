from pytorch_utils.base_options import BaseOptions
import argparse

class TrainOptions(BaseOptions):
    """ Parses command line arguments for training
    This overwrites options from BaseOptions
    """
    def __init__(self): # pylint: disable=super-init-not-called
        self.parser = argparse.ArgumentParser()

        req = self.parser.add_argument_group('Required')
        req.add_argument('--name', required=True, help='Name of the experiment')

        gen = self.parser.add_argument_group('General')
        gen.add_argument('--time_to_run', type=int, default=3600000,
                         help='Total time to run in seconds')
        gen.add_argument('--resume', dest='resume', default=False,
                         action='store_true',
                         help='Resume from checkpoint (Use latest checkpoint by default')
        gen.add_argument('--num_workers', type=int, default=0,
                         help='Number of processes used for data loading')
        pin = gen.add_mutually_exclusive_group()
        pin.add_argument('--pin_memory', dest='pin_memory', action='store_true',
                         help='Number of processes used for data loading')
        pin.add_argument('--no_pin_memory', dest='pin_memory', action='store_false',
                         help='Number of processes used for data loading')
        gen.set_defaults(pin_memory=True)

        in_out = self.parser.add_argument_group('io')
        in_out.add_argument('--log_dir', default='logs/', help='Directory to store logs')
        in_out.add_argument('--checkpoint', default=None, help='Path to checkpoint')
        in_out.add_argument('--from_json', default=None,
                            help='Load options from json file instead of the command line')

        train = self.parser.add_argument_group('Training Options')
        train.add_argument('--num_epochs', type=int, default=1000,
                           help='Total number of training epochs')
        train.add_argument('--batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_batch_size', type=int, default=1, help='Batch size')
        train.add_argument('--test_nav_batch_size', type=int, default=1, help='Batch size during navigation test')
        shuffle_train = train.add_mutually_exclusive_group()
        shuffle_train.add_argument('--shuffle_train', dest='shuffle_train', action='store_true',
                                   help='Shuffle training data')
        shuffle_train.add_argument('--no_shuffle_train', dest='shuffle_train', action='store_false',
                                   help='Don\'t shuffle training data')
        shuffle_test = train.add_mutually_exclusive_group()
        shuffle_test.add_argument('--shuffle_test', dest='shuffle_test', action='store_true',
                                  help='Shuffle testing data')
        shuffle_test.add_argument('--no_shuffle_test', dest='shuffle_test', action='store_false',
                                  help='Don\'t shuffle testing data')

        train.add_argument('--dataset_percentage', dest='dataset_percentage', type=float, default=1.0,
                            help='percentage of dataset to be used during training for ensemble learning')

        train.add_argument('--summary_steps', type=int, default=200,
                           help='Summary saving frequency')
        train.add_argument('--image_summary_steps', type=int, default=500,
                           help='Image summary saving frequency')
        train.add_argument('--checkpoint_steps', type=int, default=10000,
                           help='Chekpoint saving frequency')
        train.add_argument('--test_steps', type=int, default=10000, help='Testing frequency')


        train.add_argument('--is_train', dest='is_train', action='store_true',
                            help='Define whether training or testing mode')


        self.parser.add_argument('--config_file', type=str, dest='config_file',
                                default='configs/habitat_config.yaml')

        
        self.parser.add_argument('--model_exp_dir', type=str, dest='model_exp_dir', default=None,
                                help='Path for experiment containing the model used for testing')                        

        self.parser.add_argument('--n_spatial_classes', type=int, default=3, dest='n_spatial_classes',
                                help='number of categories for spatial prediction')
        self.parser.add_argument('--n_object_classes', type=int, default=27, dest='n_object_classes',
                                choices=[18,27], help='number of categories for object prediction')

        self.parser.add_argument('--global_dim', type=int, dest='global_dim', default=512)
        self.parser.add_argument('--grid_dim', type=int, default=192, dest='grid_dim',
                                    help='Semantic grid size (grid_dim, grid_dim)')
        self.parser.add_argument('--cell_size', type=float, default=0.05, dest="cell_size",
                                    help='Physical dimensions (meters) of each cell in the grid')
        self.parser.add_argument('--crop_size', type=int, default=64, dest='crop_size',
                                    help='Size of crop around the agent')

        self.parser.add_argument('--img_size', dest='img_size', type=int, default=256)
        self.parser.add_argument('--img_segm_size', dest='img_segm_size', type=int, default=128)


        train.add_argument('--map_loss_scale', type=float, default=1.0, dest='map_loss_scale')
        train.add_argument('--img_segm_loss_scale', type=float, default=1.0, dest='img_segm_loss_scale')


        train.add_argument('--init_gaussian_weights', dest='init_gaussian_weights', default=False, action='store_true',
                            help='initializes the model weights from gaussian distribution')


        train.set_defaults(shuffle_train=True, shuffle_test=True)

        optim = self.parser.add_argument_group('Optim')
        optim.add_argument("--lr_decay", type=float,
                           default=0.99, help="Exponential decay rate")
        optim.add_argument("--wd", type=float,
                           default=0, help="Weight decay weight")

        self.parser.add_argument('--test_iters', type=int, default=20000)

        optimizer_options = self.parser.add_argument_group('Optimizer')
        optimizer_options.add_argument('--lr', type=float, default=0.0002)
        optimizer_options.add_argument('--beta1', type=float, default=0.5)

        self.parser.add_argument('--img_segm_model_dir', dest='img_segm_model_dir', default="../",
                                    help='job path that contains the pre-trained img segmentation model')


        self.parser.add_argument('--max_steps', type=int, dest='max_steps', default=500,
                                  help='Maximum steps for each test episode')

        self.parser.add_argument('--steps_after_plan', type=int, dest='steps_after_plan', default=1,
                                 help='how many times to use the local policy before selecting long-term-goal and replanning')

        self.parser.add_argument('--stop_dist', type=float, dest='stop_dist', default=0.5,
                                
                                 help='decision to stop distance')
        self.parser.add_argument('--goal_conf_thresh', dest='goal_conf_thresh', type=float, default=0.6,
                                help='Goal confidence threshold to decide whether goal is valid')

        self.parser.add_argument('--success_dist', type=float, dest='success_dist', default=3.0,
                                 help='Radius around the target considered successful')

        self.parser.add_argument('--turn_angle', dest='turn_angle', type=int, default=15,
                                help='angle to rotate left or right in degrees for habitat simulator')
        self.parser.add_argument('--forward_step_size', dest='forward_step_size', type=float, default=0.25,
                                help='distance to move forward in meters for habitat simulator')

        self.parser.add_argument('--save_nav_images', dest='save_nav_images', action='store_true',
                                 help='Keep track and store maps during navigation testing')


        self.parser.add_argument('--root_path', type=str, dest='root_path', default="../")

        self.parser.add_argument('--scenes_dir', type=str, dest='scenes_dir', default='../')

        self.parser.add_argument('--stored_episodes_dir', type=str, dest='stored_episodes_dir', default='../')

        self.parser.add_argument('--split', type=str, dest='split', default='train',
                                 choices=['train', 'val', 'val_seen', 'val_unseen', 'test'])        

        self.parser.add_argument('--local_policy_model', type=str, dest='local_policy_model', default='4plus',
                                choices=['2plus', '4plus'])

        self.parser.add_argument('--scenes_list', nargs='+')
        self.parser.add_argument('--gpu_capacity', type=int, dest='gpu_capacity', default=2)

        self.parser.add_argument('--occupancy_height_thresh', type=float, dest='occupancy_height_thresh', default=-1.0,
                                help='used when estimating occupancy from depth')

        self.parser.add_argument('--save_img_dir', dest='save_img_dir', type=str, default='test_examples/')

        self.parser.add_argument('--save_test_images', dest='save_test_images', default=False, action='store_true',
                                help='save plots for waypoints and attention during testing')


        self.parser.add_argument('--num_waypoints', dest='num_waypoints', type=int, default=10,
                                 help='Number of waypoints sampled for each trajectory. Affects both sampling and waypoint prediction model')
        self.parser.add_argument('--heatmap_size', dest='heatmap_size', type=int, default=24,
                                 help='Waypoint heatmap size, should match hourglass output size.')

        self.parser.add_argument('--position_loss_scale', type=float, default=1.0, dest='position_loss_scale')
        self.parser.add_argument('--cov_loss_scale', type=float, default=1.0, dest='cov_loss_scale')

        self.parser.add_argument('--loss_norm', dest='loss_norm', default=False, action='store_true',
                                 help='If enabled uses the default normalization for L2 loss, and not "sum" ')

        self.parser.add_argument('--eval_set', dest='eval_set', default=False, action='store_true',
                                help='during testing: whether to choose train or eval part of the set the model was trained on')


        self.parser.add_argument('--d_model', dest='d_model', type=int, default=128,
                                 help='Input embedding dimensionality')
        self.parser.add_argument('--d_ff', dest='d_ff', type=int, default=64,
                                 help='Dimensionality of last feedforward layer')
        self.parser.add_argument('--d_k', dest='d_k', type=int, default=16,
                                 help='Dimensionality of W_k and W_q matrices. This is d_model/n_heads.')
        self.parser.add_argument('--d_v', dest='d_v', type=int, default=16,
                                 help='Dimensionality of W_v matrix. This is d_model/n_heads.')
        self.parser.add_argument('--n_heads', dest='n_heads', type=int, default=8,
                                 help='Number of heads in multi-head attention')
        self.parser.add_argument('--n_layers', dest='n_layers', type=int, default=4,
                                 help='Number of layers in encoder and decoder')

        self.parser.add_argument('--pad_text_feat', dest='pad_text_feat', default=False, action='store_true',
                                 help='Pad text features with zeroes. For batch_size>1 this has to be true')
        

        self.parser.add_argument('--use_first_waypoint', dest='use_first_waypoint', default=False, action='store_true',
                                help='use the first waypoint as an input to the waypoint prediction model')

        self.parser.add_argument('--with_lstm', dest='with_lstm', default=False, action='store_true',
                                help='use lstm layers when predicting the waypoints')


        self.parser.add_argument('--vln', dest='vln', default=False, action='store_true',
                                help='Enables the VLN part of this project')
        self.parser.add_argument('--vln_no_map', dest='vln_no_map', default=False, action='store_true',
                                help='Enables the VLN part of this project were we assume no map is given')

        self.parser.add_argument('--num_poses_per_example', dest='num_poses_per_example', type=int, default=10,
                                help='when storing episodes for vln how many poses to use in the same episode')

        self.parser.add_argument('--datasets', nargs='+', default=['R2R_VLNCE_v1-2'],
                                help='Choose only episodes in vln offline training from the selected datasets')

        self.parser.add_argument('--finetune_bert_last_layer', dest='finetune_bert_last_layer', default=False, action='store_true',
                                help='Finetune only last layer of bert. If disabled, the entire bert is finetuned.')

        self.parser.add_argument('--without_attn_1', dest='without_attn_1', default=False, action='store_true',
                                help='If enabled, does not use the cross-modal attention for map prediction. --vln_no_map should be True')

        self.parser.add_argument('--sample_1', dest='sample_1', default=False, action='store_true',
                                help='During training, randomly sample a single example from each sequence. If False, it passes the entire sequence.')