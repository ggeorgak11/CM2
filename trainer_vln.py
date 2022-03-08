import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from pytorch_utils.base_trainer import BaseTrainer
from datasets.dataloader import HabitatDataVLNOffline
from models import get_model_from_options
import datasets.util.utils as utils
import datasets.util.viz_utils as viz_utils
import datasets.util.map_utils as map_utils
import matplotlib.pyplot as plt


class TrainerVLN(BaseTrainer):
    """ Implements training for prediction models
    """
    def init_fn(self):
        print("options:")
        for k in self.options.__dict__.keys():
            print(k, self.options.__dict__[k])

        self.train_ds = HabitatDataVLNOffline(self.options, eval_set=False)
        self.test_ds = HabitatDataVLNOffline(self.options, eval_set=True)

        print('Train Data:', len(self.train_ds))
        print('Test Data:', len(self.test_ds))
        
        self.goal_pred_model = get_model_from_options(self.options)

        # Init the weights from normal distr with mean=0, std=0.02
        if self.options.init_gaussian_weights:
            self.goal_pred_model.apply(self.weights_init)

        self.models_dict = {'goal_pred_model':self.goal_pred_model}

        print("Using ", torch.cuda.device_count(), "gpus")
        for k in self.models_dict:
            self.models_dict[k] = nn.DataParallel(self.models_dict[k])

        self.optimizers_dict = {}
        for model in self.models_dict:
            self.optimizers_dict[model] = \
                    torch.optim.Adam([{'params':self.models_dict[model].parameters(),
                                    'initial_lr':self.options.lr}],
                                    lr=self.options.lr,
                                    betas=(self.options.beta1, 0.999) )


    # custom weights initialization
    def weights_init(self, m):
        classname = m.__class__.__name__
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)


    def train_step(self, input_batch, step_count):
        for model in self.models_dict:
            self.models_dict[model].train()
        for optimizer in self.optimizers_dict:
            self.optimizers_dict[optimizer].zero_grad()

        pred_waypoints, waypoints_cov_raw, waypoints_cov, _ = self.models_dict['goal_pred_model'](input_batch)

        loss_position = self.models_dict['goal_pred_model'].module.position_loss(pred_waypoints, input_batch)
        loss_cov = self.models_dict['goal_pred_model'].module.coverage_loss(waypoints_cov_raw, input_batch)

        pred_loss = loss_position['position_loss'] + loss_cov['cov_loss']
        pred_loss.sum().backward(retain_graph=True)
        self.optimizers_dict['goal_pred_model'].step()

        output={}
        output['preds'] = {'pred_waypoints':pred_waypoints.detach(),
                            'cov_waypoints':waypoints_cov.detach()
                            }
        output['metrics'] = {'pred_err_waypoints':loss_position['position_error'],
                             'cov_error':loss_cov['cov_error']
                             }
        output['losses'] = {'pred_loss_waypoints':loss_position['position_loss'],
                             'cov_loss':loss_cov['cov_loss']
                             }
        for k in output['metrics']:
            output['metrics'][k] = torch.mean(output['metrics'][k])
        for k in output['losses']:
            output['losses'][k] = torch.mean(output['losses'][k])
        return [output]


    def train_summaries(self, input_batch, save_images, model_output):
        self._save_summaries(input_batch, model_output, save_images, is_train=True)

    
    def test(self):
        for model in self.models_dict:
            self.models_dict[model].eval()

        test_data_loader = DataLoader(self.test_ds,
                                      batch_size=self.options.test_batch_size,
                                      num_workers=self.options.num_workers,
                                      pin_memory=self.options.pin_memory,
                                      shuffle=self.options.shuffle_test)
        batch = None
        self.options.test_iters = len(test_data_loader) # the length of dataloader depends on the batch size
        pck_list, pos_err_list, cov_err_list = [], [], []
        for tstep, batch in enumerate(tqdm(test_data_loader,
                                           desc='Testing',
                                           total=self.options.test_iters)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            with torch.no_grad():
                #print(batch.keys())

                pred_waypoints, waypoints_cov_raw, waypoints_cov, _ = self.models_dict['goal_pred_model'](batch)
                loss_position = self.models_dict['goal_pred_model'].module.position_loss(pred_waypoints, batch)
                loss_cov = self.models_dict['goal_pred_model'].module.coverage_loss(waypoints_cov_raw, batch)

                pos_err_list.append(loss_position['position_error'].item())
                cov_err_list.append(loss_cov['cov_error'].item())

                
                gt_waypoints = batch['goal_heatmap']
                visible_waypoints = batch['visible_waypoints']
                
                B, T, num_waypoints, cH, cW = gt_waypoints.shape
                gt_waypoints = gt_waypoints.view(B*T, num_waypoints, cH, cW)
                visible_waypoints = visible_waypoints.view(B*T, num_waypoints)

                if self.options.use_first_waypoint:
                    gt_waypoints = gt_waypoints[:,1:,:,:]
                    visible_waypoints = visible_waypoints[:,1:]

                # Return pck of predictions
                res_pck = utils.pck(gt_waypoints, pred_waypoints, visible=visible_waypoints.long())
                pck_list.append(res_pck)

                # Stop testing if test iterations has been exceeded
                #if tstep > self.options.test_iters:
                #    break
                
        pck_mean = torch.mean(torch.tensor(pck_list))
        pos_err_mean = torch.mean(torch.tensor(pos_err_list))
        cov_err_mean = torch.mean(torch.tensor(cov_err_list))

        output = {}
        output['metrics'] = {'pck':pck_mean,
                             'position_error':pos_err_mean,
                             'cov_error':cov_err_mean
                             }
        output['preds'] = {'pred_waypoints':pred_waypoints.detach(),
                           'cov_waypoints':waypoints_cov.detach()
                           }
        for k in output['metrics']:
            output['metrics'][k] = torch.mean(output['metrics'][k])

        self._save_summaries(batch, output, save_images=True, is_train=False)


    def _save_summaries(self, batch, output, save_images, is_train=False):
        prefix = 'train/' if is_train else 'test/'

        if save_images:
            # Get waypoints in original resolution for visualization
            # Assume sequence T=1

            B, T, num_waypoints, cH, cW = batch['goal_heatmap'].shape
            gt_waypoints = batch['goal_heatmap'].view(B*T, num_waypoints, cH, cW)
            map_semantic = batch['map_semantic'].view(B*T, 1, batch['map_semantic'].shape[3], batch['map_semantic'].shape[4])
            visible_waypoints = batch['visible_waypoints'].view(B*T, num_waypoints)

            if self.options.use_first_waypoint:
                pred_waypoints = torch.cat((gt_waypoints[:,0,:,:].unsqueeze(1), output['preds']['pred_waypoints']), dim=1)
            else:
                pred_waypoints = output['preds']['pred_waypoints'] # B*T x num_waypoints x 64 x 64
            
            fig_list = []
            for i in range(map_semantic.shape[0]): # batch*T
                goal = gt_waypoints[i,:,:,:] # 1 x 64 x 64
                map_s = map_semantic[i,:,:,:] # 1 x 512 x 512
                pred_goal = pred_waypoints[i,:,:,:] # 1 x 64 x 64
                visible = visible_waypoints[i,:]

                gt_heatmaps_resized = F.interpolate(goal.unsqueeze(0), size=(map_s.shape[1], map_s.shape[2]), mode='nearest')
                pred_waypoints_resized = F.interpolate(pred_goal.unsqueeze(0), size=(map_s.shape[1], map_s.shape[2]), mode='nearest')

                pred_locs, pred_vals = utils.heatmaps_to_locs(pred_waypoints_resized)
                gt_locs, gt_vals = utils.heatmaps_to_locs(gt_heatmaps_resized)

                color_sem_map = viz_utils.colorize_grid(map_s.unsqueeze(0).unsqueeze(0), color_mapping=27)
                color_sem_map = color_sem_map.squeeze(1)
                
                fig = plt.figure(figsize=(8 ,8))
                img_data = color_sem_map[0,:,:,:].permute(1,2,0)
                plt.imshow(img_data)
                for k in range(gt_locs.shape[1]):
                    if visible[k]:
                        plt.scatter(gt_locs[0,k,0], gt_locs[0,k,1], color="blue", s=20)
                        plt.scatter(pred_locs[0,k,0], pred_locs[0,k,1], color="red", s=20)
                        plt.text(gt_locs[0,k,0], gt_locs[0,k,1], s=str(k), color="blue")
                        plt.text(pred_locs[0,k,0], pred_locs[0,k,1], s=str(k), color="red")
                fig_list.append(fig)
                plt.close()
            self.summary_writer.add_figure(prefix+"grid/fig/pred_goals", fig_list, self.step_count)

        if is_train:
            types = ['losses', 'metrics']
        else:
            types = ['metrics']

        for scalar_type in types:
            for k in output[scalar_type]:
                self.summary_writer.add_scalar(prefix + k, output[scalar_type][k], self.step_count)

        if is_train:
            self.summary_writer.add_scalar(prefix + "lr", self.get_lr(), self.step_count)
