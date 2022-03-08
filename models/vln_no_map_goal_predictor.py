import torch
import torch.nn as nn
import torch.nn.functional as F


class VLNGoalPredictor_UnknownMap(nn.Module):
    def __init__(self, 
                 map_encoder, 
                 bert_model, 
                 attention_model_map_prediction, 
                 map_predictor, 
                 map_encoder_sem, 
                 attention_model_waypoints, 
                 goal_pred_model,
                 map_loss_scale,
                 pos_loss_scale, 
                 cov_loss_scale,
                 d_model, 
                 use_first_waypoint, 
                 loss_norm, 
                 text_init_dim=768):
        super(VLNGoalPredictor_UnknownMap, self).__init__()

        self.map_encoder = map_encoder
        self.bert_model = bert_model
        self.text_encoder = nn.Linear(in_features=text_init_dim, out_features=d_model) # to produce same size embeddings as the map
        self.attention_model_map_prediction = attention_model_map_prediction
        self.map_predictor = map_predictor
        self.map_encoder_sem = map_encoder_sem
        self.attention_model_waypoints = attention_model_waypoints
        self.goal_pred_model = goal_pred_model

        if loss_norm:
            reduction = "mean"
        else:
            reduction = "sum"

        self.position_mse_loss = nn.MSELoss(reduction=reduction)
        self.cel_loss_cov = nn.CrossEntropyLoss(reduction=reduction)
        self.cel_loss_spatial = nn.CrossEntropyLoss(reduction=reduction)
        self.cel_loss_objects = nn.CrossEntropyLoss(reduction=reduction)

        self.map_loss_scale = map_loss_scale
        self.pos_loss_scale = pos_loss_scale
        self.cov_loss_scale = cov_loss_scale
        self.d_model = d_model
        self.use_first_waypoint = use_first_waypoint

        if self.use_first_waypoint:
            self.conv_point1 = nn.Conv2d(d_model+1, d_model, 1, 1, padding=0)

    
    def forward(self, batch):

        # egocentric inputs
        step_ego_grid_maps = batch['step_ego_grid_maps']
        ego_segm_maps = batch['ego_segm_maps']
        B, T, spatial_C, H, W = step_ego_grid_maps.shape # batch, sequence, 3, height, width
        step_ego_grid_maps = step_ego_grid_maps.view(B*T, spatial_C, H, W)
        
        objects_C = ego_segm_maps.shape[2]
        ego_segm_maps = ego_segm_maps.view(B*T, objects_C, H, W)

        # bert inputs
        tokens_tensor = batch['tokens_tensor'].long()
        segments_tensors = batch['segments_tensors'].long()
        tokens_tensor = tokens_tensor.squeeze(1)
        segments_tensors = segments_tensors.squeeze(1)

        outputs = self.bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

        text_feat=[]
        for b in range(tokens_tensor.shape[0]):
            text_feat.append(hidden_states[-1][b][:])
        text_feat = torch.stack(text_feat)

        encoded_text = self.text_encoder(text_feat)

        # replicate encoded text for number of encoded_map_in
        if T>1:
            encoded_text = encoded_text.unsqueeze(1).repeat(1, T, 1, 1)
            encoded_text = encoded_text.view(B*T, encoded_text.shape[2], encoded_text.shape[3])

        if self.attention_model_map_prediction is not None: # case when we use the cross-modal attention for map prediction
            ### Prepare the input embeddings of the map and text
            encoded_map = self.map_encoder(step_ego_grid_maps) # B x 128 x 32 x 32
            map_enc_res = encoded_map.shape[2]

            encoded_map_in = encoded_map.permute(0,2,3,1).view(encoded_map.shape[0], -1, self.d_model)

            ### Apply attention between the map and text
            dec_out_map, dec_enc_attns_map = self.attention_model_map_prediction(enc_inputs=encoded_text, dec_inputs=encoded_map_in) # B x 1024 x 128

            ### Use the attention-augmented map encoding for map hallucination
            dec_out_map = dec_out_map.permute(0,2,1).view(dec_out_map.shape[0], -1, map_enc_res, map_enc_res) # reshape map attn # B x 128 x 32 x 32

            # Map predictor (hallucination)
            pred_maps_raw_spatial, pred_maps_raw_objects = self.map_predictor(input=step_ego_grid_maps, img_segm=ego_segm_maps, attn_dec_out=dec_out_map)
        else:
            pred_maps_raw_spatial, pred_maps_raw_objects = self.map_predictor(input=step_ego_grid_maps, img_segm=ego_segm_maps)

        # Second encoder + attention module for predicting the waypoints
        encoded_sem_map = self.map_encoder_sem(pred_maps_raw_objects)
        map_enc_res = encoded_sem_map.shape[2]
        encoded_sem_map_in = encoded_sem_map.permute(0,2,3,1).view(encoded_sem_map.shape[0], -1, self.d_model)
        dec_out_waypoints, dec_enc_attns_waypoints = self.attention_model_waypoints(enc_inputs=encoded_text, dec_inputs=encoded_sem_map_in)
        
        dec_out_waypoints = dec_out_waypoints.permute(0,2,1).view(dec_out_waypoints.shape[0], -1, map_enc_res, map_enc_res) # reshape map attn # B x 128 x 32 x 32


        if self.use_first_waypoint:
            point_heatmap = batch['goal_heatmap'][:,:,0,:,:] # B x T x 1 x 24 x 24
            point_heatmap = F.interpolate(point_heatmap, size=(dec_out_waypoints.shape[2], dec_out_waypoints.shape[3]), mode='nearest')
            point_heatmap = point_heatmap.view(B*T, point_heatmap.shape[2], point_heatmap.shape[3]).unsqueeze(1)
            pred_in = torch.cat((dec_out_waypoints, point_heatmap), dim=1)
            pred_in = self.conv_point1(pred_in)
        else:
            pred_in = dec_out_waypoints

        waypoints_heatmaps, waypoints_cov_raw = self.goal_pred_model(pred_in)
        # get the prob distribution over uncovered/covered for each waypoint
        waypoints_cov = F.softmax(waypoints_cov_raw, dim=2)


        pred_maps_raw_spatial = pred_maps_raw_spatial.view(B,T,spatial_C,H,W)
        pred_maps_spatial = F.softmax(pred_maps_raw_spatial, dim=2)
        pred_maps_raw_objects = pred_maps_raw_objects.view(B,T,objects_C,H,W)
        pred_maps_objects = F.softmax(pred_maps_raw_objects, dim=2)

        output = {'pred_maps_raw_spatial':pred_maps_raw_spatial,
                  'pred_maps_raw_objects':pred_maps_raw_objects,
                  'pred_maps_spatial':pred_maps_spatial,
                  'pred_maps_objects':pred_maps_objects,
                  'pred_waypoints':waypoints_heatmaps,
                  'waypoints_cov_raw':waypoints_cov_raw,
                  'waypoints_cov':waypoints_cov,
                  'dec_out_waypoints':dec_out_waypoints,
                  'dec_enc_attns_waypoints':dec_enc_attns_waypoints
                  }

        if self.attention_model_map_prediction is not None:
            output['dec_out_map'] = dec_out_map
            output['dec_enc_attns_map'] = dec_enc_attns_map

        return output


    def map_prediction_loss(self, pred_output, input_batch):
        pred_maps_raw_spatial = pred_output['pred_maps_raw_spatial']
        pred_maps_raw_objects = pred_output['pred_maps_raw_objects']
        B, T, spatial_C, cH, cW = pred_maps_raw_spatial.shape
        objects_C = pred_maps_raw_objects.shape[2]

        gt_crops_spatial, gt_crops_objects = input_batch['map_occupancy'].long(), input_batch['map_semantic'].long()
        pred_map_loss_spatial = self.cel_loss_spatial(input=pred_maps_raw_spatial.view(B*T,spatial_C,cH,cW), target=gt_crops_spatial.view(B*T,cH,cW))
        pred_map_loss_objects = self.cel_loss_objects(input=pred_maps_raw_objects.view(B*T,objects_C,cH,cW), target=gt_crops_objects.view(B*T,cH,cW))
        
        pred_map_err_spatial = pred_map_loss_spatial.clone().detach()
        pred_map_err_objects = pred_map_loss_objects.clone().detach()

        output = {'spatial_loss':pred_map_loss_spatial*self.map_loss_scale, 'objects_loss':pred_map_loss_objects*self.map_loss_scale,
                'spatial_error':pred_map_err_spatial, 'objects_error':pred_map_err_objects}
        return output



    def coverage_loss(self, pred_output, input_batch):
        waypoints_cov_raw = pred_output['waypoints_cov_raw']
        gt_waypoints_cov = input_batch['covered_waypoints'].long()
        B, T, num_waypoints = gt_waypoints_cov.shape
        gt_waypoints_cov = gt_waypoints_cov.view(B*T, num_waypoints)

        if self.use_first_waypoint:
            gt_waypoints_cov = gt_waypoints_cov[:,1:]
            num_waypoints = num_waypoints-1 # needed for reshaping in the loss
        cov_loss = self.cel_loss_cov(input=waypoints_cov_raw.contiguous().view(B*T*num_waypoints,2), 
                                     target=gt_waypoints_cov.contiguous().view(B*T*num_waypoints))

        cov_err = cov_loss.clone().detach()
        output = {'cov_loss': cov_loss*self.cov_loss_scale, 'cov_error': cov_err}
        return output


    def position_loss(self, pred_output, input_batch):
        pred_waypoints = pred_output['pred_waypoints']
        gt_waypoints = input_batch['goal_heatmap'] # B x T x num_waypoints x h x w
        visible_waypoints = input_batch['visible_waypoints'] # B x T x num_waypoints

        B, T, num_waypoints, cH, cW = gt_waypoints.shape
        gt_waypoints = gt_waypoints.view(B*T, num_waypoints, cH, cW)
        visible_waypoints = visible_waypoints.view(B*T, num_waypoints)
        visible_waypoints = visible_waypoints.unsqueeze(2).unsqueeze(2).repeat(1, 1, cH, cW)

        if self.use_first_waypoint:
            gt_waypoints = gt_waypoints[:,1:,:,:]
            visible_waypoints = visible_waypoints[:,1:,:,:]

        # Mask waypoints that are not visible in each example
        # pred waypoints is already shaped B*T x num_waypoints x h x w
        pred_waypoints = pred_waypoints*visible_waypoints
        gt_waypoints = gt_waypoints*visible_waypoints # non visible gt heatmaps should anyway be empty

        loss = self.position_mse_loss(pred_waypoints, gt_waypoints)
        err = loss.clone().detach()
        output = {'position_loss': loss*self.pos_loss_scale, 'position_error': err}
        return output
