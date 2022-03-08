import torch
import torch.nn as nn
import torch.nn.functional as F


class VLNGoalPredictor(nn.Module):
    def __init__(self, map_encoder, bert_model, attention_model, goal_pred_model, pos_loss_scale, cov_loss_scale,
                                                            d_model, use_first_waypoint, loss_norm, text_init_dim=768):
        super(VLNGoalPredictor, self).__init__()

        self.map_encoder = map_encoder
        self.bert_model = bert_model
        self.text_encoder = nn.Linear(in_features=text_init_dim, out_features=d_model) # to produce same size embeddings as the map
        self.attention_model = attention_model
        self.goal_pred_model = goal_pred_model

        if loss_norm:
            self.position_mse_loss = nn.MSELoss()
        else:
            self.position_mse_loss = nn.MSELoss(reduction="sum")
        
        self.cel_loss_cov = nn.CrossEntropyLoss(reduction="sum")

        self.pos_loss_scale = pos_loss_scale
        self.cov_loss_scale = cov_loss_scale
        self.d_model = d_model
        self.use_first_waypoint = use_first_waypoint

        if self.use_first_waypoint:
            self.conv_point1 = nn.Conv2d(d_model+1, d_model, 1, 1, padding=0)

    
    def forward(self, batch):
        semantic_map = batch['map_semantic']
        
        # bert inputs
        tokens_tensor = batch['tokens_tensor'].long()
        segments_tensors = batch['segments_tensors'].long()

        B, T, C, H, W = semantic_map.shape # batch, sequence, 1, height, width
        semantic_map = semantic_map.view(B*T, C, H, W)

        tokens_tensor = tokens_tensor.squeeze(1)
        segments_tensors = segments_tensors.squeeze(1)

        outputs = self.bert_model(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]

        text_feat=[]
        for b in range(tokens_tensor.shape[0]):
            text_feat.append(hidden_states[-1][b][:])
        text_feat = torch.stack(text_feat)

        ### Prepare the input embeddings of the map and text
        encoded_map = self.map_encoder(semantic_map) # B x 128 x 32 x 32
        map_enc_res = encoded_map.shape[2]

        encoded_map_in = encoded_map.permute(0,2,3,1).view(encoded_map.shape[0], -1, self.d_model)
        encoded_text = self.text_encoder(text_feat)

        # replicate encoded text for number of encoded_map_in
        if T>1:
            encoded_text = encoded_text.unsqueeze(1).repeat(1, T, 1, 1)
            encoded_text = encoded_text.view(B*T, encoded_text.shape[2], encoded_text.shape[3])

        ### Apply attention between the map and text
        dec_out, dec_enc_attns = self.attention_model(enc_inputs=encoded_text, dec_inputs=encoded_map_in) # B x 1024 x 128

        ### Use the attention-augmented map embeddings to predict the waypoints
        dec_out = dec_out.permute(0,2,1).view(dec_out.shape[0], -1, map_enc_res, map_enc_res) # reshape map attn # B x 128 x 32 x 32

        if self.use_first_waypoint:
            point_heatmap = batch['goal_heatmap'][:,:,0,:,:] # B x T x 1 x 48 x 48
            point_heatmap = F.interpolate(point_heatmap, size=(dec_out.shape[2], dec_out.shape[3]), mode='nearest')
            point_heatmap = point_heatmap.view(B*T, point_heatmap.shape[2], point_heatmap.shape[3]).unsqueeze(1)
            pred_in = torch.cat((dec_out, point_heatmap), dim=1)
            pred_in = self.conv_point1(pred_in)
        else:
            pred_in = dec_out

        waypoints_heatmaps, waypoints_cov_raw = self.goal_pred_model(pred_in)

        # get the prob distribution over uncovered/covered for each waypoint
        waypoints_cov = F.softmax(waypoints_cov_raw, dim=2)

        return waypoints_heatmaps, waypoints_cov_raw, waypoints_cov, dec_enc_attns


    def coverage_loss(self, waypoints_cov_raw, input_batch):
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


    def position_loss(self, pred_waypoints, input_batch):
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
