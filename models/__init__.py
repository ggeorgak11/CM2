
import torch
import torch.nn as nn
import os
from models.networks import MapEncoder, MapAttention, ResNetUNetGoalPred, ResNetUNetHierarchical
from .vln_goal_predictor import VLNGoalPredictor
from .vln_no_map_goal_predictor import VLNGoalPredictor_UnknownMap
from transformers import BertModel
import test_utils as tutils


def load_bert(options):
    bert_model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    print("Loaded BERT model!")
    if options.finetune_bert_last_layer:
        for param in bert_model.named_parameters():
            if param[0] == 'pooler.dense.weight' or param[0] == 'pooler.dense.bias': # keep the last layers as requires_grad=True
                continue
            param[1].requires_grad = False
    return bert_model


def get_model_from_options(options):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if options.use_first_waypoint:
        out_channels = options.num_waypoints-1
    else:
        out_channels = options.num_waypoints

    if options.vln:
        # Model that assumes egocentric ground-truth semantic map is given
        bert_model = load_bert(options)

        map_encoder = MapEncoder(n_channel_in=1, n_channel_out=options.d_model)
        attention_model = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        goal_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm)
        
        return VLNGoalPredictor(map_encoder=map_encoder,
                                bert_model=bert_model,
                                attention_model=attention_model,
                                goal_pred_model=goal_pred_model,
                                pos_loss_scale=options.position_loss_scale,
                                cov_loss_scale=options.cov_loss_scale,
                                loss_norm=options.loss_norm,
                                d_model=options.d_model,
                                use_first_waypoint=options.use_first_waypoint)

    elif options.vln_no_map:
        # Model that performs hallucination in addition to waypoint prediction
        bert_model = load_bert(options)
        
        if options.without_attn_1:
            map_encoder = None
            attention_model_map_prediction = None
        else:
            # Attention for map prediction
            map_encoder = MapEncoder(n_channel_in=options.n_spatial_classes, n_channel_out=options.d_model)
            attention_model_map_prediction = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        # Hierarchical prediction model
        map_predictor = ResNetUNetHierarchical(out1_n_class=options.n_spatial_classes, out2_n_class=options.n_object_classes, without_attn=options.without_attn_1)

        # Second encoder + attention for waypoint prediction
        map_encoder_sem = MapEncoder(n_channel_in=options.n_object_classes, n_channel_out=options.d_model)
        attention_model_waypoints = MapAttention(d_model=options.d_model, d_ff=options.d_ff, d_k=options.d_k, d_v=options.d_v, 
                                                                n_heads=options.n_heads, n_layers=options.n_layers, device=device)

        goal_pred_model = ResNetUNetGoalPred(n_channel_in=options.d_model, n_class_out=out_channels, with_lstm=options.with_lstm)
        
        return VLNGoalPredictor_UnknownMap(map_encoder=map_encoder,
                                        bert_model=bert_model,
                                        attention_model_map_prediction=attention_model_map_prediction,
                                        map_predictor=map_predictor,
                                        map_encoder_sem=map_encoder_sem,
                                        attention_model_waypoints=attention_model_waypoints,
                                        goal_pred_model=goal_pred_model,
                                        map_loss_scale=options.map_loss_scale,
                                        pos_loss_scale=options.position_loss_scale,
                                        cov_loss_scale=options.cov_loss_scale,
                                        loss_norm=options.loss_norm,
                                        d_model=options.d_model,
                                        use_first_waypoint=options.use_first_waypoint)
