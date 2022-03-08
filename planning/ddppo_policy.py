import torch
import torch.nn as nn

import numpy as np

from gym.spaces import dict_space
from gym.spaces.box import Box
from habitat_baselines.rl.ddppo.policy.resnet_policy import PointNavResNetPolicy

from gym.spaces import Box, Dict, Discrete

class DdppoPolicy(nn.Module):
    def __init__(self,
                 path):
        super().__init__()

        spaces = {
            'pointgoal_with_gps_compass': Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            )
        }

        spaces["depth"] = Box(
                low=0,
                high=1,
                shape=(256, 256, 1),
                dtype=np.float32,
            )

        observation_space = Dict(spaces)
        action_space = Discrete(4)

        checkpoint = torch.load(path)
        self.hidden_size = checkpoint['model_args'].hidden_size
        # The model must be named self.actor_critic to make the namespaces correct for loading
        self.actor_critic = PointNavResNetPolicy(observation_space=observation_space,
            action_space=action_space,
            hidden_size=self.hidden_size,
            num_recurrent_layers=2,
            rnn_type="LSTM",
            backbone="resnet50")

        self.actor_critic.load_state_dict(
                {
                    k[len("actor_critic.") :]: v
                    for k, v in checkpoint['state_dict'].items()
                    if "actor_critic" in k
                }
            )
        self.actor_critic.eval()

        self.hidden_state = torch.zeros(1, self.actor_critic.net.num_recurrent_layers, checkpoint['model_args'].hidden_size)
        self.prev_actions = torch.zeros(1, 1, dtype=torch.long)



    def plan(self, depth, goal,t):
        batch = {'depth': depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2]),
                 'pointgoal_with_gps_compass': goal.view(1, -1)}

        if t ==0:
            not_done_masks = torch.zeros(1, 1, dtype=torch.bool, device=depth.device)
        else:
            not_done_masks = torch.ones(1, 1, dtype=torch.bool, device=depth.device)

        _, actions, _, self.hidden_state = self.actor_critic.act(batch,
                                                                 self.hidden_state.to(depth.device),
                                                                 self.prev_actions.to(depth.device),
                                                                 not_done_masks,
                                                                 deterministic=False)
        self.prev_actions = torch.clone(actions)
        return actions.item()

    def reset(self):
        self.hidden_state = torch.zeros_like(self.hidden_state)
        self.prev_actions = torch.zeros_like(self.prev_actions)
