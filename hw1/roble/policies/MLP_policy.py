import abc
import itertools
from typing import Any
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from hw1.roble.infrastructure import pytorch_util as ptu
from hw1.roble.policies.base_policy import BasePolicy


class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self,
                 *args,
                 **kwargs
                 ):
        super().__init__()

        self._learn_policy_std = kwargs.get('learn_policy_std', False)  # Default to False
        if self._discrete:
            self._logits_na = ptu.build_mlp(input_size=self._ob_dim,
                                           output_size=self._ac_dim,
                                           params=self._network)
            self._logits_na.to(ptu.device)
            self._mean_net = None
            self._logstd = None
            self._optimizer = optim.Adam(self._logits_na.parameters(),
                                        self._learning_rate)
        else:
            self._logits_na = None
            self._mean_net = ptu.build_mlp(input_size=self._ob_dim,
                                      output_size=self._ac_dim,
                                      params=self._network)
            self._mean_net.to(ptu.device)

            if self._deterministic:
                self._optimizer = optim.Adam(
                    itertools.chain(self._mean_net.parameters()),
                    self._learning_rate
                )
            else:
                self._std = nn.Parameter(
                    torch.ones(self._ac_dim, dtype=torch.float32, device=ptu.device) * 0.1
                )
                self._std.to(ptu.device)
                if self._learn_policy_std:
                    self._optimizer = optim.Adam(
                        itertools.chain([self._std], self._mean_net.parameters()),
                        self._learning_rate
                    )
                else:
                    self._optimizer = optim.Adam(
                        itertools.chain(self._mean_net.parameters()),
                        self._learning_rate
                    )

        if self._nn_baseline:
            self._baseline = ptu.build_mlp(
                input_size=self._ob_dim,
                output_size=1,
                params=self._network
            )
            self._baseline.to(ptu.device)
            self._baseline_optimizer = optim.Adam(
                self._baseline.parameters(),
                self._critic_learning_rate,
            )
        else:
            self._baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        ## Provide the logic to produce an action from the policy

        if obs.ndim == 1:
            obs = np.expand_dims(obs, axis=0)
    
        actions = []
        for observation in obs:
            observation_tensor = torch.tensor(observation, dtype=torch.float32).to(ptu.device)
            action = self.forward(observation_tensor)
            actions.append(action.detach().cpu().numpy()) 
        return np.array(actions)  

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        if self._discrete:
            # Corrige
            logits = self._logits_na(observation)
            action_distribution = distributions.Categorical(logits=logits)
            return action_distribution
        else:
            if isinstance(observation, np.ndarray):
                observation = torch.tensor(observation, dtype=torch.float32).to(ptu.device)
            if self._deterministic:
                ##  TODO output for a deterministic policy
                action_distribution = self._mean_net(observation)
            else:    
                ##  TODO output for a stochastic policy
                mean = self._mean_net(observation)
                std = self._std.exp() # corriger
                action_distribution = distributions.Normal(mean, std).rsample()
        return action_distribution
    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        # pass
        raise NotImplementedError

#####################################################
#####################################################

class MLPPolicySL(MLPPolicy):
    import hw1.roble.util.class_util as classu
    @classu.hidden_member_initialize
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._loss = nn.MSELoss()

    def update(
        self, observations, actions,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        
        # TODO: update the policy and return the loss
        pred_actions = [self.forward(observation) for observation in observations ]
        pred_actions = torch.stack(pred_actions)
        actions = torch.tensor(actions, dtype=torch.float32)
        loss = self._loss(pred_actions,actions)
        assert pred_actions.shape == actions.shape, f"Shape mismatch: {pred_actions.shape} vs {actions.shape}"
        self._optimizer.zero_grad()
        loss.backward()
        self._optimizer.step()
        #print(f"pred_actions shape: {observations.shape}")
        #print(f"pred_actions shape: {pred_actions.shape}")
        #print(f"actions shape: {actions.shape}")

        return {
            'Training Loss': ptu.to_numpy(loss),
        }

    def update_idm(
        self, observations, actions, next_observations,
        adv_n=None, acs_labels_na=None, qvals=None
        ):
        
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        
        # TODO: Transform the numpy arrays to torch tensors (for obs, next_obs and actions)
        
        # TODO: Create the full input to the IDM model (hint: it's not the same as the actor as it takes both obs and next_obs)
        
        # TODO: Get the predicted actions from the IDM model (hint: you need to call the forward function of the IDM model)
        
        # TODO: Compute the loss using the MLP_policy loss function
        
        # TODO: Update the IDM model.
        loss = TODO
        return {
            'Training Loss IDM': ptu.to_numpy(loss),
        }