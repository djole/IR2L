import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init
from model import ControllerInstinct, weight_init
from copy import deepcopy


def custom_weight_init(module):
    stdv = 0.3  # 1. / math.sqrt(module.weight.size(1))
    if isinstance(module, nn.Linear):
        module.weight.data.uniform_(-stdv, stdv)
        module.bias.data.uniform_(-stdv, stdv)


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class PolicyWithInstinct(nn.Module):
    def __init__(self, obs_shape, action_space, init_log_std, base=None, base_kwargs=None, load_instinct=False):
        super(PolicyWithInstinct, self).__init__()
        self.policy = Policy(obs_shape, action_space, init_log_std, base, base_kwargs)
        instinct_input_shape = (obs_shape[0] + action_space.shape[0],)  # Add policy output to instinct input
        instinct_action_space = deepcopy(action_space)
        instinct_action_space.shape = (action_space.shape[0],)
        self.instinct = Policy(instinct_input_shape, instinct_action_space, init_log_std, base, base_kwargs)

        self.freeze_instinct = load_instinct
        self.apply(custom_weight_init)

    @property
    def is_recurrent(self):
        return False

    @property
    def recurrent_hidden_state_size(self):
        return self.policy.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def get_combinator_params(self):
        return self.policy.parameters()

    def get_evolvable_params(self):
        comb_params = []
        dct = self.named_parameters()
        for pkey, ptensor in dct:
            if not ("instinct" in pkey and self.freeze_instinct):
                comb_params.append(ptensor)
        return comb_params

    def parameters(self):
        return self.policy.parameters()

    def act(self, inputs, rnn_hxs, i_rnn_hxs, masks, i_masks, deterministic=False, instinct_deterministic=False,
            instinct_on=False):
        instinct_on = False
        value, action, action_log_probs, rnn_hxs = self.policy.act(inputs, rnn_hxs, masks, deterministic)
        instinct_inputs = torch.cat([inputs, torch.zeros_like(action)], dim=1)  # TODO remove the zeros after debugging
        instinct_value, instinct_outputs, instinct_outputs_log_prob, i_rnn_hxs = \
            self.instinct.act(instinct_inputs, i_rnn_hxs, i_masks, instinct_deterministic)

        half_output = int(instinct_outputs.shape[1] / 2)
        instinct_action = instinct_outputs #[:, half_output:]
        # instinct_control = instinct_outputs[:, :half_output]
        # TODO Remove
        # instinct_outputs[:, :half_output] = instinct_outputs[:, :half_output] * 0.0


        # controlled_stoch_action = action * instinct_control
        controlled_instinct_action = instinct_action  # * (1 - instinct_control)

        if instinct_on:
            final_action = controlled_instinct_action  # + control_stoch_action # TODO return this after debugging
            assert False, "instinct is on / TESTING"
        else:
            final_action = action

        return (value, action, action_log_probs, rnn_hxs), \
               (instinct_value, instinct_outputs, instinct_outputs_log_prob, i_rnn_hxs, instinct_inputs), \
               final_action

    def get_value(self, inputs, rnn_hxs, masks):
        assert False, "This make no sense at this point"
        return self.policy.get_value(inputs, rnn_hxs, masks)

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        assert False, "This make no sense at this point"
        return self.policy.evaluate_actions(inputs, rnn_hxs, masks, action)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, init_log_std, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                base = CNNBase
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs, init_log_std)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                         .any(dim=-1)
                         .nonzero()
                         .squeeze()
                         .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class MLPBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(MLPBase, self).__init__(recurrent, num_inputs, hidden_size)

        if recurrent:
            num_inputs = hidden_size

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
        #                       constant_(x, 0), np.sqrt(2))

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh(),
            nn.Linear(hidden_size, hidden_size), nn.Tanh())

        self.critic_linear = nn.Linear(hidden_size, 1)

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = inputs

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        hidden_critic = self.critic(x)
        hidden_actor = self.actor(x)

        return self.critic_linear(hidden_critic), hidden_actor, rnn_hxs


def init_ppo(env, init_log_std):
    actor_critic = PolicyWithInstinct(
        env.observation_space.shape,
        env.action_space,
        init_log_std=init_log_std,
        base_kwargs={'recurrent': False},
    )
    return actor_critic


def init_default_ppo(env, init_log_std):
    actor_critic = Policy(env.observation_space.shape,
                          env.action_space,
                          init_log_std=init_log_std,
                          base_kwargs={'recurrent': False})
    return actor_critic
