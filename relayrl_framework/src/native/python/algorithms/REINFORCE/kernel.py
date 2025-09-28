from torch.distributions import Categorical

from _common._algorithms.BaseKernel import mlp, ForwardKernelAbstract, StepKernelAbstract

from typing import Optional, Type

import torch
import torch.nn as nn
from numpy import ndarray


class DiscretePolicyNetwork(ForwardKernelAbstract):
    def __init__(self, obs_dim, hidden_sizes, act_dim):
        super().__init__()
        self.pi_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim),
        )

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        x = self.pi_network(obs)
        logits = torch.squeeze(x, -1)
        logits = logits + (mask-1) * 1e8
        probs = torch.softmax(logits, dim=-1)
        return probs, logits

    def _sample_for_action(self, probs):
        return torch.multinomial(probs, 1)

    def _log_prob_from_distribution(self, logits, act):
        act = act.long().unsqueeze(-1)
        act, log_pmf = torch.broadcast_tensors(act, logits)
        act = act[..., :1]
        return log_pmf.gather(-1, act).squeeze(-1)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: Optional[torch.Tensor] = None):
        probs, logits = self._distribution(obs, mask)

        if act is not None:
            logp_a = self._log_prob_from_distribution(logits, act)
            return probs, logits, logp_a

        return probs, logits, None


class ContinuousPolicyNetwork(ForwardKernelAbstract):
    def __init__(self, obs_dim, hidden_sizes, act_dim):
        super().__init__()
        self.pi_network = nn.Sequential(
            nn.Linear(obs_dim, hidden_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_sizes[1], act_dim),
        )

        log_std = -0.5 * torch.ones(act_dim, dtype=torch.float32)
        self.log_std = torch.nn.Parameter(log_std)

    def _distribution(self, obs: torch.Tensor, mask: torch.Tensor):
        x = obs.view(-1, obs.size(-1), obs.shape[1])
        x = self.pi_network(x)
        mean = torch.squeeze(x, -1)
        mean = mean + (mask - 1) * 1e8
        return torch.distributions.Normal(mean, self.log_std.exp())

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, act: Optional[torch.Tensor] = None):
        pi = self._distribution(obs, mask)
        if act is not None:
            logp_a = self._log_prob_from_distribution(pi, act)
            return pi, logp_a
        return pi, None


class BaselineValueNetwork(ForwardKernelAbstract):
    def __init__(self, obs_dim, hidden_sizes, activation):
        super().__init__()
        self.v_network = mlp([obs_dim] + hidden_sizes + [1], activation)

    def forward(self, obs: torch.Tensor, mask: torch.Tensor):
        return torch.squeeze(self.v_network(obs), -1)


class PolicyWithBaseline(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, discrete, hidden_sizes, activation):
        super().__init__()
        if discrete:
            self.policy = DiscretePolicyNetwork(obs_dim, hidden_sizes, act_dim)
        else:
            self.policy = ContinuousPolicyNetwork(obs_dim, hidden_sizes, act_dim)
        self.baseline = BaselineValueNetwork(obs_dim, hidden_sizes, activation)
        
        self.input_dim = obs_dim
        self.output_dim = act_dim

    @torch.jit.export
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            probs, logits, _ = self.policy.forward(obs, mask)
            act = self.policy._sample_for_action(probs)
            logp_a = self.policy._log_prob_from_distribution(logits, act)
            v = self.baseline.forward(obs, mask)
        data = {'logp_a': logp_a, 'v': v}
        return act, data
        
    @torch.jit.export  # Used for rust model validation
    def get_input_dim(self):
        return self.input_dim

    @torch.jit.export  # Used for rust model validation
    def get_output_dim(self):
        return self.output_dim

class PolicyWithoutBaseline(StepKernelAbstract):
    def __init__(self, obs_dim, act_dim, discrete, hidden_sizes):
        super().__init__()
        if discrete:
            self.policy = DiscretePolicyNetwork(obs_dim, hidden_sizes, act_dim)
        else:
            self.policy = ContinuousPolicyNetwork(obs_dim, hidden_sizes, act_dim)
        
        self.input_dim = obs_dim
        self.output_dim = act_dim
            
    @torch.jit.export
    def step(self, obs: torch.Tensor, mask: torch.Tensor):
        with torch.no_grad():
            probs, logits, _ = self.policy.forward(obs, mask)
            act = self.policy._sample_for_action(probs)
            logp_a = self.policy._log_prob_from_distribution(logits, act)
        data = {'logp_a': logp_a}
        return act, data
        
    @torch.jit.export  # Used for rust model validation
    def get_input_dim(self):
        return self.input_dim

    @torch.jit.export  # Used for rust model validation
    def get_output_dim(self):
        return self.output_dim
        