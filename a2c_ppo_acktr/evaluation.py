import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, eval_envs, num_processes,
             device, instinct_on=True, visualise=False):

    vec_norm = utils.get_vec_normalize(eval_envs)
    if vec_norm is not None:
        vec_norm.eval()
        vec_norm.ob_rms = ob_rms

    eval_episode_rewards = []

    obs = eval_envs.reset()
    eval_recurrent_hidden_states = torch.zeros(
        num_processes, actor_critic.recurrent_hidden_state_size, device=device)
    eval_masks = torch.zeros(num_processes, 1, device=device)

    done = False
    cummulative_reward = 0
    cost_hazards = 0
    cost = 0
    while not done:
        with torch.no_grad():
            _, final_action, i_control, _ = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True,
            )

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(final_action)
        total_reward = reward
        for info in infos:
            # total_reward -= info['cost']
            cost -= info['cost']

        # Add a regularization clause to discurage instinct to activate if not necessary
        for i_control_idx in range(len(i_control)):
            i_control_on_idx = i_control[i_control_idx]
            cost -= (1 - i_control_on_idx).sum().item() * 0.01

        if visualise:
            eval_envs.render()

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        cummulative_reward += total_reward
    return cummulative_reward, {'cost': cost}
