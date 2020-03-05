import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, eval_envs, num_processes,
             device, instinct_on):


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
    while not done:
        with torch.no_grad():
            _, action, _, eval_recurrent_hidden_states, (final_action, _) = actor_critic.act(
                obs,
                eval_recurrent_hidden_states,
                eval_masks,
                deterministic=True,
                instinct_on=instinct_on,
            )

        # Obser reward and next obs
        obs, reward, done, infos = eval_envs.step(final_action)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        cummulative_reward += reward
    return cummulative_reward, infos
