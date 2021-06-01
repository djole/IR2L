import numpy as np
import torch

from a2c_ppo_acktr import utils
from a2c_ppo_acktr.envs import make_vec_envs


def evaluate(actor_critic, ob_rms, eval_envs, num_processes, reward_cost_combinator,
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
    hazard_collisions = 0
    total_instinct_reward = 0

    plot_info_step_dicts = []
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
        hazard_collisions += infos[0]['cost']
        total_reward, instinct_reward = reward_cost_combinator(reward, infos, 1, i_control)


        if visualise:
            eval_envs.render()
            plot_info_dict = infos[0]['plot_info'].copy()
            plot_info_dict['instinct regulation'] = (1.0 - i_control.clone().item())
            plot_info_step_dicts.append(plot_info_dict)

        eval_masks = torch.tensor(
            [[0.0] if done_ else [1.0] for done_ in done],
            dtype=torch.float32,
            device=device)

        cummulative_reward += total_reward
        total_instinct_reward += instinct_reward
    return cummulative_reward, {'instinct_reward': total_instinct_reward,
                                'hazard_collisions': hazard_collisions,
                                'plot_info_steps': plot_info_step_dicts,
                                }
