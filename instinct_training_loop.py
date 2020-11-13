from math import log

import torch
import numpy as np
import gym
from gym.envs.registration import register
from gym.utils import seeding
import safety_gym_mod
import matplotlib.pyplot as plt

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_default_ppo, Policy, custom_weight_init
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
try:
    from exp_dir_util import get_experiment_save_dir
except:
    pass

from copy import deepcopy
from os.path import join
from torch.utils.tensorboard import SummaryWriter
from enum import Enum

from double_rl_loop_main import  ENV_NAME, NUM_PROC,policy_instinct_combinator, reward_cost_combinator, \
    compare_two_models, EvalActorCritic


def instinct_loop_ppo(
        args,
        learning_rate,
        num_steps,
        num_updates,
        inst_on,
        visualize,
        save_dir
):
    torch.set_num_threads(1)
    log_writer = SummaryWriter(save_dir, max_queue=1, filename_suffix="log")
    device = torch.device("cpu")

    env_name = ENV_NAME #"Safexp-PointGoal1-v0"
    envs = make_vec_envs(env_name, np.random.randint(2 ** 32), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    eval_envs = make_vec_envs(env_name, np.random.randint(2 ** 32), 1,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

    actor_critic_policy = torch.load("pretrained_policy.pt")

    # Prepare modified observation shape for instinct
    obs_shape = envs.observation_space.shape
    inst_action_space = deepcopy(envs.action_space)
    inst_obs_shape = list(obs_shape)
    inst_obs_shape[0] = inst_obs_shape[0] + envs.action_space.shape[0]
    # Prepare modified action space for instinct
    inst_action_space.shape = list(inst_action_space.shape)
    inst_action_space.shape[0] = inst_action_space.shape[0] + 1
    inst_action_space.shape = tuple(inst_action_space.shape)
    actor_critic_instinct = Policy(tuple(inst_obs_shape),
                                   inst_action_space,
                                   init_log_std=log(args.init_sigma),
                                   base_kwargs={'recurrent': False})
    actor_critic_policy.to(device)
    actor_critic_instinct.to(device)

    #agent_policy = algo.PPO(
    #    actor_critic_policy,
    #    args.clip_param,
    #    args.ppo_epoch,
    #    args.num_mini_batch,
    #    args.value_loss_coef,
    #    args.entropy_coef,
    #    lr=learning_rate,
    #    eps=args.eps,
    #    max_grad_norm=args.max_grad_norm)

    agent_instinct = algo.PPO(
        actor_critic_instinct,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=learning_rate,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)


    rollouts_cost = RolloutStorage(num_steps, NUM_PROC,
                                   inst_obs_shape, inst_action_space,
                                   actor_critic_instinct.recurrent_hidden_state_size)

    obs = envs.reset()
    i_obs = torch.cat([obs, torch.zeros((NUM_PROC, envs.action_space.shape[0]))], dim=1)  # Add zero action to the observation
    rollouts_cost.obs[0].copy_(i_obs)
    rollouts_cost.to(device)

    fitnesses = []
    best_fitness_so_far = float("-Inf")

    masks = torch.ones(num_steps + 1, NUM_PROC, 1)
    recurrent_hidden_states = torch.zeros(num_steps + 1, NUM_PROC, actor_critic_policy.recurrent_hidden_state_size)

    for j in range(num_updates):
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                # (value, action, action_log_probs, rnn_hxs), (instinct_value, instinct_action, instinct_outputs_log_prob, i_rnn_hxs), final_action
                value, action, action_log_probs, recurrent_hidden_states = actor_critic_policy.act(
                    obs,
                    recurrent_hidden_states,
                    masks,
                    deterministic=True
                )
                instinct_value, instinct_action, instinct_outputs_log_prob, instinct_recurrent_hidden_states = actor_critic_instinct.act(
                    rollouts_cost.obs[step],
                    rollouts_cost.recurrent_hidden_states[step],
                    rollouts_cost.masks[step],
                    deterministic=False,
                )

            # Combine two networks
            final_action, i_control = policy_instinct_combinator(action, instinct_action)
            obs, reward, done, infos = envs.step(final_action)
            envs.render()

            reward, violation_cost = reward_cost_combinator(reward, infos, NUM_PROC, i_control)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            i_obs = torch.cat([obs, action], dim=1)
            rollouts_cost.insert(i_obs, instinct_recurrent_hidden_states, instinct_action, instinct_outputs_log_prob,
                                 instinct_value, violation_cost, masks, bad_masks)

        with torch.no_grad():

            next_value_instinct = actor_critic_instinct.get_value(rollouts_cost.obs[-1],
                                                                  rollouts_cost.recurrent_hidden_states[-1],
                                                                  rollouts_cost.masks[-1].detach())

        rollouts_cost.compute_returns(next_value_instinct, args.use_gae, args.gamma,
                                      args.gae_lambda, args.use_proper_time_limits)

        print("training instinct")
        # Instinct training phase
        p_before = deepcopy(actor_critic_policy)
        val_loss_i, action_loss_i, dist_entropy_i = agent_instinct.update(rollouts_cost)
        p_after = deepcopy(actor_critic_policy)
        assert compare_two_models(p_before, p_after), "policy changed when it shouldn't"


        rollouts_cost.after_update()

        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms

        fits, info = evaluate(EvalActorCritic(actor_critic_policy, actor_critic_instinct), ob_rms, eval_envs, NUM_PROC,
                                    reward_cost_combinator, device, instinct_on=inst_on, visualise=visualize)
        eval_cost = info['cost']
        print(
            f"Step {j}, Fitness {fits.item()} ")
        print(
            f"Step {j}, Cost {eval_cost}, value_loss instinct = {val_loss_i}, action_loss instinct= {action_loss_i}, "
            f"dist_entropy instinct = {dist_entropy_i}")
        print("-----------------------------------------------------------------")

        # Tensorboard logging
        log_writer.add_scalar("fitness", fits.item(), j)
        log_writer.add_scalar("cost", eval_cost, j)
        log_writer.add_scalar("value loss instinct", val_loss_i, j)
        log_writer.add_scalar("action loss instinct", action_loss_i, j)
        log_writer.add_scalar("dist entropy instinct", dist_entropy_i, j)

        fitnesses.append(fits)
        if fits.item() > best_fitness_so_far:
            best_fitness_so_far = fits.item()
            torch.save(actor_critic_instinct, join(save_dir, "model_rl_instinct.pt"))
        torch.save(actor_critic_instinct, join(save_dir, "model_rl_instinct_latest.pt"))
    return (fitnesses[-1]), 0, 0


def main():
    args = get_args()
    print("start the train function")

    args.init_sigma = 0.6
    args.lr = 0.001

    # plot_weight_histogram(parameters)
    exp_save_dir = get_experiment_save_dir(args)

    instinct_loop_ppo(
        args,
        args.lr,
        num_steps=1000,
        num_updates=4000,
        inst_on=False,
        visualize=False,
        save_dir=exp_save_dir
    )


if __name__ == "__main__":
    main()
