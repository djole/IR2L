import copy
import statistics
from math import log, sin, cos, pi

import torch
import numpy as np
import gym
from gym.envs.registration import register
from gym.utils import seeding
import safety_gym
import matplotlib.pyplot as plt

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_default_ppo, init_ppo, PolicyWithInstinct
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
from main_es import get_model_weights
from copy import deepcopy
from torch.utils.tensorboard import SummaryWriter

# config = {
#    'robot_base': 'xmls/point.xml',
#    #'observe_sensors': False,
#    'observe_goal_lidar': True,
#    'constrain_hazards': True,
#    'hazards_num': 4,
# }

NP_RANDOM, _ = seeding.np_random(None)
HAZARD_LOC_PARAM = 1
HLP = HAZARD_LOC_PARAM

# GOAL_LOC_PARAM = 0.8
# GLP = GOAL_LOC_PARAM
# GOALS = [(-GLP, -GLP), (GLP, GLP), (GLP, -GLP), (-GLP, GLP)]

# register(id='SafexpCustomEnvironment-v0',
#         entry_point='safety_gym.envs.mujoco:Engine',
#         kwargs={'config': config})

NUM_PROC = 1


def plot_weight_histogram(parameters):
    flattened_params = []
    for p in parameters:
        flattened_params.append(p.flatten())

    params_stacked = np.concatenate(flattened_params)
    plt.hist(params_stacked, bins=300)
    plt.show()


def apply_from_list(weights, model: PolicyWithInstinct):
    to_params_dct = model.get_evolvable_params()

    for ptensor, w in zip(to_params_dct, weights):
        w_tensor = torch.Tensor(w)
        ptensor.data.copy_(w_tensor)


def initialize_model(envs, init_sigma, learning_rate):
    blueprint_model = init_ppo(envs, log(init_sigma))
    parameters = get_model_weights(blueprint_model)
    parameters.append(np.array([learning_rate]))
    return parameters


def inner_loop_ppo(
        args,
        learning_rate,
        num_steps,
        num_updates,
        inst_on,
        visualize
):
    torch.set_num_threads(1)
    log_writer = SummaryWriter("./log", max_queue=1, filename_suffix="log")
    device = torch.device("cpu")

    env_name = "Safexp-PointGoal0-v0"
    envs = make_vec_envs(env_name, np.random.randint(2 ** 32), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

    # actor_critic = torch.load("/Users/djrg/code/instincts/modular_rl_safety_gym/model_rl.pt")  # init_ppo(envs, log(args.init_sigma))
    actor_critic = init_ppo(envs, log(args.init_sigma))
    actor_critic.to(device)
    actor_critic_policy = actor_critic.policy
    actor_critic_instinct = actor_critic.instinct

    # apply the weights to the model
    weights = initialize_model(envs, args.init_sigma, args.lr)
    apply_from_list(weights, actor_critic)

    agent_policy = algo.PPO(
        actor_critic_policy,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=learning_rate,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

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

    rollouts_rewards = RolloutStorage(num_steps, NUM_PROC,
                                      envs.observation_space.shape, envs.action_space,
                                      actor_critic.recurrent_hidden_state_size)

    instinct_observation_space_shape = (envs.observation_space.shape[0] + envs.action_space.shape[0],)
    instinct_action_space = deepcopy(envs.action_space)
    instinct_action_space.shape = (envs.action_space.shape[0],)
    rollouts_cost = RolloutStorage(num_steps, NUM_PROC,
                                   instinct_observation_space_shape, instinct_action_space,
                                   actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    i_obs = torch.cat([obs, torch.zeros((NUM_PROC, envs.action_space.shape[0]))],
                      dim=1)  # Add zero action to the observation
    rollouts_rewards.obs[0].copy_(obs)
    rollouts_rewards.to(device)
    rollouts_cost.obs[0].copy_(i_obs)
    rollouts_cost.to(device)

    fitnesses = []
    violation_cost = 0
    training_episode_cum_reward = 0
    training_episode_cum_cost = 0

    best_fitness_so_far = float("-Inf")

    for j in range(num_updates):

        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                # (value, action, action_log_probs, rnn_hxs), (instinct_value, instinct_action, instinct_outputs_log_prob, i_rnn_hxs), final_action
                policy_output_package, instinct_output_package, final_action = actor_critic.act(
                    rollouts_rewards.obs[step],
                    rollouts_rewards.recurrent_hidden_states[step], rollouts_cost.recurrent_hidden_states[step],
                    rollouts_rewards.masks[step], rollouts_cost.masks[step],
                    instinct_on=inst_on)

            # Unpack data from two streams (policy and instinct)
            value, action, action_log_probs, recurrent_hidden_states = policy_output_package
            instinct_value, instinct_action, instinct_outputs_log_prob, instinct_recurrent_hidden_states, i_obs = \
                instinct_output_package
            # Observe reward and next obs
            obs, reward, done, infos = envs.step(final_action)
            # envs.render()

            # Count the cost

            violation_cost = torch.Tensor([[0]])
            for info in infos:
                violation_cost -= info['cost']  # Violation costs should be negative when training instinct

            training_episode_cum_reward += reward
            training_episode_cum_cost += violation_cost
            if done[0]:
                # print(f"{training_episode_cum_reward[0][0]},")
                training_episode_cum_reward = 0
                training_episode_cum_cost = 0
            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts_rewards.insert(obs, recurrent_hidden_states, action,
                                    action_log_probs, value, reward, masks, bad_masks)
            rollouts_cost.insert(i_obs, instinct_recurrent_hidden_states, instinct_action, instinct_outputs_log_prob,
                                 instinct_value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value_policy = actor_critic_policy.get_value(
                rollouts_rewards.obs[-1], rollouts_rewards.recurrent_hidden_states[-1],
                rollouts_rewards.masks[-1]).detach()
            next_value_instinct = actor_critic_instinct.get_value(rollouts_cost.obs[-1],
                                                                  rollouts_cost.recurrent_hidden_states[-1],
                                                                  rollouts_cost.masks[-1].detach())

        rollouts_rewards.compute_returns(next_value_policy, args.use_gae, args.gamma,
                                         args.gae_lambda, args.use_proper_time_limits)
        rollouts_cost.compute_returns(next_value_instinct, args.use_gae, args.gamma,
                                      args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent_policy.update(rollouts_rewards)
        val_loss_i, action_loss_i, dist_entropy_i = agent_instinct.update(rollouts_cost)

        rollouts_rewards.after_update()
        rollouts_cost.after_update()

        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms

        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device, instinct_on=inst_on,
                              visualise=visualize)
        eval_cost = info['cost']
        # print(f"Fitness {fits.item()}, cost = {eval_cost}, value_loss = {value_loss}, action_loss = {action_loss}, "
        #      f"dist_entropy = {dist_entropy}")
        # print(f"Step {j}, Fitness {fits.item()}, cost = {eval_cost}, value_loss = {val_loss_i}, action_loss = {action_loss_i}, "
        #      f"dist_entropy = {dist_entropy_i}")
        print(
            f"Step {j}, Fitness {fits.item()}, cost = {eval_cost}, value_loss = {value_loss}, action_loss = {action_loss}, "
            f"dist_entropy = {dist_entropy}")

        # Tensorboard logging
        # log_writer.add_scalar("fitness", fits.item(), j)
        # log_writer.add_scalar("value loss", val_loss_i, j)
        # log_writer.add_scalar("action loss", action_loss_i, j)
        # log_writer.add_scalar("dist entropy", dist_entropy_i, j)

        log_writer.add_scalar("fitness", fits.item(), j)
        log_writer.add_scalar("value loss", value_loss, j)
        log_writer.add_scalar("action loss", action_loss, j)
        log_writer.add_scalar("dist entropy", dist_entropy, j)

        fitnesses.append(fits)
        if fits.item() > best_fitness_so_far:
            best_fitness_so_far = fits.item()
            torch.save(actor_critic, "model_rl.pt")
    return (fitnesses[-1]), 0, 0


def main():
    args = get_args()
    print("start the train function")

    args.init_sigma = 0.6
    args.lr = 0.001

    # plot_weight_histogram(parameters)

    inner_loop_ppo(
        args,
        args.lr,
        num_steps=10000,
        num_updates=1000,
        inst_on=False,
        visualize=False
    )


if __name__ == "__main__":
    main()
