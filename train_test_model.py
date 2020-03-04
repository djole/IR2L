""" Module for training functions """
import copy
from collections import deque

import numpy as np
import torch
from gym.utils import seeding

import navigation_2d
from model import ControllerCombinator, ControllerNonParametricCombinator
from a2c_ppo_acktr.model import Policy
from a2c_ppo_acktr import algo
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr import utils
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.envs import make_vec_envs
from gym.envs.registration import register

EPS = np.finfo(np.float32).eps.item()

ENV_NAME = "Navigation2d-v0"
NUM_PROC = 1

def select_model_action(model, state):
    state_ = state
    state_ = torch.from_numpy(state_).float()
    # dist_2_nogo = torch.tensor([dist_2_nogo])
    # model_input = torch.cat([position, dist_2_nogo])
    action, action_log_prob, debug_info = model(state_)
    # return action.item()
    return action.detach().numpy(), action_log_prob, debug_info


def update_policy(optimizer, args, rewards, log_probs):
    R = 0
    policy_loss = []
    returns = []
    for r in rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / (returns.std() + EPS)
    for log_prob, R in zip(log_probs, returns):
        policy_loss.append(-log_prob * R)
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    optimizer.step()


def episode_rollout(model, env, vis=False):

    # new_task = env.sample_tasks()
    # env.reset_task(new_task[goal_index])

    state = env.reset()
    cummulative_reward = 0
    rewards = []
    action_log_probs = []

    ######
    # Visualisation elements
    action_records = list()
    path_records = list()
    if vis:
        path_records.append(env._state)
    debug_info_records = list()
    # ---------------------

    while True:
        action, action_log_prob, debug_info = select_model_action(model, state)
        action = action.flatten()
        state, reward, done, infos = env.step(action)
        cummulative_reward += reward

        rewards.append(reward)
        action_log_probs.append(action_log_prob)
        ######
        # Visualisation elements
        if vis:
            action_records.append(action)
            path_records.append(env._state)
            debug_info_records.append(debug_info)
        # ---------------------
        if done:
            env.reset()
            break

    return (
        cummulative_reward,
        infos['reached'],
        (rewards, action_log_probs),
        (action_records, path_records, debug_info_records, env._goal),
    )


def train_maml_like_ppo(
        init_model,
        args,
        learning_rate,
        num_episodes=20,
        num_updates=1,
        vis=False,
        run_idx=0,):

    num_steps = num_episodes * 100
    # Register the environment
    try:
        register(
            id="Navigation2d-v0",
            entry_point="navigation_2d:Navigation2DEnv",
            max_episode_steps=200,
            reward_threshold=0.0,
            kwargs={
                "rm_nogo": args.rm_nogo,
                "reduced_sampling": args.reduce_goals,
                "dist_to_nogo": args.dist_to_nogo,
                "nogo_large": args.large_nogos,
                "all_dist_to_nogo": args.all_dist_to_nogo,
            },
        )
    except:
        pass

    torch.set_num_threads(1)
    device = torch.device("cpu")

    envs = make_vec_envs(ENV_NAME, seeding.create_seed(None), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    raw_env = navigation_2d.unpeele_navigation_env(envs, 0)

    #raw_env.set_arguments(args.rm_nogo, args.reduce_goals, True, args.large_nogos)
    new_task = raw_env.sample_tasks(run_idx)
    raw_env.reset_task(new_task[0])


   # actor_critic = Policy(
   #     envs.observation_space.shape,
   #     envs.action_space,
   #     base_kwargs={'recurrent': args.recurrent_policy})
    actor_critic = copy.deepcopy(init_model)
    actor_critic.to(device)


    agent = algo.PPO(
            actor_critic,
            args.clip_param,
            args.ppo_epoch,
            args.num_mini_batch,
            args.value_loss_coef,
            args.entropy_coef,
            lr=learning_rate,
            eps=args.eps,
            max_grad_norm=args.max_grad_norm)

    rollouts = RolloutStorage(num_steps, NUM_PROC,
                              envs.observation_space.shape, envs.action_space,
                              actor_critic.recurrent_hidden_state_size)

    obs = envs.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(device)

    fitnesses = []

    instinct_control_sum = 0
    offending_steps_num = 0
    for j in range(num_updates):
        #if args.use_linear_lr_decay:
        #    # decrease learning rate linearly
        #    utils.update_linear_schedule(
        #        agent.optimizer, j, num_updates,
        #        agent.optimizer.lr if args.algo == "acktr" else args.lr)
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, (final_action, ctrl) = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
                instinct_control_sum += ctrl

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(final_action)

            if done[0]:
                offending_steps_num += len(infos[0]["offending"])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()


        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()


        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms
        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device)
        fitnesses.append(fits - (offending_steps_num*10))
    return fitnesses[-1], info[0]['reached'], (instinct_control_sum/(num_steps * num_updates))

def train_maml_like(
    init_model,
    args,
    learning_rate,
    num_episodes=20,
    num_updates=1,
    vis=False,
    run_idx=0,
):
    env = navigation_2d.Navigation2DEnv(
        rm_nogo=args.rm_nogo, reduced_sampling=args.reduce_goals, sample_idx=run_idx
    )
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator) or isinstance(
        model, ControllerNonParametricCombinator
    ):
        optimizer = torch.optim.Adam(model.get_combinator_params(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rewards = []
    action_log_probs = []

    fitness_list = []
    ### evaluate for the zero updates
    if vis:
        model.controller.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    for u_idx in range(num_updates):
        avg_exploration_fitness = 0
        ### Train
        model.controller.deterministic = False
        for ne in range(num_episodes):
            exploration_fitness, reached, (
                rewards_,
                action_log_probs_,
            ), _ = episode_rollout(model, env, False)
            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)
            avg_exploration_fitness = (
                exploration_fitness + ne * avg_exploration_fitness
            ) / (ne + 1)

        # Reduce the learning rate of the optimizer by half in the first iteration
        if u_idx > 0:
            new_learning_rate = learning_rate / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.controller.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        fitness_list.append(fitness)

    rm_exp_fit = args.rm_nogo or args.rm_exploration_fit
    avg_exploration_fitness = 0.0 if rm_exp_fit else avg_exploration_fitness
    avg_exploitation_fitness = sum(fitness_list) / num_updates
    ret_fit = (
        fitness_list if vis else avg_exploitation_fitness + avg_exploration_fitness
    )
    return ret_fit, reached, vis_info


def train_maml_like_for_trajectory(
    init_model, args, learning_rate, num_episodes=20, num_updates=1, vis=False
):
    # TODO Remove this function, this is bad programming
    assert False, "Obsolete piece of code, remove it!"
    env = navigation_2d.Navigation2DEnv(args.rm_nogo, args.reduce_goals)
    new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = copy.deepcopy(init_model)

    optimizer = None
    if isinstance(model, ControllerCombinator):
        optimizer = torch.optim.Adam(model.get_combinator_params(), lr=learning_rate)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    rewards = []
    action_log_probs = []

    fitness_list = []
    ### evaluate for the zero updates
    vis_info_collection = []
    if vis:
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        vis_info_collection.append(vis_info)
        fitness_list.append(fitness)

    for u_idx in range(num_updates):
        ### Train
        model.deterministic = False
        for _ in range(num_episodes):
            _, reached, (rewards_, action_log_probs_), vis_info = episode_rollout(
                model, env, True
            )
            vis_info_collection.append(vis_info)

            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)

        # Reduce the learning rate of the optimizer by half in the first iteration
        if u_idx == 0 and vis:
            new_learning_rate = args.lr / 2.0
            for param_group in optimizer.param_groups:
                param_group["lr"] = new_learning_rate

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=vis)
        vis_info_collection.append(vis_info)
        fitness_list.append(fitness)

    ret_fit = fitness_list if vis else fitness_list[-1]
    return ret_fit, reached, vis_info_collection
