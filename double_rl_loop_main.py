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

EPISODE_LENGTH = 1000
HAZARD_PUNISHMENT = 100.0
HAZARD_PUNISHMENT_4_POLICY = 10.0 # 100.0 # 500.0
ACTIVATION_DISCOUNT = 0.3
REWARD_SCALE = 50.0

config = {
    'num_steps': EPISODE_LENGTH,
    'hazards_cost': 1,
    'constrain_indicator': True,
    'observe_goal_lidar': True,
    'observe_buttons': True,
    'observe_box_lidar': True,
    'observe_circle': True,  # Observe the origin with a lidar
    'observe_button_goal_lidar': True,

    'lidar_max_dist': 1,
    'goal_lidar_max': 7,
    'hazard_lidar_max': 1,

    'lidar_num_bins': 16,
    'task': 'goal',  # Task definition
    'buttons_num': 4,
    # 'buttons_locations': [(-1.5, -1.5), (1.5, 1.5), (-1.5, 1.5), (1.5, -1.5)],
    'goal_size': 0.3,
    'goal_keepout': 0.305,
    'hazards_size': 0.2,
    'hazards_keepout': 0.18,
    'robot_base': 'xmls/point.xml',
    'robot_locations': [(0, 0)],
    'robot_rot': 0 * 3.1415,
    'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
    'placements_extents': [-2, -2, 2, 2],
    'hazards_num': 24,
    'hazards_locations': [(-1, -1), (1, 1), (-1, 1), (1, -1),  # inner corners
                          (0.0, 1), (0.0, -1), (1, 0.0), (-1, 0.0),  # inner cross
                          (-2, -2), (2, -2,), (2, 2), (-2, 2),  # outer corners
                          (0, -2), (0, 2), (-2, 0), (2, 0),  # outer cross
                          (-1, -2), (1, 2), (1, -2), (-1, 2),  # outer horizontal hole fillers
                          (-2, -1), (-2, 1), (2, 1), (2, -1),  # outer vertical hole fillers
                          (-0.5, 1), (-1.5, -1), (1.5, 1), (0.5, -1),  # side vertical path stoppers
                          (0, 1.5), (0, -1.5), (-1, -0.5), (1, 0.5),  # ide horizontal path stoppers
                          (0, 0), (0, -0.5),
                          ],  # center
    'vases_num': 0,
    'constrain_hazards': True,
    'observe_hazards': True,
    'observe_vases': False}

ENV_NAME = 'SafexpCustomEnvironmentGoal1-v0'
register(id=ENV_NAME,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config})

## Register all goals
ENV_NAME_BUTTON_EASY = 'SafexpCustomEnvironmentButtons0-v0'
config_button_easy = deepcopy(config)
config_button_easy['button_goal_idx'] = None
config_button_easy['task'] = "button"
config_button_easy['hazards_num'] = 8
config_button_easy['hazards_locations'] = []
config_button_easy['hazards_keepout'] = 0.6
config_button_easy['placements_extents'] = [-3, -3, 3, 3]
config_button_easy['robot_locations'] = []
config_button_easy['robot_rot'] = None
register(id=ENV_NAME_BUTTON_EASY,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config_button_easy})


ENV_NAME_BUTTON_HARDER = 'SafexpCustomEnvironmentButtons1-v0'
config_button_harder = deepcopy(config)
config_button_harder['button_goal_idx'] = None
config_button_harder['task'] = "button"
config_button_harder['hazards_num'] = 12
register(id=ENV_NAME_BUTTON_HARDER,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config_button_harder})


ENV_NAME_BOX = 'SafexpCustomEnvironmentBox-v0'
config_box = deepcopy(config)
config_box['num_steps'] = EPISODE_LENGTH * 2
config_box['button_goal_idx'] = 0
config_box['task'] = "push"
config_box['hazards_num'] = 20
config_box['hazards_locations'] = [
                                    (-3, 2.5), (-3, 1.25), (-3, 0), (-3, -1.25), (-3, -2.5),
                                    (-2, 2.5), (-2, 1.25), (-2, 0), (-2, -1.25), (-2, -2.5),
                                    (-1, 2.5), (-1, 1.25), (-1, 0), (-1, -1.25), (-1, -2.5),
                                    (0, 2.5), (0, 1.25), (0, 0), (0, -1.25), (0, -2.5),
                                    ]
config_box['robot_locations'] = [(-4, 0), (-4, -2), (-4, 2)]
config_box['goal_placements'] = [(3, -2, 4, 2)]
config_box['box_placements'] = [(1, -2, 2, -1), (1, 1, 2, 2)]
config_box['buttons_num'] = 1
config_box['buttons_locations'] = [(3, -3)]
register(id=ENV_NAME_BOX,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config_box})


ENV_NAME_BOX_NO_HAZARDS = 'SafexpCustomEnvironmentBox-v1'
config_box_no_haz = deepcopy(config_box)
config_box_no_haz['num_steps'] = EPISODE_LENGTH * 2
config_box_no_haz['button_goal_idx'] = 0
config_box_no_haz['task'] = "push"
config_box_no_haz['hazards_num'] = 0
config_box_no_haz['hazards_locations'] = [
                                    (-3, 2.5), (-3, 1.25), (-3, 0), (-3, -1.25), (-3, -2.5),
                                    (-2, 2.5), (-2, 1.25), (-2, 0), (-2, -1.25), (-2, -2.5),
                                    (-1, 2.5), (-1, 1.25), (-1, 0), (-1, -1.25), (-1, -2.5),
                                    (0, 2.5), (0, 1.25), (0, 0), (0, -1.25), (0, -2.5),
                                    ]
config_box_no_haz['robot_locations'] = [(-4, 0), (-4, -2), (-4, 2)]
config_box_no_haz['goal_placements'] = [(3, -2, 4, 2)]
config_box_no_haz['box_placements'] = [(1, -2, 2, -1), (1, 1, 2, 2)]
config_box_no_haz['buttons_num'] = 1
config_box_no_haz['buttons_locations'] = [(-3, -3)]


register(id=ENV_NAME_BOX_NO_HAZARDS,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config_box_no_haz})

NP_RANDOM, _ = seeding.np_random(None)
NUM_PROC = 24

PHASE_LENGTH = 1000


# Phase enum
class TrainPhases(Enum):
    POLICY_TRAIN_PHASE = 0
    INSTINCT_TRAIN_PHASE = 1


class PolicyPhase(Enum):
    TRAINED_POLICY = 0
    RANDOM_POLICY = 1


def phase_shifter(iteration, phase_length=100, num_phases=2):
    return (iteration // phase_length) % num_phases


def compare_two_models(model1, model2):
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        if p1.data.ne(p2.data).sum() > 0:
            return False
    return True


def plot_weight_histogram(parameters):
    flattened_params = []
    for p in parameters:
        flattened_params.append(p.flatten())

    params_stacked = np.concatenate(flattened_params)
    plt.hist(params_stacked, bins=300)
    plt.show()


def policy_instinct_combinator(policy_actions, instinct_outputs):
    # Split the shape
    instinct_half_shape = int(instinct_outputs.shape[1] - 1)

    # Test if the shapes work
    assert instinct_half_shape == policy_actions.shape[0] or len(policy_actions.shape) == len(instinct_outputs.shape), \
        "Wrong matrices shapes"
    if len(policy_actions.shape) > 1:
        assert policy_actions.shape[0] == instinct_outputs.shape[0], "Wrong matrices shapes"

    # Divert the control from action in the instinct
    instinct_control = instinct_outputs[:, instinct_half_shape:]
    instinct_action = instinct_outputs[:, :instinct_half_shape]  # Bring tanh(x) to [0, 1] range
    instinct_control = (instinct_control + 1) * 0.5

    # Control the policy and instinct outputs
    ctrl_policy_actions = instinct_control * policy_actions
    ctrl_instinct_actions = (1 - instinct_control) * instinct_action

    # Combine the two controlled outputs
    combined_action = ctrl_instinct_actions + ctrl_policy_actions
    return combined_action, instinct_control


def reward_cost_combinator(reward_list, infos, num_processors, i_control):
    # Count the cost
    violation_cost = torch.Tensor([[0]] * num_processors)
    modded_reward_list = []
    # Add a regularization clause to discourage instinct to activate if not necessary
    for i_control_idx in range(len(i_control)):
        i_control_on_idx = i_control[i_control_idx]
        i_reward = reward_list[i_control_idx]
        safety = (1 - infos[i_control_idx]['cost'] * HAZARD_PUNISHMENT)
        instinct_activation = (1 - torch.clamp(torch.mean(i_control_on_idx), 0.0, 1.0).item())
        violation_cost[i_control_idx][0] = safety * (1 - instinct_activation * ACTIVATION_DISCOUNT) * (
                    i_reward * REWARD_SCALE)

        modded_reward_list.append([i_reward - (infos[i_control_idx]['cost']*HAZARD_PUNISHMENT_4_POLICY)])

    # Normalize the cost to the episode length
    violation_cost /= float(EPISODE_LENGTH)
    modded_reward_list = torch.tensor(modded_reward_list)
    # modded_reward_list /= float(EPISODE_LENGTH)

    return modded_reward_list, violation_cost


def make_instinct_input(obs, action):
    i_obs = torch.cat([obs, action], dim=1)
    # for i_obs_n in i_obs:  # Blind instinct to goal sensors
    #     i_obs_n[51:67] = torch.zeros(16)
    return i_obs


class EvalActorCritic:
    def __init__(self, policy, instinct):
        self.instinct = instinct
        self.policy = policy

    @property
    def recurrent_hidden_state_size(self):
        return self.policy.recurrent_hidden_state_size

    def act(self, obs, eval_recurrent_hidden_states, eval_masks, deterministic=True):
        _, a, _, _ = self.policy.act(obs, eval_recurrent_hidden_states, eval_masks, deterministic=deterministic)
        i_obs = make_instinct_input(obs, a)
        _, ai, _, _ = self.instinct.act(i_obs, eval_recurrent_hidden_states, eval_masks, deterministic=deterministic)
        total_action, i_control = policy_instinct_combinator(a, ai)
        return None, total_action, i_control, None


def inner_loop_ppo(
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

    env_name = ENV_NAME  # "Safexp-PointGoal1-v0"
    envs = make_vec_envs(env_name, np.random.randint(2 ** 32), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    eval_envs = make_vec_envs(env_name, np.random.randint(2 ** 32), 1,
                              args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

    actor_critic_policy = init_default_ppo(envs, log(args.init_sigma))

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
                                      actor_critic_policy.recurrent_hidden_state_size)

    rollouts_cost = RolloutStorage(num_steps, NUM_PROC,
                                   inst_obs_shape, inst_action_space,
                                   actor_critic_instinct.recurrent_hidden_state_size)

    obs = envs.reset()
    i_obs = torch.cat([obs, torch.zeros((NUM_PROC, envs.action_space.shape[0]))],
                      dim=1)  # Add zero action to the observation
    rollouts_rewards.obs[0].copy_(obs)
    rollouts_rewards.to(device)
    rollouts_cost.obs[0].copy_(i_obs)
    rollouts_cost.to(device)

    fitnesses = []
    best_fitness_so_far = float("-Inf")
    is_instinct_training = False
    for j in range(num_updates):
        is_instinct_training_old = is_instinct_training
        is_instinct_training = phase_shifter(j, PHASE_LENGTH,
                                             len(TrainPhases)) == TrainPhases.INSTINCT_TRAIN_PHASE.value
        is_instinct_deterministic = not is_instinct_training
        is_policy_deterministic = not is_instinct_deterministic
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                # (value, action, action_log_probs, rnn_hxs), (instinct_value, instinct_action, instinct_outputs_log_prob, i_rnn_hxs), final_action
                value, action, action_log_probs, recurrent_hidden_states = actor_critic_policy.act(
                    rollouts_rewards.obs[step],
                    rollouts_rewards.recurrent_hidden_states[step],
                    rollouts_rewards.masks[step],
                    deterministic=is_policy_deterministic
                )
                instinct_value, instinct_action, instinct_outputs_log_prob, instinct_recurrent_hidden_states = actor_critic_instinct.act(
                    rollouts_cost.obs[step],
                    rollouts_cost.recurrent_hidden_states[step],
                    rollouts_cost.masks[step],
                    deterministic=is_instinct_deterministic,
                )

            # Combine two networks
            final_action, i_control = policy_instinct_combinator(action, instinct_action)
            obs, reward, done, infos = envs.step(final_action)
            # envs.render()

            reward, violation_cost = reward_cost_combinator(reward, infos, NUM_PROC, i_control)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts_rewards.insert(obs, recurrent_hidden_states, action,
                                    action_log_probs, value, reward, masks, bad_masks)
            i_obs = torch.cat([obs, action], dim=1)
            rollouts_cost.insert(i_obs, instinct_recurrent_hidden_states, instinct_action, instinct_outputs_log_prob,
                                 instinct_value, violation_cost, masks, bad_masks)

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

        if not is_instinct_training:
            print("training policy")
            # Policy training phase
            p_before = deepcopy(agent_instinct.actor_critic)
            value_loss, action_loss, dist_entropy = agent_policy.update(rollouts_rewards)
            val_loss_i, action_loss_i, dist_entropy_i = 0, 0, 0
            p_after = deepcopy(agent_instinct.actor_critic)
            assert compare_two_models(p_before, p_after), "policy changed when it shouldn't"
        else:
            print("training instinct")
            # Instinct training phase
            value_loss, action_loss, dist_entropy = 0, 0, 0
            p_before = deepcopy(agent_policy.actor_critic)
            val_loss_i, action_loss_i, dist_entropy_i = agent_instinct.update(rollouts_cost)
            p_after = deepcopy(agent_policy.actor_critic)
            assert compare_two_models(p_before, p_after), "policy changed when it shouldn't"

        rollouts_rewards.after_update()
        rollouts_cost.after_update()

        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms

        fits, info = evaluate(EvalActorCritic(actor_critic_policy, actor_critic_instinct), ob_rms, eval_envs, NUM_PROC,
                              reward_cost_combinator, device, instinct_on=inst_on,
                              visualise=visualize)
        instinct_reward = info['instinct_reward']
        eval_hazard_collisions = info['hazard_collisions']
        print(
            f"Step {j}, Fitness {fits.item()}, value_loss = {value_loss}, action_loss = {action_loss}, "
            f"dist_entropy = {dist_entropy}")
        print(
            f"Step {j}, Instinct reward {instinct_reward}, value_loss instinct = {val_loss_i}, action_loss instinct= {action_loss_i}, "
            f"dist_entropy instinct = {dist_entropy_i} hazard_collisions = {eval_hazard_collisions}")
        print("-----------------------------------------------------------------")

        # Tensorboard logging
        log_writer.add_scalar("fitness", fits.item(), j)
        log_writer.add_scalar("value loss", value_loss, j)
        log_writer.add_scalar("action loss", action_loss, j)
        log_writer.add_scalar("dist entropy", dist_entropy, j)

        log_writer.add_scalar("cost/instinct_reward", instinct_reward, j)
        log_writer.add_scalar("cost/hazard_collisions", eval_hazard_collisions, j)
        log_writer.add_scalar("value loss instinct", val_loss_i, j)
        log_writer.add_scalar("action loss instinct", action_loss_i, j)
        log_writer.add_scalar("dist entropy instinct", dist_entropy_i, j)

        fitnesses.append(fits)
        if fits.item() > best_fitness_so_far:
            best_fitness_so_far = fits.item()
            torch.save(actor_critic_policy, join(save_dir, "model_rl_policy.pt"))
            torch.save(actor_critic_instinct, join(save_dir, "model_rl_instinct.pt"))
        if is_instinct_training != is_instinct_training_old:
            torch.save(actor_critic_policy, join(save_dir, f"model_rl_policy_update_{j}.pt"))
            torch.save(actor_critic_instinct, join(save_dir, f"model_rl_instinct_update_{j}.pt"))
        torch.save(actor_critic_policy, join(save_dir, "model_rl_policy_latest.pt"))
        torch.save(actor_critic_instinct, join(save_dir, "model_rl_instinct_latest.pt"))
    return (fitnesses[-1]), 0, 0


def main():
    args = get_args()
    print("start the train function")

    args.init_sigma = 0.6
    args.lr = 0.001

    # plot_weight_histogram(parameters)
    exp_save_dir = get_experiment_save_dir(args)

    inner_loop_ppo(
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
