import copy
import statistics
from math import log

import torch
import numpy as np
import gym
from gym.envs.registration import register
import safety_gym


from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_default_ppo, init_ppo, PolicyWithInstinct
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args
from main_es import get_model_weights

#config = {
#    'robot_base': 'xmls/point.xml',
#    #'observe_sensors': False,
#    'observe_goal_lidar': True,
#    'constrain_hazards': True,
#    'hazards_num': 4,
#}

HAZARD_LOC_PARAM = 1
HLP = HAZARD_LOC_PARAM

GOAL_LOC_PARAM = 1.8
GLP = GOAL_LOC_PARAM
GOALS = [(-GLP, -GLP), (GLP, GLP), (GLP, -GLP), (-GLP, GLP)]
config = {'num_steps': 200,
          'observe_goal_lidar': False,
          'observe_box_lidar': False,
          'observe_qpos': True,
          'observe_hazards': True,
          'goal_locations': [(-GLP, -GLP)],
          'robot_locations': [(0, 0)],
          'robot_rot': 0 * 3.1415,
          'lidar_max_dist': 1,
          'task': 'goal',
          'goal_size': 0.1,
          'goal_keepout': 0.305,
          'hazards_size': 0.4,
          'hazards_keepout': 0.18,
          'hazards_num': 4,
          'hazards_cost': -10.0,
          'hazards_locations': [(-HLP, -HLP), (HLP, HLP), (HLP, -HLP), (-HLP, HLP)],
          'constrain_hazards': True,
          'robot_base': 'xmls/point.xml',
          'sensors_obs': ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
          'lidar_num_bins': 8,
          'placements_extents': [-2, -2, 2, 2]}

#register(id='SafexpCustomEnvironment-v0',
#         entry_point='safety_gym.envs.mujoco:Engine',
#         kwargs={'config': config})

CUSTOM_ENV = 'SafexpCustomEnvironment-v0'
#ENV_NAME = "Safexp-PointGoal0-v0"
ENV_NAME = CUSTOM_ENV
NUM_PROC = 1

def register_set_goal(goal_idx):
    goal = GOALS[goal_idx]
    config['goal_locations'] = [goal]

    env_name = f'SafexpCustomEnvironmentGoal{goal_idx}-v0'

    try:
        register(id=env_name,
                 entry_point='safety_gym.envs.mujoco:Engine',
                 kwargs={'config': config})
    except:
        pass

    return env_name


def apply_from_list(weights, model : PolicyWithInstinct):
    to_params_dct = model.get_evolvable_params()

    for ptensor, w in zip(to_params_dct, weights):
        w_tensor = torch.Tensor(w)
        ptensor.data.copy_(w_tensor)

def inner_loop_ppo(
    weights,
    args,
    learning_rate,
    num_steps,
    num_updates,
    run_idx,
    inst_on,
    visualize
):

    torch.set_num_threads(1)
    device = torch.device("cpu")

    env_name = register_set_goal(run_idx)

    envs = make_vec_envs(env_name, np.random.randint(2**32), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

    actor_critic = init_ppo(envs, log(args.init_sigma))
    actor_critic.to(device)

    # apply the weights to the model
    apply_from_list(weights, actor_critic)


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
    violation_cost = 0

    for j in range(num_updates):

        episode_step_counter = 0
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states, (final_action, _) = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])
            # Obser reward and next obs
            obs, reward, done, infos = envs.step(final_action)
            episode_step_counter += 1

            # Count the cost
            total_reward = reward
            for info in infos:
                violation_cost += info['cost']
                total_reward -= info['cost']

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, total_reward, masks, bad_masks)

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

        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device, instinct_on=inst_on, visualise=visualize)
        if j % 1 == 0:
            print(fits[-1])
        fitnesses.append(fits)
        torch.save(actor_critic, "model_rl.pt")
    return (fitnesses[-1]), 0, 0


if __name__ == "__main__":
    args = get_args()
    env_name = register_set_goal(0)

    envs = make_vec_envs(
        env_name, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )
    print("start the train function")
    #parameters = torch.load("/Users/djrg/code/instincts/modular_rl_safety_gym/trained_models/pulled_from_server/es_testing/2ba1a655bd_0/saved_weights_gen_238.dat")

    args.init_sigma = 1.0
    args.lr = 0.001
    blueprint_model = init_ppo(envs, log(args.init_sigma))
    parameters = get_model_weights(blueprint_model)
    parameters.append(np.array([args.lr]))

    fitness = inner_loop_ppo(
        parameters,
        args,
        args.lr,
        num_steps=40000,
        num_updates=200,
        run_idx=0,
        inst_on=True,
        visualize=False
    )


