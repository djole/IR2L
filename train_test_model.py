import copy
import statistics
from math import log

import torch
import numpy as np

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_ppo, PolicyWithInstinct
from a2c_ppo_acktr.storage import RolloutStorage
from arguments import get_args

from env_util import register_set_goal

NUM_PROC = 1

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
    envs,
):

    torch.set_num_threads(1)
    device = torch.device("cpu")
    #env_name = register_set_goal(run_idx)

    #envs = make_vec_envs(env_name, np.random.randint(2**32), NUM_PROC,
    #                     args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
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

        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device)
        fitnesses.append(fits)

    return (fitnesses[-1]), 0, 0


if __name__ == "__main__":
    args = get_args()
    env_name = register_set_goal(0)

    envs = make_vec_envs(
        env_name, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )
    print("start the train function")
    init_sigma = args.init_sigma
    init_model = init_ppo(envs, log(init_sigma))
    #init_model = torch.load("saved_model.pt")

    fitness = inner_loop_ppo(
        init_model,
        args,
        args.lr,
        num_steps=40000,
        num_updates=150,
    )

    print(fitness)