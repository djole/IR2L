import time
from collections import deque
import copy

import torch

from a2c_ppo_acktr import algo, utils
from arguments import get_args
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.model import init_ppo, init_default_ppo
from a2c_ppo_acktr.storage import RolloutStorage
from a2c_ppo_acktr.evaluation import evaluate
from gym.envs.registration import register
from gym.utils import seeding

import navigation_2d
from visualisations.vis_path import vis_path

# register(
#    id="Navigation2d-v0",
#    entry_point="navigation_2d:Navigation2DEnv",
#    max_episode_steps=200,
#    reward_threshold=0.0,
# )

ENV_NAME = "Navigation2d-v0"
NUM_PROC = 1

def train_maml_like_ppo_(
    init_model,
    args,
    learning_rate,
    num_episodes=20,
    num_updates=1,
    vis=False,
    run_idx=0,
    use_linear_lr_decay=False,
):
    num_steps = num_episodes * 100

    torch.set_num_threads(1)
    device = torch.device("cpu")

    envs = make_vec_envs(ENV_NAME, seeding.create_seed(None), NUM_PROC,
                         args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    raw_env = navigation_2d.unpeele_navigation_env(envs, 0)

    # raw_env.set_arguments(args.rm_nogo, args.reduce_goals, True, args.large_nogos)
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

    for j in range(num_updates):

        # if args.use_linear_lr_decay:
        #    # decrease learning rate linearly
        #    utils.update_linear_schedule(
        #        agent.optimizer, j, num_updates,
        #        agent.optimizer.lr if args.algo == "acktr" else args.lr)
        min_c_rew = float("inf")
        vis = []
        offending = []
        for step in range(num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob, recurrent_hidden_states = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            if done[0]:
                c_rew = infos[0]["cummulative_reward"]
                vis.append((infos[0]['path'], infos[0]['goal']))
                offending.extend(infos[0]['offending'])
                if c_rew < min_c_rew:
                    min_c_rew = c_rew
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
        print(f"fitness {fits} update {j+1}")
        if (j+1) % 1 == 0:
            vis_path(vis, eval_path_rec=info['path'], offending=offending)
        fitnesses.append(fits)

    return fitnesses[-1], info[0]['reached'], None


if __name__ == "__main__":
    args = get_args()

    register(
        id='Navigation2d-v0',
        entry_point='navigation_2d:Navigation2DEnv',
        max_episode_steps=200,
        reward_threshold=0.0,
        kwargs={'rm_nogo': args.rm_nogo,
                'reduced_sampling': args.reduce_goals,
                'rm_dist_to_nogo': args.rm_dist_to_nogo,
                'nogo_large': args.large_nogos}
    )

    envs = make_vec_envs(
        ENV_NAME, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )
    print("start the train function")
    import math

    ###### Load the saved model and the learning rate ######
    load_m = torch.load(
        "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/2_deterministic_goals/small_zones_NOdistance2zones_PPO/dist2nogo_individual_CTRL_731.pt"
    )
    init_model = load_m[0]
    learning_rate = load_m[1]
    #args.lr = learning_rate
    init_sigma = args.init_sigma

    from math import log
    fitness = train_maml_like_ppo_(
        #init_model,
        init_ppo(envs, log(init_sigma)),
        args,
        args.lr,
        num_episodes=40,
        num_updates=200,
        run_idx=0,
    )
    print(fitness)
