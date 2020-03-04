import re
from os.path import join
from os import listdir

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
from math import log

import navigation_2d
from visualisations.vis_path import vis_path

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
    gen_idx=0,
    use_linear_lr_decay=False,
    inst_on=True,
    start_state=(0, 0)
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
    raw_env.set_start_state(start_state)

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
                value, action, action_log_prob, recurrent_hidden_states, (final_action, _) = actor_critic.act(
                    rollouts.obs[step], rollouts.recurrent_hidden_states[step],
                    rollouts.masks[step], instinct_on=inst_on)

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(final_action)
            if done[0]:
                c_rew = infos[0]["cummulative_reward"]
                vis.append((infos[0]['path'], infos[0]['goal']))
                offending.extend(infos[0]['offending'])
                if c_rew < min_c_rew:
                    min_c_rew = c_rew

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor([[0.0] if 'bad_transition' in info.keys() else [1.0] for info in infos])
            rollouts.insert(obs, recurrent_hidden_states, action,
                            action_log_prob, value, reward, masks, bad_masks)

        with torch.no_grad():
            next_value = actor_critic.get_value(
                rollouts.obs[-1], rollouts.recurrent_hidden_states[-1],
                rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.use_gae, args.gamma,
                                 args.gae_lambda, args.use_proper_time_limits)

        agent.update(rollouts)
        rollouts.after_update()

        ob_rms = utils.get_vec_normalize(envs)
        if ob_rms is not None:
            ob_rms = ob_rms.ob_rms

        fits, info = evaluate(actor_critic, ob_rms, envs, NUM_PROC, device)
        print(f"fitness {fits} update {j+1}")
        if (j+1) % 1 == 0:
            vis_path(vis, saveidx=f"pathplot_gen_{gen_idx}_goal_{run_idx}", eval_path_rec=info[0]['path'], offending=offending)
            #vis_heatmap(model=actor_critic.instinct, env=raw_env, alldists=args.all_dist_to_nogo)
            #vis_instinct_action(model=actor_critic.instinct, env=raw_env, alldists=args.all_dist_to_nogo)
            print(f"Number of offending steps = {len(offending)}")
        fitnesses.append((fits, -len(offending)*10))

    return fitnesses[-1], info[0]['reached'], (actor_critic, info[0]['path'][-1])


def evaluate_model(model_file, args, gen_idx, outputfile):

    ###### Load the saved model and the learning rate ######
    load_m = torch.load(model_file)

    saved_model = load_m[0]
    learning_rate = load_m[1]
    args.lr = learning_rate
    print(learning_rate)

    init_model = saved_model
    print(torch.exp(init_model.policy.dist.logstd._bias.data))

    start_state = (0.0, 0.0)
    fitness1sum, fitness2sum = 0, 0
    for g in [0, 1, 2, 3]:
        (fitness1, fitness2), reached, (_, _) = train_maml_like_ppo_(
            init_model,
            args,
            args.lr,
            num_episodes=40,
            num_updates=1,
            run_idx=g,
            gen_idx=gen_idx,
            inst_on=True,
            start_state=start_state
        )
        fitness1sum += fitness1
        fitness2sum += fitness2

    outputfile.write(f"Generation -> {gen_idx}\n")
    outputfile.write(f"Distance fitness -> {fitness1sum}\n")
    outputfile.write(f"NOGO fitness -> {fitness2sum}\n")
    outputfile.write(f"TOTAL fitness -> {fitness1sum+fitness2sum}\n")
    outputfile.write("----------------------------\n")

def match_part_vs_genidx(log_file_list):
    gen_idx__ptrn = re.compile("=============== Generation index")
    int_ptrn = re.compile("\d+")
    part_ptrn = re.compile("part\d+")

    log_file_indexes = {}

    for log_file in log_file_list:
        part_token = part_ptrn.findall(log_file)[0]
        log_file_indexes[part_token] = []
        with open(log_file, "r") as src_file:
            for log_line in src_file:
                best_line = gen_idx__ptrn.search(log_line)
                if best_line is not None:
                    val = int_ptrn.findall(log_line)[0]
                    log_file_indexes[part_token].append(int(val))

    return log_file_indexes

def main():
    args = get_args()
    register(
        id='Navigation2d-v0',
        entry_point='navigation_2d:Navigation2DEnv',
        max_episode_steps=200,
        reward_threshold=0.0,
        kwargs={'rm_nogo': args.rm_nogo,
                'reduced_sampling': args.reduce_goals,
                'dist_to_nogo': args.dist_to_nogo,
                'nogo_large': args.large_nogos,
                'all_dist_to_nogo': args.all_dist_to_nogo}
    )

    root_dir = "/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/4goals_lidar_ctrl_noCoordiantes/balance_plot_visualisation"
    # list all *log files
    log_files = []
    evolution_dirs = []
    for file in listdir(root_dir):
        if file.endswith(".log"):
            log_files.append(join(root_dir, file))
        if file.startswith("evolution"):
            evolution_dirs.append(join(root_dir, file))

    log_files.sort()
    evolution_dirs.sort()
    part_vs_gen_idx_dict = match_part_vs_genidx(log_files)

    absolute_gen_idx = 0
    outputfile = open(join(root_dir, "output.txt"), "w")
    for part, gens in part_vs_gen_idx_dict.items():
        evol_dir = [ed for ed in evolution_dirs if part in ed]
        evol_dir = evol_dir[0]
        gens.sort()

        for g in gens:
            model_file = join(evol_dir, str(g), f"individual_{g}.pt")
            print(model_file)
            evaluate_model(model_file, args, absolute_gen_idx, outputfile)
            absolute_gen_idx += 1




    # run the model through an evaluation/plot function
    # Record two fitnesses
    # Plot trajectoriers for each goal
    pass


if __name__ == "__main__":
    main()
