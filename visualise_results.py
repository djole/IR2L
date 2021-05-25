from double_rl_loop_main import EvalActorCritic
from a2c_ppo_acktr.model import init_default_ppo, Policy, custom_weight_init
from math import log
import torch
import numpy as np
import gym
import safety_gym_mod
from gym.envs.registration import register

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
# from a2c_ppo_acktr.model import init_default_ppo, Policy, custom_weight_init
from arguments import get_args
from double_rl_loop_main import reward_cost_combinator, config_box  # , config1, config2, config3, config4
from copy import deepcopy
import pickle
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

env_name = 'SafexpCustomEnvironmentGoal1Test-v0'
register(id=env_name,
         entry_point='safety_gym_mod.envs.mujoco:Engine',
         kwargs={'config': config_box})


def visualise_values_over_path(data_list):
    path = [(dt['pos_x'], dt['pos_y']) for dt in data_list]
    # instinct_rewards = [dt['instinct reward'] for dt in data_list]
    # policy_rewards = [dt['policy reward'] for dt in data_list]
    hazards_pos = [dt['hazards_pos'] for dt in data_list][0]
    buttons_pos = [dt['button_pos'] for dt in data_list][0]
    box_pos = [dt['box_pos'] for dt in data_list]
    goal_pos = [dt['goal_pos'] for dt in data_list]
    instinct_reg = [dt['instinct regulation'] for dt in data_list]

    # safety = [dt['safety'] for dt in data_list]
    # discount = [dt['discount_term'] for dt in data_list]
    # rew_c = [dt['reward_calc'] for dt in data_list]
    # visualize_coordinates_with_value(path, "instinct reward", instinct_rewards, hazards_pos, goal_pos)
    # visualize_coordinates_with_value(path, "policy reward", policy_rewards, hazards_pos, goal_pos)
    visualize_coordinates_with_value(path, "instinct regulation", instinct_reg, hazards_pos, goal_pos, None, box_pos, 0, 1)

    # visualize_coordinates_with_value(path, "safety", safety, hazards_pos, goal_pos, -10, 1)
    # visualize_coordinates_with_value(path, "discount", discount, hazards_pos, goal_pos, 0, 1)
    # visualize_coordinates_with_value(path, "rew_calc", rew_c, hazards_pos, goal_pos, 0, 1)
    plt.show()


def visualize_coordinates_with_value(path, title, values, hazards_pos, goal_pos, buttons_pos, box_pos, cmin=-0.01, cmax=0.01):
    # plt.figure()
    fig, ax1 = plt.subplots(1, 1)
    ax1.set_title(title)
    ax1.set_xlim(-4.0, 4.0)
    ax1.set_ylim(-4.0, 4.0)
    path = np.array(path)
    norm = colors.Normalize(vmin=cmin, vmax=cmax)
    cmap = cm.get_cmap('jet')
    plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap))
    for i in range(1, len(path)):
        px1 = path[i - 1, 0]
        py1 = path[i - 1, 1]
        px2 = path[i, 0]
        py2 = path[i, 1]
        val = values[i]
        ax1.plot([px1, px2], [py1, py2], linewidth=3, color=cmap(norm(val)))
    for h in hazards_pos:
        ax1.add_patch(plt.Circle([h[0], h[1]], 0.25, color='b', alpha=0.2))

    if goal_pos is not None:
        for g in goal_pos:
            ax1.add_patch(plt.Circle([g[0], g[1]], 0.3, color='g', alpha=1.0))

    if buttons_pos is not None:
        for b in buttons_pos:
            ax1.add_patch(plt.Circle([b[0], b[1]], 0.1, color='orange', alpha=1.0))

    if box_pos is not None:
        for bx in box_pos:
            ax1.add_patch(plt.Rectangle([bx[0], bx[1]], 0.25, 0.25, color='orange', alpha=1.0))

    print(f"max value = {max(values)}, min_value = {min(values)}")
    print("stop here")


def main(repeat_num):
    args = get_args()
    print("start the train function")
    args.init_sigma = 0.6
    args.lr = 0.001
    device = torch.device("cpu")

    # Init the environment
    # env_name = "Safexp-PointGoal1-v0"
    eval_envs = make_vec_envs(env_name, np.random.randint(2 ** 32), 1,
                              args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    obs_shape = eval_envs.observation_space.shape
    actor_critic_policy = init_default_ppo(eval_envs, log(args.init_sigma))

    # Prepare modified action space for instinct
    inst_action_space = deepcopy(eval_envs.action_space)
    inst_obs_shape = list(obs_shape)
    inst_obs_shape[0] = inst_obs_shape[0] + eval_envs.action_space.shape[0]

    inst_action_space.shape = list(inst_action_space.shape)
    inst_action_space.shape[0] = inst_action_space.shape[0] + 1
    inst_action_space.shape = tuple(inst_action_space.shape)
    actor_critic_instinct = Policy(tuple(inst_obs_shape),
                                   inst_action_space,
                                   init_log_std=log(args.init_sigma),
                                   base_kwargs={'recurrent': False})

    title = "baseline_pretrained_hh_10"
    # f = open(f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BUTTON_more_space/{title}.csv", "w")
    actor_critic_policy = torch.load(
        # f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space_more_time/hh_10_baseline_centered_noHaz/model_rl_policy_latest.pt"
        "/home/calavera/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space/hh_10/model_rl_policy_latest.pt"
        # "/home/calavera/code/ITU_work/IR2L_master/pretrained_policy.pt"
    )
    actor_critic_instinct = torch.load(
        f"/home/calavera/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space/hh_10/model_rl_instinct_latest.pt"
    )

    ob_rms = utils.get_vec_normalize(eval_envs)

    if ob_rms is not None:
        ob_rms = ob_rms.ob_rms
    ob_rms = pickle.load(open(
        f"/home/calavera/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space/hh_10/ob_rms.p",
        "rb"))

    for _ in range(repeat_num):
        fits, info = evaluate(
            # EvalActorCritic(actor_critic_policy, actor_critic_instinct, det_policy=True, det_instinct=True),
            EvalActorCritic(actor_critic_policy, actor_critic_instinct),
            ob_rms, eval_envs, 1, reward_cost_combinator, device, instinct_on=True, visualise=True
        )
        visualise_values_over_path(info['plot_info'])

        # f.write(f"fitness; {fits.item()}; hazard_collisions; {info['hazard_collisions']}\n")
        # f.flush()

        print(f"{info['hazard_collisions']}")
        print(f"fitness; {fits.item()}; hazard_collisions; {info['hazard_collisions']}\n")


if __name__ == "__main__":
    main(50)
