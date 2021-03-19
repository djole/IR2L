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
#from a2c_ppo_acktr.model import init_default_ppo, Policy, custom_weight_init
from arguments import get_args
from double_rl_loop_main import reward_cost_combinator, config_box_no_haz#, config1, config2, config3, config4
from copy import deepcopy
import pickle


env_name = 'SafexpCustomEnvironmentGoal1Test-v0'
register(id=env_name,
             entry_point='safety_gym_mod.envs.mujoco:Engine',
             kwargs={'config': config_box_no_haz})


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
    #f = open(f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BUTTON_more_space/{title}.csv", "w")
    actor_critic_policy = torch.load(
        f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space_more_time/hh_10_baseline_centered_noHaz/model_rl_policy_latest.pt"
        )
    #actor_critic_instinct = torch.load(
    #    f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BUTTON_more_space_more_time/hh_10_instinct_noHaz/model_rl_instinct_latest.pt"
    #    )

    ob_rms = utils.get_vec_normalize(eval_envs)

    if ob_rms is not None:
        ob_rms = ob_rms.ob_rms
    ob_rms = pickle.load(open(f"/Users/djgr/pulled_from_server/evaluate_instinct_all_inputs_task_switch_button/real_safety_tasks_easier/sweep_eval_hazard_param_BOX_more_space_more_time/hh_10_baseline_centered_noHaz/ob_rms.p", "rb"))

    for _ in range(repeat_num):
        fits, info = evaluate(
                    #EvalActorCritic(actor_critic_policy, actor_critic_instinct, det_policy=True, det_instinct=True),
                    EvalActorCritic(actor_critic_policy, actor_critic_instinct),
                    ob_rms, eval_envs, 1, reward_cost_combinator, device, instinct_on=True, visualise=True
        )

        #f.write(f"fitness; {fits.item()}; hazard_collisions; {info['hazard_collisions']}\n")
        #f.flush()

        print(f"{info['hazard_collisions']}")
        print(f"fitness; {fits.item()}; hazard_collisions; {info['hazard_collisions']}\n")


if __name__ == "__main__":
    main(50)
