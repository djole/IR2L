from double_rl_loop_main import EvalActorCritic

import torch
import numpy as np
import gym
import safety_gym
from gym.envs.registration import register

from a2c_ppo_acktr import algo, utils
from a2c_ppo_acktr.envs import make_vec_envs
from a2c_ppo_acktr.evaluation import evaluate
from a2c_ppo_acktr.model import init_default_ppo, Policy, custom_weight_init
from arguments import get_args


def main(repeat_num):
    args = get_args()
    print("start the train function")

    args.init_sigma = 0.6
    args.lr = 0.001

    device = torch.device("cpu")

    # plot_weight_histogram(parameters)
    actor_critic_policy = torch.load(
        "/Users/djrg/code/instincts/modular_rl_safety_gym/trained_models/pulled_from_server/double_rl_experiments/policy_plus_instinct/instinct_regularization_param_sweep/649e65831f_0_p007/model_rl_policy.pt")
    actor_critic_instinct = torch.load(
        "/Users/djrg/code/instincts/modular_rl_safety_gym/trained_models/pulled_from_server/double_rl_experiments/policy_plus_instinct/instinct_regularization_param_sweep/649e65831f_0_p007/model_rl_instinct.pt")

    # Init the environment

    env_name = "Safexp-PointGoal1-v0"
    eval_envs = make_vec_envs(env_name, np.random.randint(2 ** 32), 1,
                              args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
    ob_rms = utils.get_vec_normalize(eval_envs)
    if ob_rms is not None:
        ob_rms = ob_rms.ob_rms

    for _ in range(repeat_num):
        fits, info = evaluate(EvalActorCritic(actor_critic_policy, actor_critic_instinct), ob_rms, eval_envs, 1,
                              device, instinct_on=True,
                              visualise=True)

    print(f"fitness = {fits.item()}, cost = {info['cost']}")


if __name__ == "__main__":
    main(10)
