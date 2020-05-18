import os

from train_test_model import inner_loop_ppo
from arguments import get_args
from env_util import register_set_goal
from math import log
from a2c_ppo_acktr.model import init_ppo, PolicyWithInstinct
from a2c_ppo_acktr.envs import make_vec_envs
from simpleES import EvolutionStrategy
from exp_dir_util import get_experiment_save_dir, get_start_gen_idx

from functools import partial
import torch
import numpy as np

def get_model_weights(model: PolicyWithInstinct):
    params = model.get_evolvable_params()
    copy_params = []

    for p in params:
        copy_params.append(p.data.clone().detach().numpy())

    return copy_params



# Fitness function
def es_fitness_funct(parameters, args, num_steps, num_updates, num_goals):
    weights = parameters[:-1]
    learning_rate = parameters[-1][0]
    learning_rate = -learning_rate if learning_rate < 0 else learning_rate

    goal_info = [
        inner_loop_ppo(
            weights, args, learning_rate, num_steps, num_updates, run_idx=num_att
        )
        for num_att in range(num_goals)
    ]
    goal_fitnesses, _, _ = list(zip(*goal_info))
    return sum(goal_fitnesses)



if __name__ == "__main__":

    pop_size = 504
    num_steps = 4000

    args = get_args()

    # set up the parallelization
    try:
        from mpipool import Pool
        pool = Pool()
    except:
        pool = None

    experiment_save_dir = get_experiment_save_dir(args)
    env_name = register_set_goal(0)
    init_sigma = args.init_sigma

    envs = make_vec_envs(
        env_name, args.seed, 1, args.gamma, None, torch.device("cpu"), False
    )


    if args.load_ga:
        last_iter = get_start_gen_idx(args.load_ga, experiment_save_dir) - 1
        start_weights = torch.load(os.path.join(experiment_save_dir, f"saved_weights_gen_{last_iter}.dat"))
    else:
        blueprint_model = init_ppo(envs, log(init_sigma))
        start_weights = get_model_weights(blueprint_model)
        start_weights.append(np.array([args.lr]))

    #fitness_function = make_es_fitness_funct(args, num_steps, 1, args.num_goal_samples)
    fitness_function = partial(
            es_fitness_funct, args=args, num_steps=num_steps, num_updates=1, num_goals=args.num_goal_samples
        )

    es = EvolutionStrategy(
        start_weights,
        fitness_function,
        population_size=pop_size,
        sigma=0.1,
        learning_rate=0.1,
        decay=0.995,
        experiment_save_dir=experiment_save_dir
    )

    es.run(1000, pool=pool, print_step=1, start_iteration=get_start_gen_idx(args.load_ga, experiment_save_dir))
