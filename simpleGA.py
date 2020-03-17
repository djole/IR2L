import copy
import itertools as itools
import os
import time
from functools import partial
from math import log
from statistics import mean

import numpy as np
import torch

from env_util import register_set_goal
from a2c_ppo_acktr.model import init_ppo
from a2c_ppo_acktr.envs import make_vec_envs
from train_test_model import inner_loop_ppo

from exp_dir_util import save_population, LOAD_SUBDIR, get_start_gen_idx

# MAXTSK_CHLD = 10
START_LEARNING_RATE = 7e-4


def get_population_files(load_ga_dir):

    ind_files = [name for name in os.listdir(load_ga_dir)]
    ind_files = list(map(partial(os.path.join, load_ga_dir), ind_files))

    return ind_files


class Individual:
    """ A struct containing the evolvable elements per individual """

    def __init__(self, model, device, rank, learn_rate):
        self.model = model
        self.device = device
        self.rank = rank
        # A set of masks that will prevent some weigths from being changed by the optimizer
        # The mask is initialized to all ones to maintain the default behavior
        self.model_plasticity_masks = []
        self.learning_rate = learn_rate


class EA:
    """ EA class """

    def _compute_ranks(self, x):
        assert x.ndim == 1
        ranks = np.empty(len(x), dtype=int)
        ranks[x.argsort()] = np.arange(len(x))
        return ranks

    def _compute_centered_ranks(self, fitnesses):
        x = np.array(fitnesses)
        y = self._compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
        y /= x.size - 1
        y -= 0.5
        return y.tolist()

    def __init__(self, args, device, pop_size, elite_prop, load_pop_dir):
        if pop_size < 1:
            raise ValueError(
                "Population size has to be one or greater, otherwise this doesn't make sense"
            )
        self.pop_size = pop_size
        self.population = []  # a list of lists/generators of model parameters
        self.selected = []  # a buffer for the selected individuals
        self.to_select = int(self.pop_size * elite_prop)
        if self.to_select == 0:
            self.to_select = 1
        self.fitnesses = []
        self.reached = []
        self.instinct_average_list = []
        self.args = args

        self.sigma = 0.01
        self.sigma_decay = 0.999
        self.min_sigma = 0.001

        # if recover GA, load a list of files representing the population
        if args.load_ga:
            saved_files = get_population_files(load_pop_dir)

        ref_env_name = register_set_goal(0)

        reference_envs = make_vec_envs(ref_env_name, np.random.randint(2 ** 32), 1,
                             args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)

        for n in range(pop_size + self.to_select):
            if args.load_ga:
                file_idx = n % len(saved_files)
                start_model, start_lr = torch.load(saved_files[file_idx])
                print("Load individual from {}".format(saved_files[file_idx]))
            else:
                start_model = (
                    init_ppo(
                        reference_envs,
                        log(args.init_sigma),
                    )
                )
                start_lr = args.lr

            ind = Individual(start_model, device, rank=n, learn_rate=start_lr)

            if n < self.pop_size:
                self.population.append(ind)
                self.fitnesses.append(0)
                self.reached.append(0)
                self.instinct_average_list.append(0)
            else:
                self.selected.append(ind)
            print(
                "Built {} individuals out of {}".format(n, (pop_size + self.to_select))
            )

    def ask(self):
        return self.population

    def tell(self, fitnesses):
        if len(fitnesses) != len(self.fitnesses):
            raise ValueError("Fitness array mismatch")

        fitness_list, reached_list, instinct_average_list = list(zip(*fitnesses))
        self.fitnesses = fitness_list
        self.reached = reached_list
        self.instinct_average_list = instinct_average_list

    def step(self, generation_idx, args, device):
        """One step of the evolution"""
        # Sort the population by fitness and select the top
        sorted_fit_idxs = list(reversed(sorted(zip(self.fitnesses, itools.count()))))
        sorted_pop = [self.population[ix] for _, ix in sorted_fit_idxs]

        # recalculate the fitness of the elite subset and find the best individual
        elite_pop = sorted_pop[: len(self.selected)]
        # re_fit_max = float("-inf")
        # max_idx = 0
        # fitness_recalclulation_ = partial(self.fitness_calculation, num_processes=10)
        # re_fits = []

        # with Pool(processes=NUM_PROC, maxtasksperchild=MAXTSK_CHLD) as pool:
        # re_fits = map(fitness_recalclulation_, elite_pop)

        # for re_fit, (_, elite_ix) in zip(re_fits, sorted_fit_idxs):
        #    if re_fit > re_fit_max:
        #        max_idx = elite_ix
        #        re_fit_max = re_fit
        max_fitness, max_idx = sorted_fit_idxs[0]
        for cp_from, cp_to in zip(sorted_pop, self.selected):
            cp_to.model.load_state_dict(cp_from.model.state_dict())

        print(
            "\n=============== Generation index {} ===============".format(
                generation_idx
            )
        )
        print("best in the population ----> ", sorted_fit_idxs[0][0])
        print("best's learning rate ------>", self.population[max_idx].learning_rate)
        print("best in population reached {} goals".format(self.reached[max_idx]))
        print("best in population instinct activation average ------>", self.instinct_average_list[max_idx])
        # print("best in the population after stabilization", re_fit_max)
        print("worst in the population ----> ", sorted_fit_idxs[-1][0])
        print("worst parent --------------->", sorted_fit_idxs[self.to_select - 1][0])
        print("average fitness ------> ", sum(self.fitnesses) / len(self.fitnesses))
        print("===================================================\n")

        # next generation
        for i in range(self.pop_size):
            if i == max_idx:
                continue

            dart = int(torch.rand(1) * self.to_select)
            # Select parent and child
            parent = self.selected[dart]
            child = self.population[i]
            # copy the parent genes to the child genes
            child.model.load_state_dict(parent.model.state_dict())
            child.learning_rate = parent.learning_rate
            # apply mutation to model parameters
            for p in child.model.get_evolvable_params():
                mutation = torch.randn_like(p.data) * self.sigma
                p.data += mutation
            # apply mutation to learning rate
            child.learning_rate += torch.randn((1, 1)).item() * 0.001
            if child.learning_rate < 0:
                child.learning_rate *= -1

        if self.sigma > self.min_sigma:
            self.sigma *= self.sigma_decay
        elif self.sigma < self.min_sigma:
            self.sigma = self.min_sigma

        return (self.population[max_idx], max_fitness)

    def fitness_calculation(self, individual, args, num_attempts=20):
        torch.set_num_threads(1)
        # fits = [episode_rollout(individual.model, args, env, rollout_index=ri, adapt=args.ep_training) for ri in range(num_attempts)]
        fits = [
            inner_loop_ppo(
                individual.model, args, individual.learning_rate, run_idx=num_att
            )
            for num_att in range(num_attempts)
        ]
        fits, reacheds, instinct_control_avgs = list(zip(*fits))
        return sum(fits), sum(reacheds), mean(instinct_control_avgs)


def rollout(args, din, dout, pool, device, exp_save_dir, pop_size=140, elite_prop=0.1, debug=False):
    assert (
        elite_prop < 1.0 and elite_prop > 0.0
    ), "Elite needs to be a measure of proportion of population, 0 < elite_prop < 1"
    if debug:
        pop_size = 10
        elite_prop = 0.2

    # torch.manual_seed(args.seed)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    load_pop_from = os.path.join(exp_save_dir, LOAD_SUBDIR)
    solver = EA(args, device, pop_size, elite_prop=elite_prop, load_pop_dir=load_pop_from)
    fitness_list = [0 for _ in range(pop_size)]
    for iteration in range(get_start_gen_idx(args.load_ga, exp_save_dir), 1000):
        start_time = time.time()
        solutions = solver.ask()
        num_env_samples = args.num_goal_samples

        fitness_calculation_ = partial(
            solver.fitness_calculation, args=args, num_attempts=num_env_samples
        )
        if args.debug:
            fitness_list = list(map(fitness_calculation_, solutions))
        else:
            fitness_list = list(pool.map(fitness_calculation_, solutions))

        solver.tell(fitness_list)
        result, best_f = solver.step(iteration, args, device)
        # ========= Render =========
        # episode_rollout(result.model)
        # env.render_episode()
        # ==========================
        gen_time = time.time()
        save_population(args, solver.population, result, iteration, exp_save_dir)
        print(
            "Generation: {}\n The best individual has {} as the reward".format(
                iteration, best_f
            )
        )
        print("wall clock time == {}".format(gen_time - start_time))
    return result
