import numpy as np
import time
import os

import torch
from torch.utils.tensorboard import SummaryWriter

np.random.seed(0)

def compute_ranks(x):
    """
    Returns ranks in [0, len(x))
    Note: This is different from scipy.stats.rankdata, which returns ranks in [1, len(x)].
    (https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py)
    """
    assert x.ndim == 1
    ranks = np.empty(len(x), dtype=int)
    ranks[x.argsort()] = np.arange(len(x))
    # print("RANKS")
    return ranks


def compute_centered_ranks(x):
    """
    https://github.com/openai/evolution-strategies-starter/blob/master/es_distributed/es.py
    """
    y = compute_ranks(x.ravel()).reshape(x.shape).astype(np.float32)
    y /= x.size - 1
    y -= 0.5
    return y


def flatten_weights(weights):
    w_flat = np.array([])
    for w in weights:
        w_flat = np.append(w_flat, w.flatten())
    return w_flat

def worker_process(get_reward_func, weights):  # (arg):
    # !!!get_reward_func, weights = arg
    # print("P")
    # print(np.mean(weights) )
    # print(0.01 * np.mean(weights) )
    #get_reward_func = input[0]
    #weights = input[1]
    wp = flatten_weights(weights)

    # weights decay
    decay = -0.01 * np.mean(wp * wp)
    # print(decay)
    r = get_reward_func(weights) + decay

    return r  # get_reward_func(weights)


class EvolutionStrategy(object):
    def __init__(
        self,
        weights,
        get_reward_func,
        population_size=50,
        sigma=0.1,
        learning_rate=0.03,
        decay=0.999,
        experiment_save_dir=""
    ):

        self.weights = weights
        self.get_reward = get_reward_func
        self.POPULATION_SIZE = population_size
        self.SIGMA = sigma
        self.learning_rate = learning_rate
        self.decay = decay
        self.exp_save_dir = experiment_save_dir

    def _get_weights_try(self, w, p):
        weights_try = []
        for index, i in enumerate(p):
            jittered = self.SIGMA * i
            weights_try.append(w[index] + jittered)

        return weights_try

    def get_weights(self):
        return self.weights

    def _get_population(self):
        population = []
        for i in range(int(self.POPULATION_SIZE / 2)):
            x = []
            x2 = []
            for w in self.weights:
                j = np.random.randn(*w.shape)
                # print(j)
                x.append(j)
                x2.append(-j)
                # print(j, -j)

            population.append(x)
            population.append(x2)

        return population

    def _get_rewards(self, pool, population):

        if pool is not None:

            results = []
            worker_args = []
            for p in population:

                weights_try1 = []
                # weights_try2 = []

                for index, i in enumerate(p):
                    jittered = self.SIGMA * i

                    w = self.weights[index] + jittered
                    weights_try1.append(w)
                    # weights_try2.append(self.weights[index] - jittered)

                # print ("test ", worker_process((self.get_reward, weights_try1)) )

                #a = (self.get_reward, weights_try1)
                #results += [pool.apply_async(worker_process, args=a)]

                worker_args.append( (self.get_reward, weights_try1) )

                # worker_args.append( (self.get_reward, weights_try2) )

            # worker_args = ((self.get_reward, self._get_weights_try(self.weights, p)) for p in population)

            #rewards = [
            #    res.get() for res in results
            #]  # rewards = [res.get(timeout=1) for res in results] #
            rewards  = list(pool.map(worker_process, worker_args))
            # rewards = pool.apply_async(worker_process, worker_args)
            # rewards  = pool.map(, worker_args)

        else:
            rewards = []
            for p in population:
                weights_try = self._get_weights_try(self.weights, p)
                rewards.append(self.get_reward(weights_try))
        rewards = np.array(rewards)

        return rewards

    def _update_weights(self, rewards, population):
        # print(rewards)
        rewards = compute_centered_ranks(rewards)
        # print(rewards)
        # exit()

        std = rewards.std()
        if std == 0:
            return

        rewards = (rewards - rewards.mean()) / std
        for index, w in enumerate(self.weights):
            layer_population = np.array([p[index] for p in population])
            update_factor = self.learning_rate / (self.POPULATION_SIZE * self.SIGMA)
            self.weights[index] = (
                w + update_factor * np.dot(layer_population.T, rewards).T
            )

        if self.learning_rate > 0.001:
            self.learning_rate *= self.decay

        # Decay sigma
        if self.SIGMA > 0.01:
            self.SIGMA *= 0.999

        # print(self.learning_rate, self.SIGMA)

    def run(self, iterations, pool, print_step=10, start_iteration=0):
        log_writer = SummaryWriter(self.exp_save_dir, max_queue=1, filename_suffix="log")
        for iteration in range(start_iteration, iterations):

            start = time.time()
            population = self._get_population()
            rewards = self._get_rewards(pool, population)

            self._update_weights(rewards, population)
            end = time.time()

            if (iteration + 1) % print_step == 0:
                log_writer.add_scalar("Fitness", self.get_reward(self.weights), iteration+1)
                log_writer.add_scalar("learning rate", self.learning_rate, iteration+1)
                log_writer.add_scalar("Sigma", self.SIGMA, iteration+1)

                torch.save(
                    self.get_weights(), os.path.join(self.exp_save_dir, f"saved_weights_gen_{iteration}.dat")
                )

        #if pool is not None:
        #    pool.close()
        #    pool.join()
