from os import listdir, path
from functools import partial
from visualisations.vis_boxplots import run
import torch
from model import ControllerCombinator
from arguments import get_args
import pickle
from statistics import mean
from multiprocessing import Pool

EVOLUTION_DIR = "./trained_models/evolution"
NUM_EXP = 20


def get_model_eval(model_filename):
    torch.set_num_threads(1)
    args = get_args()
    m1_orig, learning_rate = torch.load(model_filename)
    m1 = ControllerCombinator(2, 100, 2, load_instinct=args.load_instinct)
    m1.load_state_dict(m1_orig.state_dict())
    experiment1_fits = [run(m1, learning_rate, False) for _ in range(NUM_EXP)]
    experiment1_fits = list(zip(*experiment1_fits))
    return mean(experiment1_fits[1]), experiment1_fits[1]


def main():
    generation_ids = listdir(EVOLUTION_DIR)
    join_gen = partial(path.join, EVOLUTION_DIR)
    gen_paths = map(join_gen, generation_ids)
    ind_file_names = map(lambda x: "individual_" + x + ".pt", generation_ids)
    model_paths = [path.join(stem, fn) for (stem, fn) in zip(gen_paths, ind_file_names)]

    with Pool(20) as pool:
        avg_fitnesses = list(pool.map(get_model_eval, model_paths))
    avg_fitnesses = list(zip(*avg_fitnesses))

    max_performance = float("-inf"), None, None  # average fitness, fitness_list, id
    for m_id, avg, flist in zip(generation_ids, avg_fitnesses[0], avg_fitnesses[1]):
        if avg > max_performance[0]:
            max_performance = (avg, flist, m_id)

    print(
        "best average fitness is {} from the {}th individual".format(
            max_performance[0], max_performance[2]
        )
    )
    with open("experiment_sweep_best.list", "wb") as pckl_file:
        pickle.dump(max_performance[1], pckl_file)


if __name__ == "__main__":
    main()
