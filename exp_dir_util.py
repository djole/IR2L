from git import Repo
import os
import torch
import copy

REPO = Repo(os.getcwd())
REF = REPO.head.reference
SHA = REPO.head.object.hexsha
SHORT_COMMIT_SHA = REPO.git.rev_parse(SHA, short=10)
LOAD_SUBDIR = "___LAST___"

def _is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False

def get_experiment_save_dir(args):
    save_exp_dir_base = os.path.join(args.save_dir, SHORT_COMMIT_SHA)

    ### Check if the previous exp with the same hash already exists
    idx = 0

    if args.load_ga:
        idx = args.load_exp_idx
        save_exp_dir = save_exp_dir_base + "_" + str(idx)
    else:
        while os.path.exists(save_exp_dir_base + "_" + str(idx)):
            idx += 1

        save_exp_dir = save_exp_dir_base + "_" + str(idx)
        os.makedirs(save_exp_dir)

    # Dump git log into a file
    with open(os.path.join(save_exp_dir, "GITLOG.log"), "w") as f:
        f.write(REPO.git.log("-n 5"))

    return save_exp_dir

def get_start_gen_idx(load_ga, experiment_dir):
    if load_ga:
        subdirs = os.listdir(experiment_dir)
        gen_subdirs = list(filter(_is_int, subdirs))
        sub_dir_idxs = list(sorted(map(int, gen_subdirs)))
        max_idx = sub_dir_idxs[-1]
        return max_idx+1
    else:
        return 0


def save_population(args, population, best_ind, generation_idx, experiment_dir):
    save_path = os.path.join(experiment_dir, str(generation_idx))
    save_path_checkpoint = os.path.join(experiment_dir, LOAD_SUBDIR)
    try:
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        if not os.path.exists(save_path_checkpoint):
            os.makedirs(save_path_checkpoint)
    except OSError:
        pass

    for individual in population:
        save_model = individual.model
        save_lr = individual.learning_rate
        if args.cuda:
            save_model = copy.deepcopy(individual.model).cpu()

        torch.save(
            (save_model, save_lr),
            os.path.join(
                save_path_checkpoint, "individual_" + str(individual.rank) + ".pt"
            ),
        )

    # Save the best
    save_model = best_ind.model
    save_lr = best_ind.learning_rate
    if args.cuda:
        save_model = copy.deepcopy(best_ind.model).cpu()
    torch.save(
        (save_model, save_lr),
        os.path.join(save_path, "individual_" + str(generation_idx) + ".pt"),
    )

if __name__ == "__main__":
    print(REPO.git.log("-n 5"))