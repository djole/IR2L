"""the entry point into the program"""
import torch
from simpleGA import rollout
from exp_dir_util import get_experiment_save_dir

try:
    from mpipool import Pool
except:
    print('using multiprocessing, be careful!')
    from multiprocessing import Pool

D_IN, D_OUT = 2, 2


def main():
    from arguments import get_args

    args = get_args()
    device = torch.device("cpu")
    pool = Pool()
    experiment_save_dir = get_experiment_save_dir(args)
    if args.debug:
        rollout(args, D_IN, D_OUT, pool, device, exp_save_dir=experiment_save_dir, pop_size=5, elite_prop=0.2)
    else:
        print('there should be only one of me!')
        rollout(args, D_IN, D_OUT, pool, device, exp_save_dir=experiment_save_dir, pop_size=args.pop_size)


if __name__ == "__main__":
    main()
