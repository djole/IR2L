"""the entry point into the program"""
import torch
from simpleGA import rollout

try:
    from mpipool import Pool
except:
    from multiprocessing import Pool

D_IN, D_OUT = 2, 2


def main():
    from arguments import get_args

    args = get_args()
    device = torch.device("cpu")
    pool = Pool()
    if args.debug:
        rollout(args, D_IN, D_OUT, pool, device, pop_size=5, elite_prop=0.2)
    else:
        rollout(args, D_IN, D_OUT, pool, device, pop_size=(args.num_proc * 8))


if __name__ == "__main__":
    main()
