"""the entry point into the program"""
import torch
from simpleGA import rollout

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
    if args.debug:
        rollout(args, D_IN, D_OUT, pool, device, pop_size=5, elite_prop=0.2)
    else:
        print('there should be only one of me!')
        rollout(args, D_IN, D_OUT, pool, device, pop_size=args.pop_size)


if __name__ == "__main__":
    main()
