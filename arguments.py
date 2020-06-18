""" Program arguments """
import argparse


def positive_nonzero_float(x):
    x = float(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 " % (x,))
    return x


def positive_nonzero_int(x):
    x = int(x)
    if x < 0.0:
        raise argparse.ArgumentTypeError("%r not bigger than 0.0 " % (x,))
    return x


def probability_float(x):
    x = float(x)
    if x < 0.0 or x > 1.0:
        raise argparse.ArgumentTypeError("%r not a probability " % (x,))
    return x


def get_args():
    parser = argparse.ArgumentParser(description="RL")
    parser.add_argument(
        "--save-dir",
        default="./trained_models/",
        help="directory to save agent logs (default: ./trained_models/)",
    )
    parser.add_argument(
        "--load-ga",
        action="store_true",
        help="Load a saved state from the last generation found in the specified directiory",
    )

    parser.add_argument(
        "--load-exp-idx",
        type=int,
        default=0,
        metavar="N",
        help="continue with the experiment matching git hash + this index",
    )

    parser.add_argument("--debug", action="store_true", default=False)

    parser.add_argument(
        "--seed", type=int, default=543, metavar="N", help="random seed (default: 543)"
    )
    parser.add_argument("--render", action="store_true", help="render the environment")
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="interval between training status logs (default: 10)",
    )

    """ Learning specific arguments """
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        metavar="G",
        help="discount factor (default: 0.99)",
    )
    parser.add_argument(
        "--lr", type=float, default=0.001, help="learning rate (default: 0.001)"
    )
    parser.add_argument("--ep-training", action="store_true", default=False)

    parser.add_argument(
        "--init-sigma",
        type=positive_nonzero_float,
        default=0.3,
        help="initialized value for the exploration sigma parameter",
    )

    """Arguments that change the behaviour of the model """
    parser.add_argument(
        "--num-proc",
        type=positive_nonzero_int,
        default=5,
        help="Define over how many processes to parallelize the fitness evaluation.",
    )

    parser.add_argument(
        "--pop-size",
        type=positive_nonzero_int,
        default=480,
        help="Define how large is the genetic algorithm population",
    )
    parser.add_argument(
        "--rm-hazard",
        action="store_true",
        default=False,
        help="if TRUE, the supported environment will not have no-go zones",
    )
    parser.add_argument(
        "--num-goal-samples",
        default=3,
        type=positive_nonzero_int,
        help='Defined how many steps in the cycle through goals',
    )
    model_type_group = parser.add_mutually_exclusive_group()
    model_type_group.add_argument(
        "--parametric-combinator",
        action="store_true",
        default=False,
        help="if TRUE, genetic algorithm will instantiate parametric combinator",
    )
    model_type_group.add_argument(
        "--ppo",
        action="store_true",
        default=False,
        help="if TRUE, genetic algorithm will instantiate PPO as the main controller model",
    )
    model_type_group.add_argument(
        "--instinct-sigma",
        action="store_true",
        default=False,
        help="If TRUE, initialize the model that uses an instinct network that outputs the sigma value of the exploration distribution",
    )

    parser.add_argument(
        "--reduce-goals",
        action="store_true",
        default=False,
        help="if TRUE, run meta-learning on two predetermined goals. Used for quick experiments.",
    )

    """ Arguments specific to PPO module and are copied directly from the PPO main implementation as defaults"""
    parser.add_argument(
        "--clip-param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--ppo-epoch", type=int, default=4, help="number of ppo epochs (default: 4)"
    )
    parser.add_argument(
        "--num-mini-batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy-coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--value-loss-coef",
        type=float,
        default=0.5,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=0.5,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use-gae",
        action="store_true",
        default=False,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gae-lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use-proper-time-limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--start-gen-idx",
        type=positive_nonzero_int,
        default=0,
        help="What's the starting generation. Used when restarting the search to avoid overriding stuff from the previous run.",
    )
    parser.add_argument(
        "--norm-vectors",
        action="store_true",
        default=False,
        help="Encase the environment into a wrapper that normalizes outputs of the said environment",
    )

    args = parser.parse_args()
    args.cuda = False

    return args
