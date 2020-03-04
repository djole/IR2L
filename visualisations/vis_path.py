"""Visualise a trained model"""
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch

import navigation_2d
from arguments import get_args
from model import ControllerCombinator, ControllerMonolithic
from train_test_model import train_maml_like
from navigation_2d import dist_2_nogo

NUM_EPISODES = 40
NUM_UPDATES = 1
NUM_EXP = 5
SMALL_NOGO_UPPER = 0.3
SMALL_NOGO_LOWER = 0.2
LARGE_NOGO_UPPER = 0.4
LARGE_NOGO_LOWER = 0.05

def vis_instinct_action(model, env=None, alldists=False):
    input_xs = get_mesh()
    select_model_action = lambda modl, inputs: modl(inputs)
    amplify_action = lambda instinct_action, control: (instinct_action * (1 - control))

    def get_lidar_data(x):
        env._state = x
        return env._lidar_no_go_perception()

    glue_input = lambda i: torch.tensor([np.append(i, get_lidar_data(i))], dtype=torch.float32)

    # Normalized instinct output at (0, 0)
    # ref_vec = amplify_action(*select_model_action(model, glue_input(torch.tensor([0.0, 0.0])))).flatten()

    z = [
        amplify_action(*select_model_action(model, glue_input(x))).flatten() for x in input_xs
    ]
    plt.figure()
    axis = plt.gca()
    # x, y = input_xs.transpose()[0], input_xs.transpose()[1]
    # x = np.reshape(x, (40, 40))
    # y = np.reshape(y, (40, 40))
    # z = np.reshape(z, (40, 40))
    # axis.pcolormesh(x, y, z, cmap="YlGn")
    input_xs_t = input_xs.transpose()
    z_tensor = torch.stack(z)
    z_tensor_t = z_tensor.t()
    for i in range(len(z)):
        axis.arrow(
            input_xs_t[0][i],
            input_xs_t[1][i],
            z_tensor_t[0][i].detach().numpy(),
            z_tensor_t[1][i].detach().numpy(),
        )
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)

    plt.show()


def vis_path(vis, saveidx=None, slice=None, nogo_large=False, eval_path_rec=None, offending=None):
    """ Visualize the path """
    nogo_lower = LARGE_NOGO_LOWER if nogo_large else SMALL_NOGO_LOWER
    nogo_upper = LARGE_NOGO_UPPER if nogo_large else SMALL_NOGO_UPPER
    nogo_size = nogo_upper - nogo_lower
    plt.figure()
    axis = plt.gca()
    # Plot the exploration paths
    for v in vis:
        path_rec = v[0]
        goal = v[1]


        if slice is None:
            pth = list(zip(*path_rec))
        else:
            pth = list(zip(*path_rec[slice - 1:slice]))
        axis.plot(*pth, "g")
        axis.scatter(*goal, color="red", s=250)

    # Plot the offending paths
    if offending is not None:
        for st in offending:
            stz = list(zip(*st))
            axis.plot(stz[0], stz[1], color='orange')


    # Plot the evaluation path
    if eval_path_rec is not None:
        eval_path_rec = list(zip(*eval_path_rec))
        axis.plot(*eval_path_rec, color='purple')

    axis.add_patch(plt.Rectangle((nogo_lower, nogo_lower), nogo_size, nogo_size, fc="b", alpha=0.5))
    axis.add_patch(plt.Rectangle((-nogo_upper, nogo_lower), nogo_size, nogo_size, fc="b", alpha=0.5))
    axis.add_patch(plt.Rectangle((nogo_lower, -nogo_upper), nogo_size, nogo_size, fc="b", alpha=0.5))
    axis.add_patch(plt.Rectangle((-nogo_upper, -nogo_upper), nogo_size, nogo_size, fc="b", alpha=0.5))

    axis.set_xlim(-1.0, 1.0)
    axis.set_ylim(-1.0, 1.0)
    if saveidx is None:
        plt.show()
    else:
        plt.savefig("/Users/djgr/code/instincts/modular_rl/trained_models/pulled_from_server/second_phase_instinct/n_deterministic_goals/4goals_lidar_ctrl_noCoordiantes/balance_plot_visualisation/plots/img_{}".format(saveidx))
        plt.close()

def show_empty_yard(nogo_large=False):
    nogo_lower = LARGE_NOGO_LOWER if nogo_large else SMALL_NOGO_LOWER
    nogo_upper = LARGE_NOGO_UPPER if nogo_large else SMALL_NOGO_UPPER
    nogo_size = nogo_upper - nogo_lower
    axis = plt.gca()
    axis.add_patch(plt.Rectangle((nogo_lower, nogo_lower), nogo_size, nogo_size, fc="b", alpha=0.3))
    axis.add_patch(plt.Rectangle((-nogo_upper, nogo_lower), nogo_size, nogo_size, fc="b", alpha=0.3))
    axis.add_patch(plt.Rectangle((nogo_lower, -nogo_upper), nogo_size, nogo_size, fc="b", alpha=0.3))
    axis.add_patch(plt.Rectangle((-nogo_upper, -nogo_upper), nogo_size, nogo_size, fc="b", alpha=0.3))

    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)
    plt.show()


def get_mesh():
    input_x = torch.arange(-1, 1, 0.025)
    input_y = torch.arange(-1, 1, 0.025)

    input_xy = np.stack(np.meshgrid(input_x, input_y))
    input_xy = input_xy.reshape(2, -1)
    # input_xy = torch.tensor(input_xy).t()
    return input_xy.transpose()


def vis_heatmap2(model, env=None, alldists=False):
    input_xs = get_mesh()
    select_model_action = lambda modl, inputs: modl(inputs)
    norm_action = lambda instinct_action, control: torch.norm(control)
    def get_lidar_data(x):
        env._state = x
        return env._lidar_no_go_perception()
    glue_input = lambda i: torch.tensor([np.append(i, get_lidar_data(i))], dtype=torch.float32)

    z = [
        norm_action(*select_model_action(model, glue_input(x))).flatten() for x in input_xs
    ]
    plt.figure()
    axis = plt.gca()
    x, y = input_xs.transpose()[0], input_xs.transpose()[1]
    x = np.reshape(x, (80, 80))
    y = np.reshape(y, (80, 80))
    z = np.reshape(z, (80, 80))
    axis.pcolormesh(x, y, z, cmap="YlGnBu", vmin=np.min(z), vmax=np.max(z))
    #axis.imshow(z, vmin=z.min(), vmax=z.max())
    # axis.pcolormesh(x, y, z, cmap="Reds", alpha=0.5)
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)

    plt.show()

def vis_heatmap(model, env=None, alldists=False):
    input_xs = get_mesh()
    select_model_action = lambda modl, inputs: modl(inputs)
    def get_lidar_data(x):
        env._state = x
        return env._lidar_no_go_perception()
    glue_input = lambda i: torch.tensor([np.append(i, get_lidar_data(i))], dtype=torch.float32)
    z = [
        select_model_action(model, glue_input(x))[1].mean().item() for x in input_xs
    ]
    plt.figure()
    axis = plt.gca()
    x, y = input_xs.transpose()[0], input_xs.transpose()[1]
    x = np.reshape(x, (80, 80))
    y = np.reshape(y, (80, 80))
    z = np.reshape(z, (80, 80))
    axis.pcolormesh(x, y, z, cmap="Greens")
    #axis.imshow(z, vmin=z.min(), vmax=z.max())
    # axis.pcolormesh(x, y, z, cmap="Reds", alpha=0.5)
    axis.set_xlim(-0.5, 0.5)
    axis.set_ylim(-0.5, 0.5)

    plt.show()


def run(model, lr, unfreeze, args):
    """Run"""
    args.unfreeze_modules = unfreeze
    task_idx = 1
    # model_filename = "./trained_models/pulled_from_server/model995.pt"
    # model_filename = "./trained_models/pulled_from_server/maml_like_model_20episodes_lastGen436.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model597.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode/model977.pt"
    # model_filename = "./trained_models/pulled_from_server/20random_goals4modules20episode_monolith_multiplexor/individual_985.pt"
    # m = Controller(2, 100, 2)
    # m = ControllerCombinator(2, 2, 100, 2)
    # env.seed(args.seed)

    # c_reward, reached, _, vis = episode_rollout(module, env, vis=True)
    c_reward, reached, vis = train_maml_like(
        model,
        args,
        num_episodes=NUM_EPISODES,
        learning_rate=lr,
        num_updates=NUM_UPDATES,
        vis=True,
    )
    print("The cummulative reward for the {} task is {}.".format(task_idx, c_reward))
    print("The goal was reached" if reached else "The goal was NOT reached")
    vis_path(vis)
    # vis_heatmap(model)
    return c_reward
