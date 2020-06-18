import gym
import safety_gym
from gym.envs.registration import register
from gym.utils import seeding
import numpy as np
from math import cos, sin, pi
import torch
import copy

from a2c_ppo_acktr.envs import make_vec_envs

HAZARD_LOC_PARAM = 1
HLP = HAZARD_LOC_PARAM

GOAL_LOC_PARAM = 1.8
GLP = GOAL_LOC_PARAM
#GOALS = [np.array([-GLP, -GLP]), np.array([GLP, GLP]), np.array([GLP, -GLP]), np.array([-GLP, GLP])]
GOALS = [np.array([1.0, 1.0]), np.array([1.0, 1.8]), np.array([1.8, 1.0]), np.array([-GLP, GLP])]
CONFIG_TEMPLATE = {'num_steps': 200,
                   'observe_goal_lidar': False,
                   'observe_box_lidar': False,
                   'observe_qpos': True,
                   'observe_hazards': False,
                   'goal_locations': [(-GLP, -GLP)],
                   'robot_keepout': 1.0,
                   'robot_locations': [(0, 0)],
                   # 'robot_rot': 0 * 3.1415,
                   'lidar_max_dist': 5,
                   'task': 'goal',
                   'goal_size': 0.1,
                   'goal_keepout': 0.305,
                   'hazards_size': 0.4,
                   'hazards_keepout': 0.18,
                   'hazards_num': 0,
                   'hazards_cost': 0.0,
                   'hazards_locations': [(-HLP, -HLP), (HLP, HLP), (HLP, -HLP), (-HLP, HLP)],
                   'constrain_hazards': False,
                   'robot_base': 'xmls/point.xml',
                   'sensors_obs': ['magnetometer'],  # ['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
                   'lidar_num_bins': 8,
                   'placements_extents': [-2, -2, 2, 2],
                   }

# register(id='SafexpCustomEnvironment-v0',
#         entry_point='safety_gym.envs.mujoco:Engine',
#         kwargs={'config': config})

CUSTOM_ENV = 'SafexpCustomEnvironment-v0'
ENV_NAME = CUSTOM_ENV
NP_RANDOM, _ = seeding.np_random(None)


def _sample_goal_task():
    radius = NP_RANDOM.uniform(1, 2, size=(1, 1))[0][0]
    alpha = NP_RANDOM.uniform(0.0, 1.0, size=(1, 1)) * 2 * pi
    alpha = alpha[0][0]
    goal = np.array([radius * cos(alpha), radius * sin(alpha)])
    return goal


def _sample_start_position(goal, keepout):
    radius = NP_RANDOM.uniform(keepout, 2, size=(1, 1))[0][0]
    alpha = NP_RANDOM.uniform(0.0, 1.0, size=(1, 1)) * 2 * pi
    alpha = alpha[0][0]
    goal = np.array([goal[0] + (radius * cos(alpha)), goal[1] + (radius * sin(alpha))])
    return goal


def _array2label(arr):
    # This is out-of-ass method to turn the goal into a sensible label
    arr_flat = arr.flatten()
    arr_str = ""
    for a in arr_flat:
        arr_str += str(int(a ** 2 * 10e2))
    return arr_str


def register_set_goal(goal_idx):
    config = copy.deepcopy(CONFIG_TEMPLATE)

    goal = GOALS[goal_idx]  # _sample_goal_task() #GOALS[goal_idx]
    # start = _sample_start_position(goal, 1.0)
    config['goal_locations'] = [goal]
    # config['robot_locations'] = [start]
    # lbl = _array2label(goal) #+ _array2label(start)
    lbl = goal_idx
    env_name = f'SafexpCustomEnvironmentGoal{lbl}-v0'

    try:
        register(id=env_name,
                 entry_point='safety_gym.envs.mujoco:Engine',
                 kwargs={'config': config})
    except:
        pass

    return env_name


def make_env_list(args):
    device = torch.device("cpu")
    num_env = args.num_goal_samples
    env_list = []
    for ne in range(num_env):
        env_name = register_set_goal(ne)

        envs = make_vec_envs(env_name, np.random.randint(2 ** 32), 1,
                             args.gamma, None, device, allow_early_resets=True, normalize=args.norm_vectors)
        env_list.append(envs)

    return env_list
