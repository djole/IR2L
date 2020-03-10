import gym
import safety_gym
from gym.envs.registration import register

HAZARD_LOC_PARAM = 1
HLP = HAZARD_LOC_PARAM

GOAL_LOC_PARAM = 1.8
GLP = GOAL_LOC_PARAM
GOALS = [(-GLP, -GLP), (GLP, GLP), (GLP, -GLP), (-GLP, GLP)]
config = {'observe_goal_lidar': False,
          'observe_box_lidar': False,
          'observe_qpos': True,
          'observe_hazards': True,
          'goal_locations': [(-GLP, -GLP)],
          'robot_locations': [(0, 0)],
          'lidar_max_dist': 3,
          'lidar_num_bins': 16,
          'task': 'goal',
          'goal_size': 0.1,
          'goal_keepout': 0.305,
          'hazards_size': 0.4,
          'hazards_keepout': 0.18,
          'hazards_num': 4,
          'hazards_cost': 10.0,
          'hazards_locations': [(-HLP, -HLP), (HLP, HLP), (HLP, -HLP), (-HLP, HLP)],
          'constrain_hazards': True,
          'robot_base': 'xmls/point.xml',
          'sensors_obs': [], #['accelerometer', 'velocimeter', 'gyro', 'magnetometer'],
          'lidar_num_bins': 8,
          'placements_extents': [-2, -2, 2, 2]}

#register(id='SafexpCustomEnvironment-v0',
#         entry_point='safety_gym.envs.mujoco:Engine',
#         kwargs={'config': config})

CUSTOM_ENV = 'SafexpCustomEnvironment-v0'
ENV_NAME = CUSTOM_ENV


def register_set_goal(goal_idx):
    goal = GOALS[goal_idx]
    config['goal_locations'] = [goal]

    env_name = f'SafexpCustomEnvironmentGoal{goal_idx}-v0'

    try:
        register(id=env_name,
                 entry_point='safety_gym.envs.mujoco:Engine',
                 kwargs={'config': config})
    except:
        pass

    return env_name