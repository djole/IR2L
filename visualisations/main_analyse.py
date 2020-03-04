import torch
import navigation_2d
from train_test_model import episode_rollout, update_policy
from model import init_model
from arguments import get_args
from visualisations.vis_path import vis_heatmap, vis_path


def test_reinforce(num_episodes=40, num_updates=5):

    args = get_args()

    env = navigation_2d.Navigation2DEnv(
        rm_nogo=args.rm_nogo, reduced_sampling=args.reduce_goals
    )
    new_task = env.sample_tasks(0)
    # new_task = env.sample_tasks()
    env.reset_task(new_task[0])

    model = init_model(env.observation_space.shape[0], env.action_space.shape[0], args)
    load_m = torch.load(
        "./trained_models/pulled_from_server/second_phase_instinct/20random_goals_instinct_module_LARGE_danger_zone_outer_goal_sampling/individual_603.pt"
    )

    st_dct = load_m[0].state_dict()
    # del st_dct["sigma"]
    model.load_state_dict(st_dct)

    optimizer = torch.optim.Adam(model.get_combinator_params(), load_m[1])

    rewards = []
    action_log_probs = []

    fitness_list = []

    for u_idx in range(num_updates):
        ### Train
        model.controller.deterministic = False
        rollout_info = []
        for ne in range(num_episodes):
            _, reached, (rewards_, action_log_probs_), vis = episode_rollout(
                model, env, True
            )
            rollout_info.append(vis)
            rewards.extend(rewards_)
            action_log_probs.extend(action_log_probs_)

        assert len(rewards) > 1 and len(action_log_probs) > 1
        update_policy(optimizer, args, rewards, action_log_probs)
        rewards.clear()
        action_log_probs.clear()

        ### evaluate
        model.controller.deterministic = True
        fitness, reached, _, vis_info = episode_rollout(model, env, vis=True)
        #### Adapt sigma
        model.controller.sigma = torch.nn.Parameter(model.controller.sigma - 0.01)
        print("exploitation reward = {}".format(fitness))
        fitness_list.append(fitness)
        vis_path(rollout_info)
        # for s in range(1, 10):
        #    vis_path([vis_info], "eval_{}_{}".format(u_idx, s), s)
        #vis_path([vis_info])  # , "eval_{}".format(u_idx))
        vis_heatmap(model)
        # vis_instinct_action(model)

    avg_exploitation_fitness = sum(fitness_list) / num_updates
    ret_fit = avg_exploitation_fitness
    return ret_fit, reached, vis_info


if __name__ == "__main__":
    test_reinforce(num_updates=20)
