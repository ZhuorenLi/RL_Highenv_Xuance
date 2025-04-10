import os
import argparse
from copy import deepcopy
import numpy as np
import torch.optim
from gymnasium import spaces

from xuance import get_arguments
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.policies import MH3RL_Policy
from xuance.torch.utils import ActivationFunctions

from xuance.torch.agents import  get_total_iters, MH3RL_Agent


def parse_args():
    parser = argparse.ArgumentParser("Use MH3RL In Highway_env,Apply to Hybrid Action")
    parser.add_argument("--method", type=str, default="MH3RL")
    parser.add_argument("--env", type=str, default="highway")
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--config", type=str, default="./MH3RL_highway_config.yaml")

    return parser.parse_args()


def run(args):
    agent_name = args.agent
    set_seed(args.seed)

    # prepare directories for results
    args.model_dir = os.path.join(os.getcwd(), args.model_dir, args.env_id)
    args.log_dir = os.path.join(args.log_dir, args.env_id)

    # build environments
    envs = make_envs(args)
    args.observation_space = envs.observation_space
    args.action_space = envs.action_space
    n_envs = envs.num_envs

    # prepare the Policy
    policy = MH3RL_Policy(observation_space=args.observation_space,
                          action_space=args.action_space,
                          up_actor_hidden_size=args.up_actor_hidden_size,
                          up_qnetwork_hidden_size=args.up_qnetwork_hidden_size,
                          low_actor_hidden_size=args.low_actor_hidden_size,
                          low_qnetwork_hidden_size=args.low_qnetwork_hidden_size,
                          normalize=None,
                          initialize=torch.nn.init.orthogonal_,
                          activation=ActivationFunctions[args.activation],
                          device=args.device,
                          low_extend_obs_multiple=args.low_extend_obs_multiple)

    # prepare agent
    up_actor_optimizer = torch.optim.Adam(policy.up_actor.parameters(), args.up_actor_learning_rate)
    up_qnetwork_optimizer = torch.optim.Adam(policy.up_qnetwork.parameters(), args.up_qnetwork_learning_rate)
    up_actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(up_actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                              total_iters=get_total_iters(agent_name, args))
    up_qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(up_qnetwork_optimizer, start_factor=1.0, end_factor=0.25,
                                                                 total_iters=get_total_iters(agent_name, args))
    low_actor_optimizer = torch.optim.Adam(policy.low_actor.parameters(), args.low_actor_learning_rate)
    low_qnetwork_optimizer = torch.optim.Adam(policy.low_critic.parameters(), args.low_qnetwork_learning_rate)
    low_actor_lr_scheduler = torch.optim.lr_scheduler.LinearLR(low_actor_optimizer, start_factor=1.0, end_factor=0.25,
                                                               total_iters=get_total_iters(agent_name, args))
    low_qnetwork_lr_scheduler = torch.optim.lr_scheduler.LinearLR(low_qnetwork_optimizer, start_factor=1.0, end_factor=0.25,
                                                                total_iters=get_total_iters(agent_name, args))
    agent = MH3RL_Agent(config=args,
                        envs=envs,
                        policy=policy,
                        up_optimizer=[up_actor_optimizer, up_qnetwork_optimizer],
                        up_scheduler=[up_actor_lr_scheduler, up_qnetwork_lr_scheduler],
                        low_optimizer=[low_actor_optimizer, low_qnetwork_optimizer],
                        low_scheduler=[low_actor_lr_scheduler, low_qnetwork_lr_scheduler],
                        device=args.device)

    # start running
    envs.reset()

    if not args.test:  # train the model without testing
        n_train_steps = args.running_steps // n_envs
        agent.train(n_train_steps)
        agent.save_model("final_train_model.pth")
        print("Finish training!")
    else:  # test a trained model
        def env_fn():
            args_test = deepcopy(args)
            args_test.parallels = 1
            return make_envs(args_test)

        agent.render = True
        agent.load_model(agent.model_dir_load, args.seed, dir_name='MH3RL')
        scores = agent.test(env_fn, args.test_episode)
        print("Finish testing.")

    # the end.
    envs.close()
    agent.finish()


if __name__ == "__main__":
    parser = parse_args()
    args = get_arguments(method=parser.method,
                         env=parser.env,
                         env_id=parser.env_id,
                         config_path=parser.config,
                         parser_args=parser)
    run(args)
