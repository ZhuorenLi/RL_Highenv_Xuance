import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import Attention_MPDQN_Agent
from xuance.extend_display.attention_display import Graphics
from high_d.src.highD_test import *


def parse_args():
    parser = argparse.ArgumentParser("Use att_mpdqn of XuanCe for highway.")
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--method", type=str, default="att_mpdqn")

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="att_mpdqn_highway.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    unwrapped_env = envs.envs[0].unwrapped.env   # for attention matrix visualization
    Agent = Attention_MPDQN_Agent(config=configs, envs=envs)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if configs.is_display_attention_matrix:
        unwrapped_env.unwrapped.viewer.set_agent_display(
            lambda agent_surface, sim_surface: Graphics.display(Agent, agent_surface,
                                                                sim_surface))

    if not configs.test:
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("final_train_model.pth")
        print("Finish training!")
    else:
        def env_fn():
            configs.parallels = 1
            return make_envs(configs)

        Agent.load_model(path=Agent.model_dir_load, model='att_mpdqn')
        if configs.use_HighD_test:
            configs.test_episode = 1  # 若使用HighD数据集进行测试，则test_episode需设置为1
            configs.env_id = "highd-v0"
            high_tester = HighD_Tester(configs)
            high_tester.highD_test(Agent)
        else:
            scores = Agent.test(env_fn, configs.test_episode)
            print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
        print("Finish testing.")

    Agent.finish()