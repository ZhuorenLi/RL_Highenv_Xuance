import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import Attention_SAC_Agent
from xuance.extend_display.attention_display import Graphics


def parse_args():
    parser = argparse.ArgumentParser("Example of XuanCe: sac_CHD for highway.")
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--method", type=str, default="attn_sac_CHD")

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="attn_sac_CHD_highway.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    unwrapped_env = envs.envs[0].unwrapped.env   # for attention matrix visualization
    Agent = Attention_SAC_Agent(config=configs, envs=envs)

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

        Agent.load_model(path=Agent.model_dir_load, model='attn_sac_CHD')
        scores = Agent.test(env_fn, configs.test_episode)
        print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
        print("Finish testing.")

    Agent.finish()
