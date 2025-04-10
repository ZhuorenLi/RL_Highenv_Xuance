import argparse
import numpy as np
from copy import deepcopy
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import SACDIS_Agent


def parse_args():
    parser = argparse.ArgumentParser("Use sac_dis of XuanCe for highway.")
    parser.add_argument("--env-id", type=str, default="highway-v0")
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--method", type=str, default="sac_dis")

    return parser.parse_args()


if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir="sac_dis_highway.yaml")
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    if configs.is_use_old_discrete_action:
        configs.action_type = "DiscreteMetaAction"
    else:
        configs.action_type = "NewDiscreteMetaAction"
    envs = make_envs(configs)
    Agent = SACDIS_Agent(config=configs, envs=envs)

    train_information = {"Deep learning toolbox": configs.dl_toolbox,
                         "Calculating device": configs.device,
                         "Algorithm": configs.agent,
                         "Environment": configs.env_name,
                         "Scenario": configs.env_id}
    for k, v in train_information.items():
        print(f"{k}: {v}")

    if not configs.test:
        Agent.train(configs.running_steps // configs.parallels)
        Agent.save_model("final_train_model.pth")
        print("Finish training!")
    else:
        def env_fn():
            configs.parallels = 1
            return make_envs(configs)

        Agent.load_model(path=Agent.model_dir_load, model='sac_dis')
        scores = Agent.test(env_fn, configs.test_episode)
        print(f"Mean Score: {np.mean(scores)}, Std: {np.std(scores)}")
        print("Finish testing.")

    Agent.finish()
